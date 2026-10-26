use crate::data::{self, Example, IGNORE, KINDS, LABELS, Span};
use anyhow::{Context, Result, ensure};
use bert_burn::{
    data::BertInferenceBatch,
    loader::{load_model_config, load_pretrained},
    model::{BertModel, BertModelConfig},
};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamWConfig, GradientsParams, ModuleOptimizer},
    store::ModuleRecord,
    tensor::{Bool, Device, Int, Tensor, TensorData},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, time::Instant};
use tokenizers::Tokenizer;

#[derive(Module, Debug)]
pub struct Tagger {
    pub encoder: BertModel,
    pub classifier: Linear,
}

// Replace ignored indices BEFORE gather; an out-of-range ignore label is unsafe
// in Burn's stock cross-entropy, which gathers before applying its padding mask.
fn token_loss(logits: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
    let ignored = targets.clone().equal_elem(IGNORE);
    let safe_targets = targets.mask_fill(ignored.clone(), 0);
    let weights = ignored.bool_not().float();
    let count = weights.clone().sum();
    let selected = burn::tensor::activation::log_softmax(logits, 1)
        .gather(1, safe_targets.unsqueeze_dim::<2>(1))
        .squeeze_dim::<1>(1);
    (selected.neg() * weights).sum() / count
}
impl Tagger {
    pub fn init(config: &BertModelConfig, device: &Device) -> Self {
        Self {
            encoder: config.init(device),
            classifier: LinearConfig::new(config.hidden_size, LABELS).init(device),
        }
    }
    pub fn forward(&self, batch: BertInferenceBatch) -> Tensor<3> {
        self.classifier
            .forward(self.encoder.forward(batch).hidden_states)
    }
    fn probe(&self) -> Tensor<2> {
        self.encoder.encoder.layers[0].mha.query.weight.val()
    }
}

pub fn device(name: &str) -> Result<Device> {
    crate::storage::configure_runtime()?;
    match name {
        "metal" => Ok(Device::metal(Default::default())),
        "cpu" => Ok(Device::flex()),
        _ => anyhow::bail!("backend must be metal or cpu"),
    }
}

#[derive(Clone)]
struct Encoded {
    ids: Vec<i64>,
    offsets: Vec<(usize, usize)>,
    labels: Vec<i64>,
}

pub struct Engine {
    pub config: BertModelConfig,
    tokenizer: Tokenizer,
    max_tokens: usize,
}
impl Engine {
    pub fn new(model_dir: &Path, max_tokens: usize) -> Result<Self> {
        ensure!(
            (8..=512).contains(&max_tokens),
            "max tokens must be between 8 and 512"
        );
        let config = load_model_config(model_dir.join("burn-config.json"))?;
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|_| anyhow::anyhow!("cannot load local tokenizer"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|_| anyhow::anyhow!("cannot configure tokenizer"))?;
        Ok(Self {
            config,
            tokenizer,
            max_tokens,
        })
    }
    fn encode(&self, sample: &Example) -> Result<Encoded> {
        let encoded = self
            .tokenizer
            .encode(sample.text.as_str(), true)
            .map_err(|_| anyhow::anyhow!("tokenization failed"))?;
        ensure!(
            encoded.len() <= self.max_tokens,
            "document exceeds token limit; training does not silently truncate"
        );
        let mut labels = vec![0i64; encoded.len()];
        let offsets = encoded.get_offsets().to_vec();
        let mut previous_entity = None;
        for (i, &(start, end)) in offsets.iter().enumerate() {
            if start == end {
                labels[i] = IGNORE;
                previous_entity = None;
                continue;
            }
            let entity = sample
                .spans
                .iter()
                .position(|span| span.start <= start && end <= span.end);
            if let Some(j) = entity {
                labels[i] = (1
                    + sample.spans[j].kind.index() * 2
                    + usize::from(previous_entity == Some(j))) as i64;
            } else {
                ensure!(
                    !sample
                        .spans
                        .iter()
                        .any(|span| start < span.end && span.start < end),
                    "entity boundary splits a tokenizer token"
                );
            }
            previous_entity = entity;
        }
        // Every annotated entity must be represented, including unknown-token cases.
        for span in &sample.spans {
            ensure!(
                offsets
                    .iter()
                    .any(|&(s, e)| s < e && span.start <= s && e <= span.end),
                "entity has no aligned token"
            );
        }
        Ok(Encoded {
            ids: encoded.get_ids().iter().map(|&x| x as i64).collect(),
            offsets,
            labels,
        })
    }
    fn batch(&self, samples: &[Encoded], device: &Device) -> (BertInferenceBatch, Tensor<1, Int>) {
        let n = samples.len();
        let length = samples.iter().map(|x| x.ids.len()).max().unwrap();
        let mut ids = vec![0i64; n * length];
        let mut padding = vec![true; n * length];
        let mut labels = vec![IGNORE; n * length];
        for (i, sample) in samples.iter().enumerate() {
            for j in 0..sample.ids.len() {
                ids[i * length + j] = sample.ids[j];
                padding[i * length + j] = false;
                labels[i * length + j] = sample.labels[j];
            }
        }
        (
            BertInferenceBatch {
                tokens: Tensor::<2, Int>::from_data(TensorData::new(ids, [n, length]), device),
                mask_pad: Tensor::<2, Bool>::from_data(
                    TensorData::new(padding, [n, length]),
                    device,
                ),
            },
            Tensor::<1, Int>::from_data(TensorData::new(labels, [n * length]), device),
        )
    }
    pub fn predict(
        &self,
        model: &Tagger,
        examples: &[Example],
        device: &Device,
        batch_size: usize,
    ) -> Result<Vec<Vec<Span>>> {
        ensure!(batch_size > 0, "batch size must be positive");
        let encoded: Vec<_> = examples
            .iter()
            .map(|x| {
                self.encode(&Example {
                    spans: vec![],
                    ..x.clone()
                })
            })
            .collect::<Result<_>>()?;
        let mut result = Vec::new();
        for chunk in encoded.chunks(batch_size) {
            let (batch, _) = self.batch(chunk, device);
            let logits = model.forward(batch);
            ensure!(
                logits
                    .clone()
                    .is_finite()
                    .all()
                    .into_data()
                    .try_into_vec_as::<bool>()?[0],
                "non-finite inference output"
            );
            let output = logits
                .argmax(2)
                .squeeze_dim::<2>(2)
                .into_data()
                .try_into_vec_as::<i64>()?;
            let length = chunk.iter().map(|x| x.ids.len()).max().unwrap();
            for (i, item) in chunk.iter().enumerate() {
                result.push(decode(
                    &item.offsets,
                    &output[i * length..i * length + item.ids.len()],
                ));
            }
        }
        Ok(result)
    }
    pub fn predict_text(&self, model: &Tagger, text: &str, device: &Device) -> Result<Vec<Span>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        // Overlapping windows cover all input. No text is dropped at the model limit.
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let width = self.max_tokens.saturating_sub(2) / 2;
        let stride = (width / 2).max(1);
        let mut spans = Vec::new();
        let mut start = 0;
        while start + 1 < boundaries.len() {
            let end = (start + width).min(boundaries.len() - 1);
            let offset = boundaries[start];
            let example = Example {
                id: String::new(),
                group: String::new(),
                template: String::new(),
                text: text[offset..boundaries[end]].into(),
                spans: vec![],
            };
            for mut span in self.predict(model, &[example], device, 1)?.remove(0) {
                span.start += offset;
                span.end += offset;
                spans.push(span);
            }
            if end == boundaries.len() - 1 {
                break;
            }
            start += stride;
        }
        Ok(crate::rules::merge(spans))
    }
    pub fn reference_check(&self, model_dir: &Path, device: &Device) -> Result<serde_json::Value> {
        #[derive(Deserialize)]
        struct Reference {
            input_ids: Vec<Vec<i64>>,
            attention_mask: Vec<Vec<i64>>,
            hidden_states: Vec<Vec<Vec<f32>>>,
        }
        let reference: Reference =
            serde_json::from_slice(&fs::read(model_dir.join("reference.json"))?)?;
        let mut model = self.config.init(device);
        load_pretrained(&mut model, model_dir.join("model.safetensors"))?;
        let n = reference.input_ids.len();
        let length = reference.input_ids[0].len();
        let batch = BertInferenceBatch {
            tokens: Tensor::<2, Int>::from_data(
                TensorData::new(reference.input_ids.concat(), [n, length]),
                device,
            ),
            mask_pad: Tensor::<2, Bool>::from_data(
                TensorData::new(
                    reference
                        .attention_mask
                        .concat()
                        .iter()
                        .map(|&x| x == 0)
                        .collect::<Vec<_>>(),
                    [n, length],
                ),
                device,
            ),
        };
        let actual = model
            .valid()
            .forward(batch)
            .hidden_states
            .into_data()
            .try_to_vec::<f32>()?;
        let expected: Vec<f32> = reference
            .hidden_states
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        ensure!(actual.len() == expected.len(), "reference shape mismatch");
        ensure!(
            actual.iter().chain(&expected).all(|x| x.is_finite()),
            "non-finite reference output"
        );
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        ensure!(
            max_error.is_finite() && max_error < 0.003,
            "pretrained reference mismatch: max error {max_error}"
        );
        Ok(
            serde_json::json!({"max_absolute_error":max_error,"values":actual.len(),"tolerance":0.003,"reference":"PyTorch eager CPU"}),
        )
    }
}

fn decode(offsets: &[(usize, usize)], labels: &[i64]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    let mut active = false;
    for (&(start, end), &label) in offsets.iter().zip(labels) {
        if start == end || label <= 0 || label >= LABELS as i64 {
            active = false;
            continue;
        }
        let kind = KINDS[(label as usize - 1) / 2];
        let inside = label % 2 == 0;
        if inside
            && active
            && spans
                .last()
                .is_some_and(|s| s.kind == kind && s.end <= start)
        {
            spans.last_mut().unwrap().end = end;
        } else {
            spans.push(Span { start, end, kind });
        }
        active = true;
    }
    crate::rules::merge(spans)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub seed: u64,
    pub max_tokens: usize,
    pub backend: String,
}
#[derive(Serialize, Deserialize)]
struct State {
    completed_epochs: usize,
    steps: usize,
    best_f1: f64,
    train_fingerprint: String,
    dev_fingerprint: String,
    options: TrainOptions,
}

pub fn load_tagger(engine: &Engine, checkpoint: &Path, device: &Device) -> Result<Tagger> {
    Ok(Tagger::init(&engine.config, device)
        .load_record(ModuleRecord::load(checkpoint)?)
        .valid())
}

pub fn train(
    model_dir: &Path,
    dataset: &Path,
    run: &Path,
    options: TrainOptions,
    resume: bool,
) -> Result<serde_json::Value> {
    ensure!(
        options.epochs > 0 && options.batch_size > 0 && options.lr.is_finite() && options.lr > 0.0,
        "invalid training options"
    );
    let device = device(&options.backend)?.autodiff();
    device.seed(options.seed);
    let engine = Engine::new(model_dir, options.max_tokens)?;
    let train = data::load(&dataset.join("train.jsonl"))?;
    let dev = data::load(&dataset.join("dev.jsonl"))?;
    data::assert_disjoint(&train, &dev)?;
    let encoded: Vec<_> = train
        .iter()
        .map(|x| engine.encode(x))
        .collect::<Result<_>>()?;
    let mut model = Tagger::init(&engine.config, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_epsilon(1e-8)
        .init();
    let mut state = State {
        completed_epochs: 0,
        steps: 0,
        best_f1: -1.0,
        train_fingerprint: data::fingerprint(&train),
        dev_fingerprint: data::fingerprint(&dev),
        options: options.clone(),
    };
    if resume {
        let saved: State = serde_json::from_slice(&fs::read(run.join("state.json"))?)?;
        ensure!(
            saved.train_fingerprint == state.train_fingerprint
                && saved.dev_fingerprint == state.dev_fingerprint,
            "resume dataset changed"
        );
        ensure!(
            saved.options.lr == options.lr
                && saved.options.batch_size == options.batch_size
                && saved.options.seed == options.seed
                && saved.options.max_tokens == options.max_tokens,
            "resume hyperparameters changed"
        );
        let snapshot = run.join(format!("checkpoints/epoch-{:04}", saved.completed_epochs));
        model = model
            .load_record(ModuleRecord::load(snapshot.join("model.bpk"))?)
            .train();
        optimizer = optimizer.load(snapshot.join("optimizer.bpk"))?;
        state = saved;
    } else {
        ensure!(
            !run.join("state.json").exists(),
            "run exists; use --resume or a new run directory"
        );
        load_pretrained(&mut model.encoder, model_dir.join("model.safetensors"))?;
    }
    fs::create_dir_all(run)?;
    let initial_probe = model
        .probe()
        .clone()
        .slice([0..4, 0..4])
        .into_data()
        .try_to_vec::<f32>()?;
    let timer = Instant::now();
    let mut gradient_l1 = None;
    let mut history = Vec::new();
    for epoch in state.completed_epochs..state.completed_epochs + options.epochs {
        let mut indices: Vec<_> = (0..encoded.len()).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(
            options.seed.wrapping_add(epoch as u64),
        ));
        let mut total_loss = 0.0;
        let mut batches = 0;
        for (batch_index, indices) in indices.chunks(options.batch_size).enumerate() {
            device.seed(
                options
                    .seed
                    .wrapping_add((epoch as u64) * 100000 + batch_index as u64),
            );
            let samples: Vec<_> = indices.iter().map(|&i| encoded[i].clone()).collect();
            let (batch, targets) = engine.batch(&samples, &device);
            let logits = model.forward(batch).reshape([targets.dims()[0], LABELS]);
            let loss = token_loss(logits, targets);
            let value = loss.clone().into_data().try_to_vec::<f32>()?[0];
            ensure!(value.is_finite(), "non-finite training loss");
            let grads = loss.backward();
            if gradient_l1.is_none() {
                let gradient = model
                    .probe()
                    .grad(&grads)
                    .context("encoder gradient is missing")?;
                let norm = gradient.abs().sum().into_data().try_to_vec::<f32>()?[0];
                ensure!(
                    norm.is_finite() && norm > 0.0,
                    "encoder gradient is zero or non-finite"
                );
                gradient_l1 = Some(norm);
            }
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(options.lr, model, grads);
            total_loss += value as f64;
            batches += 1;
            state.steps += 1;
            if batches % 10 == 0 {
                eprintln!("epoch={} batch={} loss={value:.4}", epoch + 1, batches);
            }
        }
        let valid = model.valid();
        let plain = crate::model::device(&options.backend)?;
        let predictions = engine.predict(&valid, &dev, &plain, options.batch_size)?;
        let metrics = crate::eval::evaluate(&dev, &predictions);
        let f1 = metrics["entity_exact"]["f1"].as_f64().unwrap();
        let summary = serde_json::json!({"epoch":epoch+1,"mean_training_loss":total_loss/batches as f64,"dev":metrics,"elapsed_seconds":timer.elapsed().as_secs_f64()});
        eprintln!(
            "epoch={} mean_loss={:.4} dev_f1={f1:.4}",
            epoch + 1,
            total_loss / batches as f64
        );
        history.push(summary);
        if f1 > state.best_f1 {
            state.best_f1 = f1;
            model.clone().into_record().save(run.join("best.bpk"))?;
        }
        state.completed_epochs = epoch + 1;
        let snapshot = run.join(format!("checkpoints/epoch-{:04}", state.completed_epochs));
        fs::create_dir_all(&snapshot)?;
        model
            .clone()
            .into_record()
            .save(snapshot.join("model.bpk"))?;
        optimizer.save(snapshot.join("optimizer.bpk"))?;
        fs::write(
            run.join("state.json.tmp"),
            serde_json::to_vec_pretty(&state)?,
        )?;
        fs::rename(run.join("state.json.tmp"), run.join("state.json"))?;
    }
    let final_probe = model
        .probe()
        .slice([0..4, 0..4])
        .into_data()
        .try_to_vec::<f32>()?;
    let delta = initial_probe
        .iter()
        .zip(&final_probe)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    ensure!(
        final_probe.iter().all(|x| x.is_finite()) && delta > 0.0,
        "encoder parameter probe is invalid or did not change"
    );
    let resume_check = verify_next_step(
        &engine,
        &model,
        &mut optimizer,
        &encoded[..encoded.len().min(options.batch_size)],
        &device,
        options.lr,
        &run.join(format!("checkpoints/epoch-{:04}", state.completed_epochs)),
    )?;
    let report = serde_json::json!({"scope":"local_training_run","backend":options.backend,"parameter_count":model.num_params(),"encoder_gradient_l1":gradient_l1,"encoder_probe_max_change":delta,"resumed":resume,"resume_next_step_check":resume_check,"completed_epochs":state.completed_epochs,"steps":state.steps,"elapsed_seconds":timer.elapsed().as_secs_f64(),"history":history});
    fs::write(
        run.join(if resume {
            "resume-report.json"
        } else {
            "training-report.json"
        }),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

/// Compare a live optimizer's next update with a fresh model + optimizer restored from disk.
/// Both branches are discarded; these diagnostic steps never enter the saved training state.
fn verify_next_step(
    engine: &Engine,
    model: &Tagger,
    optimizer: &mut ModuleOptimizer,
    samples: &[Encoded],
    device: &Device,
    lr: f64,
    snapshot: &Path,
) -> Result<serde_json::Value> {
    fn step(
        engine: &Engine,
        model: Tagger,
        optimizer: &mut ModuleOptimizer,
        samples: &[Encoded],
        device: &Device,
        lr: f64,
    ) -> Tagger {
        device.seed(93021);
        let (batch, targets) = engine.batch(samples, device);
        let logits = model.forward(batch).reshape([targets.dims()[0], LABELS]);
        let loss = token_loss(logits, targets);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        optimizer.step(lr, model, grads)
    }
    let restored = Tagger::init(&engine.config, device)
        .load_record(ModuleRecord::load(snapshot.join("model.bpk"))?)
        .train();
    let mut restored_optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_epsilon(1e-8)
        .init()
        .load(snapshot.join("optimizer.bpk"))?;
    let live = step(engine, model.clone(), optimizer, samples, device, lr);
    let restored = step(
        engine,
        restored,
        &mut restored_optimizer,
        samples,
        device,
        lr,
    );
    let mut max_error = 0f32;
    let mut compared = 0;
    for (a, b) in [
        (live.probe(), restored.probe()),
        (
            live.classifier.weight.val(),
            restored.classifier.weight.val(),
        ),
    ] {
        let a = a.into_data().try_into_vec_as::<f32>()?;
        let b = b.into_data().try_into_vec_as::<f32>()?;
        ensure!(
            a.len() == b.len() && a.iter().chain(&b).all(|x| x.is_finite()),
            "invalid resume comparison"
        );
        compared += a.len();
        max_error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(max_error, f32::max);
    }
    ensure!(
        max_error < 0.00001,
        "restored optimizer next step diverges: {max_error}"
    );
    Ok(
        serde_json::json!({"compared_parameters":compared,"encoder_and_head_max_absolute_error":max_error,"tolerance":0.00001,"diagnostic_updates_saved":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decoding_handles_unicode_and_orphan_inside() {
        let spans = decode(&[(0, 0), (0, 3), (3, 6), (6, 9), (0, 0)], &[0, 2, 2, 0, 0]);
        assert_eq!(
            spans,
            vec![Span {
                start: 0,
                end: 6,
                kind: KINDS[0]
            }]
        );
    }
    #[test]
    fn padding_and_special_labels_do_not_change_loss() {
        let device = Device::flex();
        let first: Tensor<2> = Tensor::from_data([[1.0f32, 0.0]], &device);
        let first_target: Tensor<1, Int> = Tensor::from_data([0i64], &device);
        let padded: Tensor<2> = Tensor::from_data([[1.0f32, 0.0], [100.0, -100.0]], &device);
        let padded_target: Tensor<1, Int> = Tensor::from_data([0i64, IGNORE], &device);
        let a = token_loss(first, first_target)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        let b = token_loss(padded, padded_target)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!((a - b).abs() < 1e-6);
    }
    #[test]
    fn ignored_token_has_zero_gradient_but_o_label_is_trained() {
        let device = Device::flex().autodiff();
        let logits: Tensor<2> =
            Tensor::from_data([[1.0f32, 0.0], [100.0, -100.0]], &device).require_grad();
        let targets: Tensor<1, Int> = Tensor::from_data([0i64, IGNORE], &device);
        let grads = token_loss(logits.clone(), targets).backward();
        let values = logits
            .grad(&grads)
            .unwrap()
            .into_data()
            .try_into_vec_as::<f32>()
            .unwrap();
        assert!(values[0].abs() > 0.0 && values[1].abs() > 0.0);
        assert_eq!(&values[2..], &[0.0, 0.0]);
    }
}
