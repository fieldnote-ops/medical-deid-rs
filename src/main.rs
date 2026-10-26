mod data;
mod eval;
mod model;
mod rules;
mod storage;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(
    version,
    about = "Local Chinese medical text redaction, training and evaluation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Probe {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
    Audit {
        #[arg(long)]
        dataset: PathBuf,
    },
    Train {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 0.0001)]
        lr: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value = "metal")]
        backend: String,
        #[arg(long)]
        resume: bool,
    },
    Evaluate {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "metal")]
        backend: String,
    },
    Redact {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "metal")]
        backend: String,
    },
    Predict {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value = "metal")]
        backend: String,
    },
}

fn external_output(path: &Path) -> Result<()> {
    storage::validate_output(path)
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe { model_dir, backend } => {
            let engine = model::Engine::new(&model_dir, 128)?;
            let report = engine.reference_check(&model_dir, &model::device(&backend)?)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Audit { dataset } => {
            let train = data::load(&dataset.join("train.jsonl"))?;
            let dev = data::load(&dataset.join("dev.jsonl"))?;
            let test = data::load(&dataset.join("test.jsonl"))?;
            data::assert_disjoint(&train, &dev)?;
            data::assert_disjoint(&train, &test)?;
            data::assert_disjoint(&dev, &test)?;
            println!(
                "{}",
                serde_json::json!({"train_documents":train.len(),"dev_documents":dev.len(),"test_documents":test.len(),"group_and_exact_text_overlap":false,"scope":"partition_overlap_audit"})
            );
        }
        Command::Train {
            model_dir,
            dataset,
            run,
            epochs,
            batch_size,
            lr,
            seed,
            max_tokens,
            backend,
            resume,
        } => {
            external_output(&run)?;
            let report = model::train(
                &model_dir,
                &dataset,
                &run,
                model::TrainOptions {
                    epochs,
                    batch_size,
                    lr,
                    seed,
                    max_tokens,
                    backend,
                },
                resume,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Evaluate {
            dataset,
            report,
            model_dir,
            checkpoint,
            backend,
        } => {
            external_output(&report)?;
            let examples = data::load(&dataset)?;
            let rule_predictions: Vec<_> =
                examples.iter().map(|x| rules::detect(&x.text)).collect();
            let mut result = serde_json::json!({"dataset_sha256":data::fingerprint(&examples),"rules":eval::evaluate(&examples,&rule_predictions)});
            ensure!(
                model_dir.is_some() == checkpoint.is_some(),
                "model-dir and checkpoint must be supplied together"
            );
            if let (Some(model_dir), Some(checkpoint)) = (model_dir, checkpoint) {
                let device = model::device(&backend)?;
                let engine = model::Engine::new(&model_dir, 512)?;
                let model = model::load_tagger(&engine, &checkpoint, &device)?;
                let predictions = engine.predict(&model, &examples, &device, 8)?;
                let combined: Vec<_> = predictions
                    .iter()
                    .zip(&rule_predictions)
                    .map(|(a, b)| rules::merge(a.iter().chain(b).cloned().collect()))
                    .collect();
                result["model"] = eval::evaluate(&examples, &predictions);
                result["combined"] = eval::evaluate(&examples, &combined);
                let reloaded = model::load_tagger(&engine, &checkpoint, &device)?;
                ensure!(
                    engine.predict(&reloaded, &examples[..examples.len().min(8)], &device, 8)?
                        == predictions[..examples.len().min(8)],
                    "checkpoint reload predictions changed"
                );
                result["checkpoint_reload_predictions_equal"] = true.into();
            }
            if let Some(parent) = report.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&report, serde_json::to_vec_pretty(&result)?)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Redact {
            input,
            model_dir,
            checkpoint,
            backend,
        } => {
            ensure!(
                model_dir.is_some() == checkpoint.is_some(),
                "model-dir and checkpoint must be supplied together"
            );
            let text = fs::read_to_string(input)?;
            let mut spans = rules::detect(&text);
            if let (Some(model_dir), Some(checkpoint)) = (model_dir, checkpoint) {
                let device = model::device(&backend)?;
                let engine = model::Engine::new(&model_dir, 128)?;
                let model = model::load_tagger(&engine, &checkpoint, &device)?;
                spans.extend(engine.predict_text(&model, &text, &device)?);
            }
            let spans = rules::merge(spans);
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"schema_version":1,"redacted_text":rules::redact(&text,&spans)?,"spans":spans,"offset_unit":"utf8_bytes_end_exclusive","review_required":true})
                )?
            );
        }
        Command::Predict {
            dataset,
            output,
            model_dir,
            checkpoint,
            backend,
        } => {
            external_output(&output)?;
            let examples = data::load_inputs(&dataset)?;
            let device = model::device(&backend)?;
            let engine = model::Engine::new(&model_dir, 512)?;
            let model = model::load_tagger(&engine, &checkpoint, &device)?;
            engine.predict_text(&model, &examples[0].text, &device)?;
            let mut rows = String::new();
            let mut latencies = Vec::new();
            for example in &examples {
                let start = std::time::Instant::now();
                let spans = engine.predict_text(&model, &example.text, &device)?;
                latencies.push(start.elapsed().as_secs_f64() * 1000.0);
                let rules_spans = rules::detect(&example.text);
                let combined = rules::merge(spans.iter().chain(&rules_spans).cloned().collect());
                rows.push_str(&serde_json::to_string(&serde_json::json!({
                    "schema_version":1,"id":example.id,"spans":spans,"rules_spans":rules_spans,
                    "combined_spans":combined,"redacted_text":rules::redact(&example.text,&combined)?,
                    "offset_unit":"utf8_bytes_end_exclusive","review_required":true
                }))?);
                rows.push('\n');
            }
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&output, rows)?;
            use burn::module::Module;
            let metadata = serde_json::json!({"backend":backend,"parameter_count":model.num_params(),"documents":examples.len(),"latency_ms_per_document":latencies,"window_characters":255,"stride_characters":127,"batch_size":1,"warmup_documents":1,"dataset_sha256":data::fingerprint(&examples),"gold_labels_used_for_prediction":false});
            fs::write(
                output.with_extension("metadata.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"documents":examples.len(),"prediction_output":output})
            );
        }
    }
    Ok(())
}
