use crate::data::{Example, KINDS, Span};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Default, Serialize)]
pub struct Counts {
    pub gold: usize,
    pub predicted: usize,
    pub exact: usize,
}
impl Counts {
    fn metrics(&self) -> serde_json::Value {
        let precision = ratio(self.exact, self.predicted);
        let recall = ratio(self.exact, self.gold);
        serde_json::json!({"gold":self.gold,"predicted":self.predicted,"exact":self.exact,"precision":precision,"recall":recall,"f1":if precision+recall>0.0 {2.0*precision*recall/(precision+recall)} else {0.0}})
    }
}
fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

pub fn evaluate(examples: &[Example], predictions: &[Vec<Span>]) -> serde_json::Value {
    assert_eq!(examples.len(), predictions.len());
    let mut totals = Counts::default();
    let mut by_kind: Vec<Counts> = KINDS.iter().map(|_| Counts::default()).collect();
    let (
        mut residual_docs,
        mut sensitive_chars,
        mut covered_chars,
        mut nonsensitive_chars,
        mut overmasked_chars,
    ) = (0, 0, 0, 0, 0);
    for (sample, spans) in examples.iter().zip(predictions) {
        let gold: HashSet<_> = sample.spans.iter().collect();
        let pred: HashSet<_> = spans.iter().collect();
        totals.gold += gold.len();
        totals.predicted += pred.len();
        totals.exact += gold.intersection(&pred).count();
        for (i, kind) in KINDS.iter().enumerate() {
            by_kind[i].gold += gold.iter().filter(|s| s.kind == *kind).count();
            by_kind[i].predicted += pred.iter().filter(|s| s.kind == *kind).count();
            by_kind[i].exact += gold.intersection(&pred).filter(|s| s.kind == *kind).count();
        }
        let mut has_residual = false;
        for (start, c) in sample.text.char_indices() {
            let end = start + c.len_utf8();
            let is_gold = sample
                .spans
                .iter()
                .any(|s| s.start <= start && end <= s.end);
            let masked = spans.iter().any(|s| s.start <= start && end <= s.end);
            if is_gold {
                sensitive_chars += 1;
                covered_chars += usize::from(masked);
                has_residual |= !masked;
            } else {
                nonsensitive_chars += 1;
                overmasked_chars += usize::from(masked);
            }
        }
        residual_docs += usize::from(has_residual);
    }
    let classes: serde_json::Map<String, serde_json::Value> = KINDS
        .iter()
        .zip(by_kind)
        .map(|(k, c)| (k.label().to_string(), c.metrics()))
        .collect();
    serde_json::json!({"scope":"labelled_dataset_evaluation","documents":examples.len(),"entity_exact":totals.metrics(),"per_kind":classes,"documents_with_residual_phi":residual_docs,"residual_document_rate":ratio(residual_docs,examples.len()),"sensitive_character_recall":ratio(covered_chars,sensitive_chars),"overmasked_nonsensitive_characters":overmasked_chars,"nonsensitive_character_mask_rate":ratio(overmasked_chars,nonsensitive_chars)})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Kind;
    #[test]
    fn partial_name_counts_as_document_leak() {
        let data = [Example {
            id: "x".into(),
            group: "x".into(),
            template: "x".into(),
            text: "甲乙。".into(),
            spans: vec![Span {
                start: 0,
                end: 6,
                kind: Kind::Person,
            }],
        }];
        let report = evaluate(
            &data,
            &[vec![Span {
                start: 0,
                end: 3,
                kind: Kind::Person,
            }]],
        );
        assert_eq!(report["documents_with_residual_phi"], 1);
        assert_eq!(report["sensitive_character_recall"], 0.5);
    }
}
