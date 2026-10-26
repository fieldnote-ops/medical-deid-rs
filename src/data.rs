use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Kind {
    Person,
    Address,
    Phone,
    Email,
    Id,
    Date,
    Hospital,
}

pub const KINDS: [Kind; 7] = [
    Kind::Person,
    Kind::Address,
    Kind::Phone,
    Kind::Email,
    Kind::Id,
    Kind::Date,
    Kind::Hospital,
];
pub const LABELS: usize = 15;
pub const IGNORE: i64 = 15;

impl Kind {
    pub fn index(self) -> usize {
        KINDS.iter().position(|k| *k == self).unwrap()
    }
    pub fn label(self) -> &'static str {
        [
            "PERSON", "ADDRESS", "PHONE", "EMAIL", "ID", "DATE", "HOSPITAL",
        ][self.index()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: Kind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    pub group: String,
    pub template: String,
    pub text: String,
    pub spans: Vec<Span>,
}

pub fn validate_spans(text: &str, spans: &[Span]) -> Result<()> {
    let mut previous = 0;
    for span in spans {
        ensure!(
            span.start >= previous && span.start < span.end && span.end <= text.len(),
            "invalid or overlapping entity offsets"
        );
        ensure!(
            text.is_char_boundary(span.start) && text.is_char_boundary(span.end),
            "span is not on UTF-8 boundaries"
        );
        previous = span.end;
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<Vec<Example>> {
    let mut examples = Vec::new();
    let text = fs::read_to_string(path).context("cannot read dataset")?;
    let mut ids = std::collections::HashSet::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let sample: Example = serde_json::from_str(line)
            .with_context(|| format!("invalid JSONL record {}", i + 1))?;
        validate_spans(&sample.text, &sample.spans)
            .with_context(|| format!("invalid offsets in record {}", i + 1))?;
        ensure!(ids.insert(sample.id.clone()), "duplicate document id");
        examples.push(sample);
    }
    ensure!(!examples.is_empty(), "dataset is empty");
    Ok(examples)
}

pub fn load_inputs(path: &Path) -> Result<Vec<Example>> {
    #[derive(Deserialize)]
    struct Input {
        id: String,
        text: String,
    }
    let mut result = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for (i, line) in fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // Parse only inference fields; supplied annotation fields are ignored.
        let row: Input = serde_json::from_str(line)
            .with_context(|| format!("invalid input JSONL record {}", i + 1))?;
        ensure!(
            !row.id.is_empty() && ids.insert(row.id.clone()),
            "empty or duplicate document id"
        );
        result.push(Example {
            group: row.id.clone(),
            id: row.id,
            template: "inference".into(),
            text: row.text,
            spans: vec![],
        });
    }
    ensure!(!result.is_empty(), "input dataset is empty");
    Ok(result)
}

pub fn assert_disjoint(left: &[Example], right: &[Example]) -> Result<()> {
    use std::collections::HashSet;
    let groups: HashSet<_> = left.iter().map(|x| &x.group).collect();
    let texts: HashSet<_> = left.iter().map(|x| &x.text).collect();
    ensure!(
        !right
            .iter()
            .any(|x| groups.contains(&x.group) || texts.contains(&x.text)),
        "patient group or exact text overlaps between splits"
    );
    Ok(())
}

pub fn fingerprint(examples: &[Example]) -> String {
    use sha2::{Digest, Sha256};
    let bytes = serde_json::to_vec(examples).expect("serializable dataset");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inference_accepts_unlabelled_rows_and_ignores_annotations() {
        let path =
            std::env::temp_dir().join(format!("deid-input-test-{}.jsonl", std::process::id()));
        fs::write(&path, r#"{"id":"one","text":"甲🙂乙","spans":"unused"}"#).unwrap();
        let rows = load_inputs(&path).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].spans.is_empty());
        fs::write(
            &path,
            "{\"id\":\"one\",\"text\":\"x\"}\n{\"id\":\"one\",\"text\":\"y\"}",
        )
        .unwrap();
        assert!(load_inputs(&path).is_err());
        fs::write(&path, "").unwrap();
        assert!(load_inputs(&path).is_err());
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn rejects_mid_character_and_overlapping_offsets() {
        assert!(
            validate_spans(
                "患者甲🙂",
                &[Span {
                    start: 1,
                    end: 3,
                    kind: Kind::Person
                }]
            )
            .is_err()
        );
        assert!(
            validate_spans(
                "abcdef",
                &[
                    Span {
                        start: 1,
                        end: 4,
                        kind: Kind::Person
                    },
                    Span {
                        start: 3,
                        end: 5,
                        kind: Kind::Id
                    }
                ]
            )
            .is_err()
        );
    }
}
