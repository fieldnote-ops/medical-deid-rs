use crate::data::{Kind, Span, validate_spans};
use anyhow::Result;
use regex::Regex;
use std::sync::LazyLock;

static RULES: LazyLock<Vec<(Kind, Regex)>> = LazyLock::new(|| {
    vec![
        (
            Kind::Id,
            Regex::new(r"(?:身份证(?:号)?|证件号码)[：:\s]*([0-9]{17}[0-9Xx])").unwrap(),
        ),
        (
            Kind::Phone,
            Regex::new(r"(?:电话|手机号?|联系方式)[：:\s]*((?:\+86[ -]?)?1[3-9](?:[ -]?[0-9]){9})")
                .unwrap(),
        ),
        (
            Kind::Email,
            Regex::new(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})").unwrap(),
        ),
        (
            Kind::Id,
            Regex::new(r"(?:病案号|住院号|就诊号)[：:\s]*([A-Za-z0-9-]{4,})").unwrap(),
        ),
        (
            Kind::Date,
            Regex::new(r"([12][0-9]{3}[-/年][01]?[0-9][-/月][0-3]?[0-9]日?)").unwrap(),
        ),
    ]
});

pub fn detect(text: &str) -> Vec<Span> {
    let mut found = Vec::new();
    for (kind, regex) in RULES.iter() {
        for captures in regex.captures_iter(text) {
            let hit = captures.get(1).unwrap();
            found.push(Span {
                start: hit.start(),
                end: hit.end(),
                kind: *kind,
            });
        }
    }
    merge(found)
}

/// Conservative overlap union: never expose a piece because two detectors disagree.
pub fn merge(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
    let mut result: Vec<Span> = Vec::new();
    for span in spans {
        if let Some(last) = result.last_mut().filter(|last| span.start < last.end) {
            last.end = last.end.max(span.end);
        } else {
            result.push(span);
        }
    }
    result
}

pub fn redact(text: &str, spans: &[Span]) -> Result<String> {
    validate_spans(text, spans)?;
    let mut result = String::new();
    let mut cursor = 0;
    for span in spans {
        result.push_str(&text[cursor..span.start]);
        result.push_str(&format!("[{}]", span.kind.label()));
        cursor = span.end;
    }
    result.push_str(&text[cursor..]);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_clinical_numbers_and_unicode() {
        let text = "🙂电话：13800000000，肌酐88 μmol/L；维生素B12。";
        let result = redact(text, &detect(text)).unwrap();
        assert_eq!(result, "🙂电话：[PHONE]，肌酐88 μmol/L；维生素B12。");
    }
    #[test]
    fn unions_overlapping_predictions() {
        let spans = merge(vec![
            Span {
                start: 0,
                end: 6,
                kind: Kind::Person,
            },
            Span {
                start: 3,
                end: 9,
                kind: Kind::Address,
            },
        ]);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].end, 9);
    }
}
