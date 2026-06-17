//! The lexical analyzer: text into lowercase alphanumeric tokens.

/// Splits `text` into tokens: maximal runs of alphanumeric characters, each
/// lowercased. Every non-alphanumeric character is a separator and appears in
/// no token. Deterministic and language-agnostic.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn lowercases_and_splits_on_punctuation() {
        assert_eq!(
            tokenize("Recupere l'Error-42"),
            vec!["recupere", "l", "error", "42"]
        );
    }

    #[test]
    fn collapses_runs_of_separators() {
        assert_eq!(tokenize("  a , ; b  "), vec!["a", "b"]);
    }

    #[test]
    fn keeps_accented_letters() {
        assert_eq!(tokenize("Eté ÉCOLE"), vec!["eté", "école"]);
    }

    #[test]
    fn empty_and_separator_only_yield_nothing() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("---  ...").is_empty());
    }
}
