//! Frozen BM25 scoring. The IDF uses the non-negative variant so common terms
//! never produce a negative contribution.

/// Term-frequency saturation parameter.
pub const K1: f64 = 1.2;
/// Length-normalization parameter.
pub const B: f64 = 0.75;

/// Inverse document frequency for a term present in `doc_freq` of `corpus_size`
/// documents. Always non-negative.
#[must_use]
pub fn idf(corpus_size: usize, doc_freq: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = corpus_size as f64;
    #[allow(clippy::cast_precision_loss)]
    let df = doc_freq as f64;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// Contribution of one query term to a document's score. `avgdl` of 0 yields 0.
#[must_use]
pub fn term_score(term_frequency: u32, doc_length: u32, average_doc_length: f64, idf: f64) -> f64 {
    if average_doc_length == 0.0 {
        return 0.0;
    }
    let f = f64::from(term_frequency);
    let len = f64::from(doc_length);
    idf * (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * len / average_doc_length))
}

#[cfg(test)]
mod tests {
    use super::{idf, term_score};

    #[test]
    fn idf_is_non_negative_even_for_ubiquitous_terms() {
        // Term in every document: still >= 0 with the non-negative variant.
        assert!(idf(10, 10) >= 0.0);
        // Rarer term scores higher than a common one.
        assert!(idf(10, 1) > idf(10, 9));
    }

    #[test]
    fn term_score_matches_hand_computed_value() {
        // Corpus: docA = "the quick brown fox" (len 4), docB = "the lazy dog" (len 3).
        // N = 2, avgdl = 3.5. Query term "fox" appears only in docA, so df = 1.
        // Values below are computed by hand from spec section 7, independent of the
        // module constants, so a drift in K1/B/term_score breaks this test:
        //   IDF = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2) = 0.6931471805599453
        //   term_score = ln(2) * (1 * 2.2) / (1 + 1.2 * (0.25 + 0.75 * 4 / 3.5))
        //              = ln(2) * 2.2 / 2.3285714285714287 = 0.6548752503
        assert!((idf(2, 1) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((term_score(1, 4, 3.5, idf(2, 1)) - 0.654_875_3).abs() < 1e-6);
    }

    #[test]
    fn zero_average_length_scores_zero() {
        // The function returns the literal 0.0 via an early-return branch; exact
        // equality is intentional here.
        #[allow(clippy::float_cmp)]
        let result = term_score(3, 5, 0.0, 1.0);
        assert!(result == 0.0);
    }
}
