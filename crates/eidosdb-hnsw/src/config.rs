//! Configuration for an HNSW index.

use eidosdb_core::Metric;

/// Parameters controlling HNSW graph construction and search.
///
/// All derived quantities (`m_max0`, `m_max`, `m_l`) are computed from
/// these fields at build time to avoid repeated division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HnswConfig {
    /// Metric the graph is built for. Only this metric is supported at query time.
    pub metric: Metric,
    /// Target number of bidirectional links per node (layers > 0). Default: 16.
    pub m: usize,
    /// Candidate list size at insertion time. Default: 200.
    pub ef_construction: usize,
    /// Candidate list size at query time (may be overridden to `max(ef_search, k)`). Default: 64.
    pub ef_search: usize,
    /// Seed for the deterministic `SplitMix64` RNG. Default: a fixed constant.
    pub seed: u64,
}

impl HnswConfig {
    /// Maximum degree at layer 0 (twice `m`).
    #[must_use]
    pub fn m_max0(&self) -> usize {
        self.m.saturating_mul(2)
    }

    /// Maximum degree at layers above 0 (equals `m`).
    #[must_use]
    pub fn m_max(&self) -> usize {
        self.m
    }

    /// Level normalization factor: `1 / ln(m)`. Used by `SplitMix64::next_level`.
    #[must_use]
    pub fn m_l(&self) -> f64 {
        // m >= 2 by construction (Default enforces this); division is safe.
        #[allow(clippy::cast_precision_loss)]
        let m = self.m as f64;
        1.0 / m.ln()
    }
}

/// Fixed seed used by `Default` so two indexes built with default config and
/// the same insertion order produce identical graphs.
pub const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_1234;

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            metric: Metric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: DEFAULT_SEED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SEED, HnswConfig};
    use eidosdb_core::Metric;
    use std::f64::consts::LN_2;

    #[test]
    fn defaults_are_sane() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 64);
        assert_eq!(cfg.seed, DEFAULT_SEED);
        assert_eq!(cfg.metric, Metric::Cosine);
    }

    #[test]
    fn m_max0_is_twice_m() {
        let cfg = HnswConfig {
            m: 16,
            ..HnswConfig::default()
        };
        assert_eq!(cfg.m_max0(), 32);
        assert_eq!(cfg.m_max(), 16);
    }

    #[test]
    fn m_l_for_m2_is_one_over_ln2() {
        let cfg = HnswConfig {
            m: 2,
            ..HnswConfig::default()
        };
        assert!((cfg.m_l() - 1.0 / LN_2).abs() < 1e-12);
    }

    #[test]
    fn m_l_for_m16_uses_ln16() {
        let cfg = HnswConfig::default();
        let expected = 1.0 / 16.0_f64.ln();
        assert!((cfg.m_l() - expected).abs() < 1e-12);
    }
}
