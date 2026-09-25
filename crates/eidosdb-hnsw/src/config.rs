//! Configuration for an HNSW index.

use eidosdb_core::Metric;
use serde::{Deserialize, Serialize};

/// Parameters controlling HNSW graph construction and search.
///
/// All derived quantities (`m_max0`, `m_max`, `m_l`) are computed from
/// these fields at build time to avoid repeated division.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

/// Errors returned when validating an [`HnswConfig`].
///
/// Not implemented yet: this is the target shape for the upcoming
/// `HnswConfig::new` constructor; call sites still build `HnswConfig` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HnswConfigError {
    /// `m` was below [`HnswConfig::MIN_DEGREE`].
    #[error("m must be at least 2, got {0}")]
    DegreeTooSmall(usize),
    /// `ef_construction` was zero.
    #[error("ef_construction must be at least 1")]
    ZeroConstructionBeam,
    /// `ef_search` was zero.
    #[error("ef_search must be at least 1")]
    ZeroSearchBeam,
}

impl HnswConfig {
    /// Smallest degree accepted by [`HnswConfig::new`].
    pub const MIN_DEGREE: usize = 2;

    /// Validates and builds an HNSW configuration.
    ///
    /// Not implemented yet: the constructor and the underlying private
    /// representation land in the Building phase of this lot.
    ///
    /// # Errors
    ///
    /// Returns [`HnswConfigError::DegreeTooSmall`] when `m` is below
    /// [`HnswConfig::MIN_DEGREE`], [`HnswConfigError::ZeroConstructionBeam`]
    /// when `ef_construction` is zero, or [`HnswConfigError::ZeroSearchBeam`]
    /// when `ef_search` is zero.
    pub fn new(
        _metric: Metric,
        _m: usize,
        _ef_construction: usize,
        _ef_search: usize,
        _seed: u64,
    ) -> Result<Self, HnswConfigError> {
        todo!()
    }

    /// Returns the configured metric.
    ///
    /// Not implemented yet: lands alongside the private representation.
    #[must_use]
    pub fn metric(&self) -> Metric {
        todo!()
    }

    /// Returns the configured degree.
    ///
    /// Not implemented yet: lands alongside the private representation.
    #[must_use]
    pub fn m(&self) -> usize {
        todo!()
    }

    /// Returns the configured construction beam width.
    ///
    /// Not implemented yet: lands alongside the private representation.
    #[must_use]
    pub fn ef_construction(&self) -> usize {
        todo!()
    }

    /// Returns the configured search beam width.
    ///
    /// Not implemented yet: lands alongside the private representation.
    #[must_use]
    pub fn ef_search(&self) -> usize {
        todo!()
    }

    /// Returns the configured RNG seed.
    ///
    /// Not implemented yet: lands alongside the private representation.
    #[must_use]
    pub fn seed(&self) -> u64 {
        todo!()
    }

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
    use super::{DEFAULT_SEED, HnswConfig, HnswConfigError};
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

    #[test]
    fn new_rejects_zero_degree() {
        assert_eq!(
            HnswConfig::new(Metric::Cosine, 0, 200, 64, DEFAULT_SEED),
            Err(HnswConfigError::DegreeTooSmall(0))
        );
    }

    #[test]
    fn new_rejects_degree_one() {
        assert_eq!(
            HnswConfig::new(Metric::Cosine, 1, 200, 64, DEFAULT_SEED),
            Err(HnswConfigError::DegreeTooSmall(1))
        );
    }

    #[test]
    fn new_rejects_zero_construction_beam() {
        assert_eq!(
            HnswConfig::new(Metric::Cosine, 16, 0, 64, DEFAULT_SEED),
            Err(HnswConfigError::ZeroConstructionBeam)
        );
    }

    #[test]
    fn new_rejects_zero_search_beam() {
        assert_eq!(
            HnswConfig::new(Metric::Cosine, 16, 200, 0, DEFAULT_SEED),
            Err(HnswConfigError::ZeroSearchBeam)
        );
    }

    #[test]
    fn new_accepts_minimum_degree_and_exposes_it_via_accessors() {
        let cfg =
            HnswConfig::new(Metric::Cosine, 2, 200, 64, DEFAULT_SEED).expect("m = 2 is valid");
        assert_eq!(cfg.metric(), Metric::Cosine);
        assert_eq!(cfg.m(), 2);
        assert_eq!(cfg.ef_construction(), 200);
        assert_eq!(cfg.ef_search(), 64);
        assert_eq!(cfg.seed(), DEFAULT_SEED);
    }

    #[test]
    fn postcard_deserialization_rejects_degree_one() {
        // Built from the current (pre-migration) representation: only the
        // wire layout matters here, not how `HnswConfig` is constructed.
        let unvalidated = HnswConfig {
            metric: Metric::Cosine,
            m: 1,
            ef_construction: 200,
            ef_search: 64,
            seed: DEFAULT_SEED,
        };
        let bytes = postcard::to_allocvec(&unvalidated).expect("encode");
        let result = postcard::from_bytes::<HnswConfig>(&bytes);
        assert!(result.is_err(), "m = 1 must not survive deserialization");
    }

    #[test]
    fn postcard_bytes_from_before_the_lot_still_decode_to_the_default() {
        // Frozen reference: `postcard::to_allocvec(&HnswConfig::default())`
        // encoded with the representation as it stood before this lot
        // (public fields, plain derive). Disk compatibility requires these
        // exact bytes to keep decoding to the default config.
        const FROZEN_DEFAULT_BYTES: &[u8] = &[
            0, 16, 200, 1, 64, 180, 164, 248, 215, 252, 221, 239, 214, 222, 1,
        ];
        let decoded: HnswConfig =
            postcard::from_bytes(FROZEN_DEFAULT_BYTES).expect("frozen bytes decode");
        assert_eq!(decoded, HnswConfig::default());
    }
}
