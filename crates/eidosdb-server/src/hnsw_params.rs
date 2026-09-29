//! Resolution of the optional HNSW parameters carried by `CreateCollection`.

use eidosdb_core::Metric;
use eidosdb_hnsw::HnswConfig;
use eidosdb_proto::pb;
use tonic::Status;

/// Resolves the wire HNSW parameters into a validated [`HnswConfig`].
///
/// An absent message or field takes the value of [`HnswConfig::default`]. An
/// explicit zero on `m`, `ef_construction` or `ef_search` is rejected; `seed`
/// accepts every value, zero included.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` when a field is explicitly zero, does not fit
/// `usize`, or is rejected by [`HnswConfig::new`].
pub(crate) fn hnsw_config_from_pb(
    metric: Metric,
    params: Option<pb::HnswParams>,
) -> Result<HnswConfig, Status> {
    let params = params.unwrap_or_default();
    let defaults = HnswConfig::default();
    let m = positive_or_default(params.m, defaults.m(), "m")?;
    let ef_construction = positive_or_default(
        params.ef_construction,
        defaults.ef_construction(),
        "ef_construction",
    )?;
    let ef_search = positive_or_default(params.ef_search, defaults.ef_search(), "ef_search")?;
    let seed = params.seed.unwrap_or(defaults.seed());
    HnswConfig::new(metric, m, ef_construction, ef_search, seed)
        .map_err(|error| Status::invalid_argument(error.to_string()))
}

/// Returns `default` when `value` is absent, rejects an explicit zero, and
/// widens any other value to `usize`.
fn positive_or_default(
    value: Option<u32>,
    default: usize,
    field: &'static str,
) -> Result<usize, Status> {
    match value {
        None => Ok(default),
        Some(0) => Err(Status::invalid_argument(format!(
            "{field} must be positive when present"
        ))),
        Some(raw) => usize::try_from(raw)
            .map_err(|_| Status::invalid_argument(format!("{field} out of range"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidosdb_core::Metric;
    use eidosdb_hnsw::HnswConfig;
    use eidosdb_proto::pb;
    use tonic::Code;

    #[allow(
        clippy::unnecessary_wraps,
        reason = "returns Option to match hnsw_config_from_pb's parameter directly at each call site"
    )]
    fn params(
        m: Option<u32>,
        ef_construction: Option<u32>,
        ef_search: Option<u32>,
        seed: Option<u64>,
    ) -> Option<pb::HnswParams> {
        Some(pb::HnswParams {
            m,
            ef_construction,
            ef_search,
            seed,
        })
    }

    #[test]
    fn absent_params_take_every_default() {
        let config = hnsw_config_from_pb(Metric::Cosine, None).expect("defaults");
        let defaults = HnswConfig::default();
        assert_eq!(config.m(), defaults.m());
        assert_eq!(config.ef_construction(), defaults.ef_construction());
        assert_eq!(config.ef_search(), defaults.ef_search());
        assert_eq!(config.seed(), defaults.seed());
    }

    #[test]
    fn empty_params_take_every_default() {
        let config =
            hnsw_config_from_pb(Metric::Cosine, params(None, None, None, None)).expect("defaults");
        let defaults = HnswConfig::default();
        assert_eq!(config.m(), defaults.m());
        assert_eq!(config.ef_construction(), defaults.ef_construction());
        assert_eq!(config.ef_search(), defaults.ef_search());
        assert_eq!(config.seed(), defaults.seed());
    }

    #[test]
    fn explicit_zero_is_rejected_on_each_beam_and_degree_field() {
        for candidate in [
            params(Some(0), None, None, None),
            params(None, Some(0), None, None),
            params(None, None, Some(0), None),
        ] {
            let status =
                hnsw_config_from_pb(Metric::Cosine, candidate).expect_err("zero must be rejected");
            assert_eq!(status.code(), Code::InvalidArgument);
        }
    }

    #[test]
    fn seed_zero_is_kept() {
        let config = hnsw_config_from_pb(Metric::Cosine, params(None, None, None, Some(0)))
            .expect("seed 0 is valid");
        assert_eq!(config.seed(), 0);
    }

    #[test]
    fn explicit_values_are_kept() {
        let config = hnsw_config_from_pb(
            Metric::Cosine,
            params(Some(8), Some(100), Some(32), Some(7)),
        )
        .expect("valid");
        assert_eq!(config.m(), 8);
        assert_eq!(config.ef_construction(), 100);
        assert_eq!(config.ef_search(), 32);
        assert_eq!(config.seed(), 7);
    }

    #[test]
    fn degree_one_is_still_rejected_by_the_config() {
        let status = hnsw_config_from_pb(Metric::Cosine, params(Some(1), None, None, None))
            .expect_err("m = 1");
        assert_eq!(status.code(), Code::InvalidArgument);
    }
}
