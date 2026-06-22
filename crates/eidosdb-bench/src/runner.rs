//! Generic benchmark runner over any `VectorIndex`.

use crate::dataset::Dataset;
use crate::latency::{LatencySummary, summarize};
use crate::recall::recall_at_k;
use eidosdb_core::{FlatIndex, VectorIndex};
use std::time::Instant;

/// Aggregated result of benchmarking one index over a dataset.
#[derive(Clone, Copy, Debug)]
pub struct BenchReport {
    /// Mean recall@k across all queries, measured against the Flat oracle.
    pub mean_recall: f32,
    /// Query latency percentiles.
    pub latency: LatencySummary,
    /// Number of queries executed.
    pub query_count: usize,
}

/// Loads `dataset` into `index`, runs every query at `k`, and measures recall
/// (against a Flat oracle built from the same points) and latency.
///
/// `metric` and `dimension` must match how `index` was constructed.
#[allow(clippy::cast_precision_loss)]
pub fn run<I: VectorIndex>(mut index: I, dataset: &Dataset, k: usize) -> BenchReport {
    let mut oracle = FlatIndex::new(index.metric(), index.dimension());
    for (id, embedding) in &dataset.points {
        index
            .insert(*id, embedding.clone())
            .expect("candidate insert");
        oracle
            .insert(*id, embedding.clone())
            .expect("oracle insert");
    }

    let mut recalls = Vec::with_capacity(dataset.queries.len());
    let mut latencies = Vec::with_capacity(dataset.queries.len());
    for query in &dataset.queries {
        let truth = oracle.search(query, k).expect("oracle search");
        let start = Instant::now();
        let candidate = index.search(query, k).expect("candidate search");
        latencies.push(start.elapsed());
        recalls.push(recall_at_k(&truth, &candidate));
    }

    let mean_recall = if recalls.is_empty() {
        1.0
    } else {
        recalls.iter().sum::<f32>() / recalls.len() as f32
    };

    BenchReport {
        mean_recall,
        latency: summarize(&latencies),
        query_count: dataset.queries.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::dataset::generate;
    use eidosdb_core::{Dimension, FlatIndex, Metric};

    #[test]
    fn flat_against_itself_has_perfect_recall() {
        let dataset = generate(1, 8, 200, 10);
        let index = FlatIndex::new(Metric::Cosine, Dimension(8));
        let report = run(index, &dataset, 10);
        assert!((report.mean_recall - 1.0).abs() < 1e-6);
        assert_eq!(report.query_count, 10);
    }

    #[test]
    fn hnsw_cosine_recall_above_threshold() {
        use eidosdb_hnsw::{HnswConfig, HnswIndex};
        let cfg = HnswConfig {
            metric: Metric::Cosine,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 1,
        };
        let dataset = generate(1, 16, 500, 50);
        let index = HnswIndex::new(cfg, Dimension(16));
        let report = run(index, &dataset, 10);
        assert!(
            report.mean_recall >= 0.90,
            "cosine recall {:.3} below 0.90",
            report.mean_recall
        );
    }

    #[test]
    fn hnsw_euclidean_recall_above_threshold() {
        use eidosdb_hnsw::{HnswConfig, HnswIndex};
        let cfg = HnswConfig {
            metric: Metric::Euclidean,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 2,
        };
        let dataset = generate(2, 16, 500, 50);
        let index = HnswIndex::new(cfg, Dimension(16));
        let report = run(index, &dataset, 10);
        assert!(
            report.mean_recall >= 0.90,
            "euclidean recall {:.3} below 0.90",
            report.mean_recall
        );
    }

    #[test]
    fn hnsw_dotproduct_recall_above_threshold() {
        use eidosdb_hnsw::{HnswConfig, HnswIndex};
        let cfg = HnswConfig {
            metric: Metric::DotProduct,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 3,
        };
        let dataset = generate(3, 16, 500, 50);
        let index = HnswIndex::new(cfg, Dimension(16));
        let report = run(index, &dataset, 10);
        assert!(
            report.mean_recall >= 0.90,
            "dotproduct recall {:.3} below 0.90",
            report.mean_recall
        );
    }
}
