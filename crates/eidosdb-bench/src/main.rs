//! Benchmark CLI: runs the Flat baseline over a synthetic dataset and prints a report.

use clap::Parser;
use eidosdb_bench::dataset::generate;
use eidosdb_bench::runner::run;
use eidosdb_core::{Dimension, FlatIndex, Metric};

/// Differential benchmark for `EidosDB` indexes.
#[derive(Parser)]
#[command(name = "eidosdb-bench")]
struct Cli {
    /// RNG seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Embedding dimensionality.
    #[arg(long, default_value_t = 768)]
    dimension: usize,
    /// Number of stored points.
    #[arg(long, default_value_t = 10_000)]
    points: usize,
    /// Number of query vectors.
    #[arg(long, default_value_t = 100)]
    queries: usize,
    /// Neighbors per query.
    #[arg(long, default_value_t = 10)]
    k: usize,
}

fn main() {
    let cli = Cli::parse();
    let dataset = generate(cli.seed, cli.dimension, cli.points, cli.queries);
    let index = FlatIndex::new(Metric::Cosine, Dimension(cli.dimension));
    let report = run(index, &dataset, cli.k);

    println!("index       : FlatIndex (Cosine)");
    println!("points      : {}", cli.points);
    println!("queries     : {}", cli.queries);
    println!("k           : {}", cli.k);
    println!("mean recall : {:.4}", report.mean_recall);
    println!("latency p50 : {:?}", report.latency.p50);
    println!("latency p99 : {:?}", report.latency.p99);
}
