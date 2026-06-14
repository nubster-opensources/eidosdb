//! Benchmark CLI: runs vector index backends over a synthetic dataset and prints a report.

use clap::Parser;
use eidosdb_bench::dataset::generate;
use eidosdb_bench::runner::run;
use eidosdb_core::{Dimension, FlatIndex, Metric};

/// Index backend to benchmark.
#[derive(Clone, Copy, clap::ValueEnum)]
enum Backend {
    /// In-memory exact Flat index.
    Flat,
    /// Durable disk-backed Flat index.
    Persistent,
}

/// Output format for benchmark results.
#[derive(Clone, Copy, clap::ValueEnum)]
enum OutputFormat {
    Table,
    Json,
}

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
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
    format: OutputFormat,
    /// Index backend to benchmark.
    #[arg(long, value_enum, default_value_t = Backend::Flat)]
    backend: Backend,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let dataset = generate(cli.seed, cli.dimension, cli.points, cli.queries);

    let (report, label) = match cli.backend {
        Backend::Flat => {
            let index = FlatIndex::new(Metric::Cosine, Dimension(cli.dimension));
            let report = run(index, &dataset, cli.k);
            (report, "FlatIndex (Cosine)")
        }
        Backend::Persistent => {
            let temp = tempfile::tempdir()?;
            let index = eidosdb_storage::PersistentFlatIndex::open(
                temp.path(),
                Metric::Cosine,
                Dimension(cli.dimension),
            )?;
            let report = run(index, &dataset, cli.k);
            (report, "PersistentFlatIndex (Cosine)")
        }
    };

    match cli.format {
        OutputFormat::Table => {
            println!("index       : {label}");
            println!("points      : {}", cli.points);
            println!("queries     : {}", cli.queries);
            println!("k           : {}", cli.k);
            println!("mean recall : {:.4}", report.mean_recall);
            println!("latency p50 : {:?}", report.latency.p50);
            println!("latency p99 : {:?}", report.latency.p99);
        }
        OutputFormat::Json => {
            let value = serde_json::json!({
                "index": label,
                "seed": cli.seed,
                "dimension": cli.dimension,
                "points": cli.points,
                "queries": cli.queries,
                "k": cli.k,
                "mean_recall": report.mean_recall,
                "latency_p50_ms": report.latency.p50.as_secs_f64() * 1000.0,
                "latency_p99_ms": report.latency.p99.as_secs_f64() * 1000.0,
            });
            println!("{value:#}");
        }
    }

    Ok(())
}
