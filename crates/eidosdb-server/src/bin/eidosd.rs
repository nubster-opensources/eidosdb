//! `eidosd` is the `EidosDB` gRPC server daemon.
//!
//! It opens a persistent collection registry rooted at `--data-dir`, serves the
//! `EidosDb` gRPC service on `--listen`, and stops on Ctrl-C.  The shutdown is a
//! simple stop: in-flight requests are not drained and no final flush is forced
//! (graceful drain is tracked as follow-up issue B4-4).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use eidosdb_server::registry::Registry;
use eidosdb_server::service::serve;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Command-line arguments for the `eidosd` daemon.
#[derive(Parser, Debug)]
#[command(name = "eidosd", about = "EidosDB gRPC server daemon")]
struct Args {
    /// Directory holding the persistent collection data.
    #[arg(long, env = "EIDOSD_DATA_DIR")]
    data_dir: PathBuf,

    /// Address the gRPC server listens on.
    #[arg(long, env = "EIDOSD_LISTEN", default_value = "127.0.0.1:50051")]
    listen: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let registry = Arc::new(Registry::open(args.data_dir.clone())?);
    info!(
        data_dir = %args.data_dir.display(),
        listen = %args.listen,
        "eidosd listening"
    );

    tokio::select! {
        result = serve(registry, args.listen) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            info!("shutdown signal received, stopping");
        }
    }

    Ok(())
}
