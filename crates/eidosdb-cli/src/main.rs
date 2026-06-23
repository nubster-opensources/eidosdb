//! `eidos` is the command-line client for `EidosDB`.
//!
//! It parses arguments, runs the requested command against a server, prints the
//! JSON result on stdout, and exits non-zero with a message on stderr on error.

use clap::Parser;
use eidosdb_cli::cli::{Cli, run};

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(value) => match serde_json::to_string_pretty(&value) {
            Ok(rendered) => {
                println!("{rendered}");
                std::process::ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                std::process::ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
