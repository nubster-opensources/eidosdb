//! Library surface for the `eidos` command-line client.
//!
//! The command parsing and execution logic lives here (in [`cli`]) so it can be
//! exercised in-process by integration tests; the `eidos` binary is a thin
//! wrapper that parses arguments and prints the JSON result.

pub mod cli;
