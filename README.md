# EidosDB

> Pure-Rust vector database for the Nubster Data Plane: ANN similarity search, single binary, edge-ready.

[![CI](https://github.com/nubster-opensources/eidosdb/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nubster-opensources/eidosdb/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](./Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)](#status)
[![Made with Rust](https://img.shields.io/badge/made%20with-Rust-orange?logo=rust)](https://www.rust-lang.org/)

EidosDB is a pure-Rust vector index: it receives precomputed embeddings plus
payloads and serves approximate nearest neighbor (ANN) similarity search, with
no embedded inference and no JVM. It is designed to run as a single binary,
deployable at the edge or inside your own infrastructure.

EidosDB is sponsored by [Nubster](https://nubster.com).

## Status

**Pre-alpha.** Heavy development. Not ready for production use.

The public API is unstable and may change between minor versions.

## Quick start

> EidosDB does not yet expose a network server. The API is library-level only.

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
eidosdb-core = { git = "https://github.com/nubster-opensources/eidosdb" }
```

Run the benchmarks to verify your setup:

```sh
cargo run --release -p eidosdb-bench
```

## Why EidosDB

Most managed vector databases are cloud-only services that require sending your
embeddings to a third-party. EidosDB brings ANN search in-process or as a
self-hosted binary:

- **Pure Rust, no JVM**: no garbage collector pauses, predictable latency.
- **Single binary**: one executable, no sidecar, no daemon manager.
- **Edge-ready**: low memory footprint, no network required between the index
  and the caller.
- **Hybrid search**: combines dense ANN (HNSW) with lexical BM25 via RRF
  fusion in one query path.
- **Schemaless payloads**: store arbitrary JSON alongside each vector and
  filter at query time.

## What EidosDB is not

- **Not a full-text search engine**: BM25 in EidosDB is a complement to dense
  retrieval, not a replacement for a dedicated search system.
- **Not a transactional database**: EidosDB does not provide ACID guarantees,
  foreign keys, or relational joins.
- **Not a managed service**: there is no hosted tier. You run it yourself.
- **Not production-ready yet**: the API is pre-1.0 and breaking changes happen
  between minor versions.

## Documentation

Dedicated guides (architecture overview, HNSW tuning, hybrid search, storage
layer) are not written yet. Until then, the crate-level rustdoc comments are
the source of truth; build them locally with `cargo doc --open`.

> Documentation is a work in progress. Contributions welcome.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

EidosDB is dual-licensed under the [MIT license](LICENSE-MIT) and the
[Apache License 2.0](LICENSE-APACHE), at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual-licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details, including the Contributor
License Agreement (CLA).

Copyright (c) Nubster.
