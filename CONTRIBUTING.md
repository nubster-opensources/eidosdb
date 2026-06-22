# Contributing to EidosDB

Thank you for your interest in contributing to EidosDB.

## Conventions

EidosDB follows the Nubster general coding standards documented in
[nubster-docs](https://github.com/nubster-opensources/nubster-docs/tree/main/docs/reference/coding-standards).
In short:

- **Trunk-Based Development**: feature branches `feature/<issue>-<slug>` from
  `main`; never commit directly on `main`.
- **Conventional Commits**: all commit messages follow the
  `type(scope): description` format, enforced by `cog verify` in the
  commit-msg hook.
- **Rust style**: workspace lints `clippy::all` and `clippy::pedantic` set to
  `deny`, MSRV pinned in `rust-toolchain.toml` and `Cargo.toml`.
- **No competitor mentions**: the source code, commit messages, pull requests
  and documentation never name competing tools.
- **English everywhere**: all rustdoc comments, public APIs, and documentation
  are written in English.

## Local setup

```bash
# Pin the Rust toolchain via rustup
rustup show

# Install local git hooks
lefthook install

# Run tests
cargo test --workspace --all-features

# Check supply chain
cargo deny check
```

## Discussion before code

Open an issue before starting significant work. For small fixes (typos,
documentation), a pull request alone is fine.

## Contributor License Agreement

Contributions to this project are governed by the Nubster Contributor License
Agreement, hosted at
[github.com/nubster-opensources/cla](https://github.com/nubster-opensources/cla).

On your first pull request, the CLA Assistant bot will automatically prompt
you to sign the CLA. Once signed, your signature applies to all current and
future contributions to any `nubster-opensources` project.

The CLA is a license grant (not a copyright assignment): you keep the
copyright on your contributions and grant Nubster a broad license to use,
sub-license, and re-license them.

## License

By contributing, you agree that your contributions shall be dual-licensed
under the [MIT license](./LICENSE-MIT) and the
[Apache License 2.0](./LICENSE-APACHE), at the user's option.

Copyright (c) Nubster.
