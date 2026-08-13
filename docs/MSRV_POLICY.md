# Minimum Supported Rust Version (MSRV) policy

EidosDB targets Rust **1.89** as its Minimum Supported Rust Version.

This means every release is guaranteed to compile on stable Rust 1.89 without any nightly or unstable features.

## How the MSRV evolves

- EidosDB does not commit to supporting Rust versions older than 1.89.
- An MSRV bump is treated as a **minor** version bump per the [semver policy](SEMVER_POLICY.md). For example, raising the MSRV from 1.89 to a newer version ships in a `0.X.0` release (or `X.0.0` once at 1.0).
- The current MSRV is documented in `CHANGELOG.md` under the `Changed` section of the release that bumps it.

## Why we pick the floor we pick

- **1.89** is the August 2026 [Nubster open source fleet baseline](https://github.com/nubster-opensources/.github/blob/main/docs/MSRV_POLICY.md) and is also required by redb 4.x.
- The baseline is reviewed in February and August. EidosDB may move ahead of it only for a concrete language, dependency, correctness, or security requirement.

## How we verify the MSRV in CI

The `msrv-check` job in `.github/workflows/ci.yml` runs:

```sh
cargo +1.89 check --workspace --all-features
```

This job is part of the required status checks and blocks merging if it fails.

## Downstream impact

If you depend on EidosDB and are pinned to an older Rust toolchain, an MSRV bump is a breaking change for your build pipeline. We announce MSRV bumps in the CHANGELOG and treat them as minor version bumps to give downstream users time to upgrade.
