# Release process

This document describes how to publish a new version of EidosDB to crates.io and create a GitHub Release.

## Surfaces

### Surface 1: GitHub UI (no CLI required)

Use this when you want to bump from your browser, or when you do not have a local Rust toolchain handy.

1. Open `https://github.com/nubster-opensources/eidosdb/actions/workflows/bump.yml`.
2. Click **Run workflow**.
3. Pick the **level** input:
   - `patch`: `0.1.0` -> `0.1.1` (bug fixes)
   - `minor`: `0.1.0` -> `0.2.0` (breaking changes allowed in 0.x per [SEMVER_POLICY.md](SEMVER_POLICY.md))
   - `major`: `1.2.3` -> `2.0.0` (breaking changes in 1.x+)
   - explicit `x.y.z`: e.g. `0.3.0`
4. The workflow runs `scripts/release.sh` in CI and opens a release prep PR.
5. Review the PR, merge it, then follow [Tagging](#tagging).

### Surface 2: local script

Use this when you want to iterate locally before pushing.

Requirements: `bash`, `git`, `cargo`, `cargo-release`, `gh`, `python3`.

```sh
bash scripts/release.sh patch   # or minor / major / 0.3.0
```

The script:
1. Checks you are on `main` with a clean tree.
2. Computes the target version.
3. Creates a `release/vX.Y.Z-prep` branch.
4. Graduates `CHANGELOG.md`.
5. Runs `cargo release` to bump all `Cargo.toml` files.
6. Runs pre-flight checks (fmt, clippy, tests).
7. Pushes the branch and opens a PR via `gh`.

### Surface 3: power-user, cargo-release direct

For one-off bumps where you do not need the CHANGELOG graduation or the PR opening:

```sh
cargo release patch --workspace --execute --no-confirm
```

You will need to graduate the CHANGELOG manually and open the PR yourself.

## Tagging

After the release prep PR is reviewed and merged, push the version tag **from your local machine** (the CI does not tag):

```sh
git checkout main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag fires `.github/workflows/release.yml`, which:
1. Publishes the publishable workspace crates to crates.io, in dependency order, requires `CARGO_REGISTRY_TOKEN` secret: `eidosdb-core`, `eidosdb-lexical`, `eidosdb-hnsw`, `eidosdb-query`, `eidosdb-proto`, `eidosdb-client`.
2. Creates a GitHub Release with the matching CHANGELOG section as release notes.

## What the bump script does NOT do

- It does not push the tag (intentional: lets the maintainer review the PR first).
- It does not publish to crates.io (that is the release workflow's job, triggered by the tag).
- It does not create a GitHub Release directly.

## Failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `cargo release` fails with "dirty tree" | CHANGELOG commit was not staged | `git add CHANGELOG.md && git commit` |
| `cargo publish` fails with "crate already exists" | Version already published | Bump again to a new version |
| GitHub Release has no notes | CHANGELOG section missing for the version | Add a section `## [X.Y.Z] - YYYY-MM-DD` to CHANGELOG.md |
| `gh pr create` fails with auth error | `GH_TOKEN` not set or expired | Re-authenticate with `gh auth login` |

## Milestones

GitHub milestones follow the format `vX.Y.0 - Theme` (example: `v0.6.0 - Scheduler`). Each milestone should include a one-sentence description of its scope and link the issues that belong to it. This naming convention keeps the [roadmap](explanation/roadmap.md) and the GitHub milestone list in sync.

## Adding it to the project

This process is already wired up via:
- `.github/workflows/bump.yml` - triggers `scripts/release.sh`
- `.github/workflows/release.yml` - fires on `v*` tags
- `scripts/release.sh` - the release prep script

The only manual step required before the first release is to add the `CARGO_REGISTRY_TOKEN` secret to your repository settings.
