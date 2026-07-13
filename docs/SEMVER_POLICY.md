# Semantic Versioning policy

EidosDB follows [Semantic Versioning 2.0.0](https://semver.org/) with explicit conventions for the 0.x phase.

## 0.x phase (pre-1.0)

While the major version is 0, breaking changes are allowed on a minor version bump:

- `0.1.x` -> `0.1.y` (patch): bug fixes, performance improvements, internal refactors, additive non-breaking changes. No public API change observable by a downstream user.
- `0.x.y` -> `0.X.0` (minor): may introduce breaking changes. Removed items must have been deprecated for at least one previous minor release whenever feasible.

Reasoning: every component of EidosDB is shipped early to gather feedback. Locking ourselves into strict Semver semantics before the API surface is stable would prevent the changes we know we still need.

## 1.0 and beyond

Once 1.0 is reached, EidosDB commits to strict Semver:

- Major (`X.0.0`): breaking changes to the public API.
- Minor (`1.Y.0`): backwards-compatible additions.
- Patch (`1.x.Z`): backwards-compatible bug fixes.

## Public API definition

The public API consists of every item that is reachable from a crate's root through `pub` re-exports, except items marked `#[doc(hidden)]`. This includes:

- Public types, traits, functions, constants and modules.
- Trait method signatures and associated types.
- Public re-exports from sibling crates (when a facade crate re-exports curated items).

Items that are explicitly NOT part of the public API:

- Anything under a module annotated `#[doc(hidden)]`.
- Test-only helpers under `#[cfg(test)]`.
- Internal implementation details that do not affect observable behaviour.

TODO: extend this section with project-specific public surface definitions (e.g. on-disk schemas, wire formats, serialised types) as the project matures.

## Deprecation cycle

Before removing a public API item, it must go through at least one release where it is annotated `#[deprecated]` with a replacement hint. The replacement must exist in the same release as the deprecation.

## Breaking change documentation

Every breaking change is documented in `CHANGELOG.md` under a `### Changed` or `### Removed` sub-section, with migration instructions when the migration path is non-trivial.

## MSRV

The MSRV (Minimum Supported Rust Version) is governed by [MSRV_POLICY.md](MSRV_POLICY.md). An MSRV bump is treated as a minor version bump.
