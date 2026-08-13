# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Raise the MSRV from Rust 1.88 to 1.89 for the August 2026 fleet baseline and prefer MSRV-compatible dependency versions during Cargo updates.
- Upgrade redb from 2.6.3 to 4.1.0. Existing file-format v2 databases are migrated in place to v3 on first open; back up the store and follow [the migration guide](docs/REDB_MIGRATION.md) before upgrading.
