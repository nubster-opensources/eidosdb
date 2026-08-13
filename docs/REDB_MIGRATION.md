# redb 2 to redb 4 migration

EidosDB uses redb 4.1.0, whose minimum Rust version is 1.89 and which only
opens redb file-format v3 databases. Releases before this change used redb
2.6.3 and created file-format v2 databases by default.

## Upgrade behavior

Opening an existing EidosDB store is an in-place, one-time migration:

1. EidosDB first tries to open each `.redb` file with redb 4.
2. If redb reports file format v2, the compatibility bridge opens that file
   with the pinned redb 2.6.3 migration library and calls `Database::upgrade()`.
3. The file is closed and reopened with redb 4 before any EidosDB operation
   continues.

New databases are created directly in file format v3. An automated test creates
a real v2 database, migrates it, and verifies its row through redb 4.

## Operational guidance

- Back up the store directory before starting the first upgraded process.
- Stop every process using the store. redb requires exclusive write access for
  the migration.
- Start one upgraded process and let it open every collection before resuming
  normal traffic.
- Do not roll back to an EidosDB binary backed by redb 2 after the migration.
  redb 2 does not generally support databases already written by redb 4.

The migration can be blocked by redb persistent savepoints. EidosDB does not
create those savepoints, but if an external tool added one, opening fails with
the original redb migration error and the operator must remove the savepoint
with a redb 2.6-compatible tool before retrying.

EidosDB stores only string, byte-slice, and integer redb keys and values. It
does not use the variable-width tuple encoding changed by redb file format v3,
so no EidosDB row-schema conversion is required.
