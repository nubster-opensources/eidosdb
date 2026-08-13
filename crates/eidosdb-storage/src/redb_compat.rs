//! Compatibility bridge from redb's v2 file format to the v3 format used by redb 4.

use redb::{Database, DatabaseError};
use std::path::Path;

pub(crate) fn create(path: &Path) -> Result<Database, String> {
    match Database::create(path) {
        Ok(database) => Ok(database),
        Err(DatabaseError::UpgradeRequired(2)) => {
            upgrade_v2_file(path)?;
            Database::create(path).map_err(|error| error.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

pub(crate) fn open(path: &Path) -> Result<Database, String> {
    match Database::open(path) {
        Ok(database) => Ok(database),
        Err(DatabaseError::UpgradeRequired(2)) => {
            upgrade_v2_file(path)?;
            Database::open(path).map_err(|error| error.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn upgrade_v2_file(path: &Path) -> Result<(), String> {
    let mut database = redb_v2::Database::open(path).map_err(|error| error.to_string())?;
    database.upgrade().map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::open;
    use redb::{ReadableDatabase, TableDefinition};

    const TABLE_V2: redb_v2::TableDefinition<&str, u64> =
        redb_v2::TableDefinition::new("migration_probe");
    const TABLE_V4: TableDefinition<&str, u64> = TableDefinition::new("migration_probe");

    #[test]
    fn opening_a_v2_database_migrates_it_without_losing_rows() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("legacy.redb");
        let database = redb_v2::Database::create(&path).expect("create legacy database");
        let transaction = database.begin_write().expect("begin legacy write");
        {
            let mut table = transaction.open_table(TABLE_V2).expect("open legacy table");
            table.insert("answer", 42).expect("insert legacy row");
        }
        transaction.commit().expect("commit legacy row");
        drop(database);

        let database = open(&path).expect("migrate and open database");
        let transaction = database.begin_read().expect("begin migrated read");
        let table = transaction
            .open_table(TABLE_V4)
            .expect("open migrated table");
        let value = table
            .get("answer")
            .expect("read migrated row")
            .expect("migrated row exists")
            .value();
        assert_eq!(value, 42);
    }
}
