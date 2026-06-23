//! Error types for the `EidosDB` server layer.

use std::fmt;

/// Errors that can occur in the server layer.
#[derive(Debug)]
pub enum ServerError {
    /// An I/O error occurred while reading or writing a file.
    Io(String),
    /// A serialisation or deserialisation error occurred.
    Serde(String),
    /// A collection name is invalid.
    BadName(String),
    /// A storage-layer error occurred.
    Storage(String),
    /// An index-layer error occurred.
    Index(String),
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::Serde(msg) => write!(f, "serialisation error: {msg}"),
            Self::BadName(msg) => write!(f, "invalid collection name: {msg}"),
            Self::Storage(msg) => write!(f, "storage error: {msg}"),
            Self::Index(msg) => write!(f, "index error: {msg}"),
        }
    }
}

impl std::error::Error for ServerError {}
