//! Shared error type for all wire-to-domain and domain-to-wire conversions.

use std::fmt;

/// Error produced when converting between protobuf wire types and domain types.
#[derive(Debug)]
pub enum ConversionError {
    /// A UUID field could not be parsed.
    InvalidUuid(String),
    /// A required field was absent or set to the unspecified sentinel value.
    MissingField(&'static str),
    /// The domain layer rejected a value.
    Domain(String),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::InvalidUuid(raw) => write!(f, "invalid UUID: {raw}"),
            ConversionError::MissingField(field) => write!(f, "missing required field: {field}"),
            ConversionError::Domain(msg) => write!(f, "domain error: {msg}"),
        }
    }
}

impl std::error::Error for ConversionError {}
