//! Wire contract and domain conversions for the `EidosDB` gRPC service.

/// Generated protobuf and tonic types for the EidosDB gRPC wire protocol.
pub mod pb {
    #![allow(clippy::all, clippy::pedantic, missing_docs)]
    tonic::include_proto!("eidosdb.v1");
}

pub mod convert;
pub mod error;
