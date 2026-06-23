//! gRPC service implementation for `EidosDB`.
//!
//! [`EidosDbService`] wraps an [`Arc<Registry>`] and implements the eleven
//! RPCs declared by the `EidosDb` protobuf service.  Four RPCs handle the
//! collection lifecycle; the remaining seven are stubs that return
//! [`Status::unimplemented`] and will be filled in by tasks B6, B7, B8, and C.

use std::{net::SocketAddr, sync::Arc};

use eidosdb_core::Dimension;
use eidosdb_hnsw::HnswConfig;
use eidosdb_proto::{
    convert::{
        IndexTypeChoice, index_type_from_pb, index_type_to_pb, metric_from_pb, metric_to_pb,
    },
    pb::{
        self,
        eidos_db_server::{EidosDb, EidosDbServer},
    },
    status::{conversion_error_to_status, not_found},
};
use tonic::{Request, Response, Status};

use crate::{error::ServerError, meta::CollectionMeta, registry::Registry};

// ---------------------------------------------------------------------------
// ServerError -> Status
// ---------------------------------------------------------------------------

/// Maps a [`ServerError`] to the appropriate [`tonic::Status`].
///
/// - [`ServerError::BadName`]      -> `INVALID_ARGUMENT`
/// - [`ServerError::AlreadyExists`] -> `ALREADY_EXISTS`
/// - all others                    -> `INTERNAL`
pub(crate) fn server_error_to_status(error: &ServerError) -> Status {
    match error {
        ServerError::BadName(msg) => Status::invalid_argument(msg.clone()),
        ServerError::AlreadyExists(msg) => Status::already_exists(msg.clone()),
        ServerError::Io(msg)
        | ServerError::Serde(msg)
        | ServerError::Storage(msg)
        | ServerError::Index(msg) => Status::internal(msg.clone()),
    }
}

// ---------------------------------------------------------------------------
// EidosDbService
// ---------------------------------------------------------------------------

/// gRPC service handler backed by a shared [`Registry`].
pub struct EidosDbService {
    registry: Arc<Registry>,
}

impl EidosDbService {
    /// Creates a new [`EidosDbService`] wrapping the given registry.
    pub fn new(registry: Arc<Registry>) -> Self {
        Self { registry }
    }
}

// ---------------------------------------------------------------------------
// EidosDb trait impl
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl EidosDb for EidosDbService {
    // -----------------------------------------------------------------------
    // Collection lifecycle (4 real RPCs)
    // -----------------------------------------------------------------------

    /// Creates a new named collection.
    ///
    /// Returns `INVALID_ARGUMENT` when the metric or index type is unspecified,
    /// or when the collection name is invalid.  Returns `ALREADY_EXISTS` when a
    /// collection with that name is already registered.
    async fn create_collection(
        &self,
        request: Request<pb::CreateCollectionRequest>,
    ) -> Result<Response<pb::CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        // Decode metric (i32 -> pb enum -> domain).
        let pb_metric = pb::Metric::try_from(req.metric)
            .map_err(|_| Status::invalid_argument("unknown metric value"))?;
        let metric = metric_from_pb(pb_metric).map_err(|e| conversion_error_to_status(&e))?;

        // Reject zero dimension early.
        if req.dimension == 0 {
            return Err(Status::invalid_argument(
                "dimension must be greater than zero",
            ));
        }
        let dimension = Dimension(
            usize::try_from(req.dimension)
                .map_err(|_| Status::invalid_argument("dimension out of range"))?,
        );

        // Decode index type (i32 -> pb enum -> local choice).
        let pb_index_type = pb::IndexType::try_from(req.index_type)
            .map_err(|_| Status::invalid_argument("unknown index_type value"))?;
        let index_type =
            index_type_from_pb(pb_index_type).map_err(|e| conversion_error_to_status(&e))?;

        // Build HnswConfig only when index type is HNSW; merge defaults for zero fields.
        let hnsw = if matches!(index_type, IndexTypeChoice::Hnsw) {
            let p = req.hnsw_params.unwrap_or_default();
            let def = HnswConfig::default();
            Some(HnswConfig {
                metric,
                m: if p.m == 0 {
                    def.m
                } else {
                    usize::try_from(p.m).map_err(|_| Status::invalid_argument("m out of range"))?
                },
                ef_construction: if p.ef_construction == 0 {
                    def.ef_construction
                } else {
                    usize::try_from(p.ef_construction)
                        .map_err(|_| Status::invalid_argument("ef_construction out of range"))?
                },
                ef_search: if p.ef_search == 0 {
                    def.ef_search
                } else {
                    usize::try_from(p.ef_search)
                        .map_err(|_| Status::invalid_argument("ef_search out of range"))?
                },
                seed: if p.seed == 0 { def.seed } else { p.seed },
            })
        } else {
            None
        };

        let meta = CollectionMeta {
            name: req.name,
            metric,
            dimension,
            index_type,
            hnsw,
        };

        self.registry
            .create(meta)
            .map_err(|e| server_error_to_status(&e))?;

        Ok(Response::new(pb::CreateCollectionResponse {}))
    }

    /// Drops a named collection.
    ///
    /// Returns `existed = true` if the collection was present and removed,
    /// `existed = false` if no such collection was registered (idempotent).
    async fn drop_collection(
        &self,
        request: Request<pb::DropCollectionRequest>,
    ) -> Result<Response<pb::DropCollectionResponse>, Status> {
        let name = request.into_inner().name;
        let existed = self
            .registry
            .drop_collection(&name)
            .map_err(|e| server_error_to_status(&e))?;
        Ok(Response::new(pb::DropCollectionResponse { existed }))
    }

    /// Lists all registered collections.
    async fn list_collections(
        &self,
        _request: Request<pb::ListCollectionsRequest>,
    ) -> Result<Response<pb::ListCollectionsResponse>, Status> {
        let metas = self.registry.list();

        let collections = metas
            .into_iter()
            .map(|meta| {
                let handle = self.registry.get(&meta.name);
                let count = handle
                    .and_then(|h| h.inner.read().ok().map(|g| g.len() as u64))
                    .unwrap_or(0);
                pb::CollectionInfo {
                    name: meta.name,
                    metric: metric_to_pb(meta.metric) as i32,
                    dimension: u32::try_from(meta.dimension.0).unwrap_or(0),
                    index_type: index_type_to_pb(meta.index_type) as i32,
                    count,
                }
            })
            .collect();

        Ok(Response::new(pb::ListCollectionsResponse { collections }))
    }

    /// Returns metadata and count for a single named collection.
    ///
    /// Returns `NOT_FOUND` when no collection with that name is registered.
    async fn describe_collection(
        &self,
        request: Request<pb::DescribeCollectionRequest>,
    ) -> Result<Response<pb::CollectionInfo>, Status> {
        let name = request.into_inner().name;

        let handle = self.registry.get(&name).ok_or_else(|| not_found(&name))?;

        let meta = handle.meta.clone();
        let count = handle.inner.read().map(|g| g.len() as u64).unwrap_or(0);

        Ok(Response::new(pb::CollectionInfo {
            name: meta.name,
            metric: metric_to_pb(meta.metric) as i32,
            dimension: u32::try_from(meta.dimension.0).unwrap_or(0),
            index_type: index_type_to_pb(meta.index_type) as i32,
            count,
        }))
    }

    // -----------------------------------------------------------------------
    // Stubs — to be implemented in B6 / B7 / B8 / C
    // -----------------------------------------------------------------------

    async fn upsert(
        &self,
        _request: Request<pb::UpsertRequest>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        Err(Status::unimplemented("upsert not yet implemented"))
    }

    async fn batch_upsert(
        &self,
        _request: Request<pb::BatchUpsertRequest>,
    ) -> Result<Response<pb::BatchUpsertResponse>, Status> {
        Err(Status::unimplemented("batch_upsert not yet implemented"))
    }

    async fn bulk_upsert(
        &self,
        _request: Request<tonic::Streaming<pb::BulkUpsertRequest>>,
    ) -> Result<Response<pb::BulkUpsertResponse>, Status> {
        Err(Status::unimplemented("bulk_upsert not yet implemented"))
    }

    async fn delete(
        &self,
        _request: Request<pb::DeleteRequest>,
    ) -> Result<Response<pb::DeleteResponse>, Status> {
        Err(Status::unimplemented("delete not yet implemented"))
    }

    async fn search(
        &self,
        _request: Request<pb::SearchRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        Err(Status::unimplemented("search not yet implemented"))
    }

    async fn search_hybrid(
        &self,
        _request: Request<pb::SearchHybridRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        Err(Status::unimplemented("search_hybrid not yet implemented"))
    }

    async fn compact(
        &self,
        _request: Request<pb::CompactRequest>,
    ) -> Result<Response<pb::CompactResponse>, Status> {
        Err(Status::unimplemented("compact not yet implemented"))
    }
}

// ---------------------------------------------------------------------------
// serve()
// ---------------------------------------------------------------------------

/// Starts the gRPC server on `addr`, serving the given registry.
///
/// Blocks until the server terminates.  Maps transport errors to
/// [`ServerError::Storage`].
///
/// # Errors
///
/// Returns [`ServerError`] if the transport layer fails to bind or serve.
pub async fn serve(registry: Arc<Registry>, addr: SocketAddr) -> Result<(), ServerError> {
    let svc = EidosDbServer::new(EidosDbService::new(registry));
    tonic::transport::Server::builder()
        .add_service(svc)
        .serve(addr)
        .await
        .map_err(|e| ServerError::Storage(e.to_string()))
}
