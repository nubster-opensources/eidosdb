//! gRPC service implementation for `EidosDB`.
//!
//! [`EidosDbService`] wraps an [`Arc<Registry>`] and implements the eleven
//! RPCs declared by the `EidosDb` protobuf service: the four lifecycle RPCs,
//! plus `Upsert`, `BatchUpsert`, `BulkUpsert`, `Delete`, `Compact`, `Search`,
//! and `SearchHybrid`.

use std::{net::SocketAddr, sync::Arc};

use eidosdb_core::Dimension;
use eidosdb_hnsw::HnswConfig;
use eidosdb_proto::{
    convert::{
        IndexTypeChoice, hits_to_pb, hybrid_query_from_pb, index_type_from_pb, index_type_to_pb,
        metric_from_pb, metric_to_pb, point_from_pb, search_query_from_pb, vector_id_from_pb,
    },
    pb::{
        self,
        eidos_db_server::{EidosDb, EidosDbServer},
    },
    status::{conversion_error_to_status, not_found, query_error_to_status},
};
use tonic::{Request, Response, Status};

use crate::{
    collection_kind::CollectionKind,
    error::ServerError,
    meta::CollectionMeta,
    registry::{CollectionHandle, Registry},
};

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
// Blocking helper
// ---------------------------------------------------------------------------

/// Runs a closure that mutates a [`CollectionKind`] on a `spawn_blocking` thread.
///
/// The write-guard is acquired **inside** the blocking closure so that it is
/// never held across an `await` point, which would violate Rust's `Send`
/// requirements for futures.  The closure maps any [`tonic::Status`] error so
/// that callers do not need to convert errors twice.
async fn run_blocking<T, F>(handle: Arc<CollectionHandle>, f: F) -> Result<T, Status>
where
    F: FnOnce(&mut CollectionKind) -> Result<T, Status> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut guard = handle
            .inner
            .write()
            .map_err(|_| Status::internal("lock poisoned"))?;
        f(&mut guard)
    })
    .await
    .map_err(|_| Status::internal("blocking task failed"))?
}

/// Runs a closure that reads from a [`CollectionKind`] on a `spawn_blocking` thread.
///
/// Mirrors [`run_blocking`] but acquires a **shared read-guard**, so multiple
/// searches may run concurrently against the same collection.  The guard is
/// acquired **inside** the blocking closure and is never held across an
/// `await` point.
async fn run_blocking_read<T, F>(handle: Arc<CollectionHandle>, f: F) -> Result<T, Status>
where
    F: FnOnce(&CollectionKind) -> Result<T, Status> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let guard = handle
            .inner
            .read()
            .map_err(|_| Status::internal("lock poisoned"))?;
        f(&guard)
    })
    .await
    .map_err(|_| Status::internal("blocking task failed"))?
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

    /// Inserts or updates a single point in a named collection.
    ///
    /// Returns `NOT_FOUND` when the collection does not exist,
    /// `INVALID_ARGUMENT` when the point is missing, the vector dimension
    /// does not match the collection, or the point cannot be decoded.
    async fn upsert(
        &self,
        request: Request<pb::UpsertRequest>,
    ) -> Result<Response<pb::UpsertResponse>, Status> {
        let req = request.into_inner();

        let handle = self
            .registry
            .get(&req.collection)
            .ok_or_else(|| not_found(&req.collection))?;

        let point = req
            .point
            .ok_or_else(|| Status::invalid_argument("missing point"))?;

        if point.vector.len() != handle.meta.dimension.get() {
            return Err(Status::invalid_argument(format!(
                "vector dimension mismatch: expected {}, got {}",
                handle.meta.dimension.get(),
                point.vector.len(),
            )));
        }

        let decoded = point_from_pb(point).map_err(|e| conversion_error_to_status(&e))?;

        run_blocking(handle, move |kind| {
            kind.upsert(
                decoded.id,
                decoded.embedding,
                decoded.document.as_ref(),
                decoded.payload,
            )
            .map_err(|e| query_error_to_status(&e))
        })
        .await?;

        Ok(Response::new(pb::UpsertResponse {}))
    }

    /// Inserts or updates a batch of points in a named collection.
    ///
    /// All points are decoded and validated before any write occurs.  The
    /// entire batch is inserted sequentially under a single write-guard
    /// acquisition.  Returns `NOT_FOUND` when the collection does not exist,
    /// `INVALID_ARGUMENT` when any point is invalid or has the wrong dimension.
    async fn batch_upsert(
        &self,
        request: Request<pb::BatchUpsertRequest>,
    ) -> Result<Response<pb::BatchUpsertResponse>, Status> {
        let req = request.into_inner();

        let handle = self
            .registry
            .get(&req.collection)
            .ok_or_else(|| not_found(&req.collection))?;

        let expected_dim = handle.meta.dimension.get();

        // Decode and validate all points before entering the blocking section.
        let decoded_points = req
            .points
            .into_iter()
            .map(|point| {
                if point.vector.len() != expected_dim {
                    return Err(Status::invalid_argument(format!(
                        "vector dimension mismatch: expected {expected_dim}, got {}",
                        point.vector.len(),
                    )));
                }
                point_from_pb(point).map_err(|e| conversion_error_to_status(&e))
            })
            .collect::<Result<Vec<_>, Status>>()?;

        let count = decoded_points.len();

        run_blocking(handle, move |kind| {
            for d in decoded_points {
                kind.upsert(d.id, d.embedding, d.document.as_ref(), d.payload)
                    .map_err(|e| query_error_to_status(&e))?;
            }
            Ok(u64::try_from(count).unwrap_or(u64::MAX))
        })
        .await
        .map(|upserted| Response::new(pb::BatchUpsertResponse { upserted }))
    }

    /// Inserts or updates a stream of point chunks into a single collection.
    ///
    /// The target collection is taken from the first message; an empty stream is
    /// rejected with `INVALID_ARGUMENT`.  Each chunk is applied under its own
    /// write-guard so concurrent readers can interleave between chunks.  Returns
    /// `NOT_FOUND` when the collection does not exist, `INVALID_ARGUMENT` when a
    /// point has the wrong dimension or a later chunk names a different
    /// collection.
    async fn bulk_upsert(
        &self,
        request: Request<tonic::Streaming<pb::BulkUpsertRequest>>,
    ) -> Result<Response<pb::BulkUpsertResponse>, Status> {
        let mut stream = request.into_inner();

        // The collection is determined by the first message; an empty stream is
        // a client error.
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("bulk_upsert stream carried no messages"))?;

        let collection = first.collection.clone();
        let handle = self
            .registry
            .get(&collection)
            .ok_or_else(|| not_found(&collection))?;
        let expected_dim = handle.meta.dimension.get();

        let mut upserted: u64 = 0;
        let mut message = Some(first);

        while let Some(req) = message {
            // A later chunk naming a different collection would misroute points.
            if !req.collection.is_empty() && req.collection != collection {
                return Err(Status::invalid_argument(
                    "bulk_upsert stream changed collection mid-stream",
                ));
            }

            // Decode and validate this chunk before entering the blocking section.
            let decoded = req
                .points
                .into_iter()
                .map(|point| {
                    if point.vector.len() != expected_dim {
                        return Err(Status::invalid_argument(format!(
                            "vector dimension mismatch: expected {expected_dim}, got {}",
                            point.vector.len(),
                        )));
                    }
                    point_from_pb(point).map_err(|e| conversion_error_to_status(&e))
                })
                .collect::<Result<Vec<_>, Status>>()?;

            let chunk_len = decoded.len();
            run_blocking(Arc::clone(&handle), move |kind| {
                for d in decoded {
                    kind.upsert(d.id, d.embedding, d.document.as_ref(), d.payload)
                        .map_err(|e| query_error_to_status(&e))?;
                }
                Ok(())
            })
            .await?;
            upserted += u64::try_from(chunk_len).unwrap_or(u64::MAX);

            message = stream.message().await?;
        }

        Ok(Response::new(pb::BulkUpsertResponse { upserted }))
    }

    async fn delete(
        &self,
        request: Request<pb::DeleteRequest>,
    ) -> Result<Response<pb::DeleteResponse>, Status> {
        let req = request.into_inner();

        let handle = self
            .registry
            .get(&req.collection)
            .ok_or_else(|| not_found(&req.collection))?;

        let id = vector_id_from_pb(&req.id).map_err(|e| conversion_error_to_status(&e))?;

        let existed = run_blocking(handle, move |kind| {
            kind.delete(&id).map_err(|e| query_error_to_status(&e))
        })
        .await?;

        Ok(Response::new(pb::DeleteResponse { existed }))
    }

    /// Runs a dense vector search against a named collection.
    ///
    /// Returns `NOT_FOUND` when the collection does not exist,
    /// `INVALID_ARGUMENT` when the query vector dimension does not match the
    /// collection or the request cannot be decoded.
    async fn search(
        &self,
        request: Request<pb::SearchRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        let (name, query) = search_query_from_pb(request.into_inner())
            .map_err(|e| conversion_error_to_status(&e))?;

        let handle = self.registry.get(&name).ok_or_else(|| not_found(&name))?;

        if query.embedding.dimension().get() != handle.meta.dimension.get() {
            return Err(Status::invalid_argument(format!(
                "query dimension mismatch: expected {}, got {}",
                handle.meta.dimension.get(),
                query.embedding.dimension().get(),
            )));
        }

        let hits = run_blocking_read(handle, move |kind| {
            kind.search(&query).map_err(|e| query_error_to_status(&e))
        })
        .await?;

        Ok(Response::new(hits_to_pb(&hits)))
    }

    /// Runs a hybrid (dense + lexical) search fused by reciprocal rank fusion.
    ///
    /// Returns `NOT_FOUND` when the collection does not exist,
    /// `INVALID_ARGUMENT` when a supplied query vector dimension does not match
    /// the collection or the request cannot be decoded.  Dimension validation
    /// is skipped when no query vector is supplied (text-only search).
    async fn search_hybrid(
        &self,
        request: Request<pb::SearchHybridRequest>,
    ) -> Result<Response<pb::SearchResponse>, Status> {
        let (name, query) = hybrid_query_from_pb(request.into_inner())
            .map_err(|e| conversion_error_to_status(&e))?;

        let handle = self.registry.get(&name).ok_or_else(|| not_found(&name))?;

        if let Some(vector) = &query.vector {
            if vector.dimension().get() != handle.meta.dimension.get() {
                return Err(Status::invalid_argument(format!(
                    "query dimension mismatch: expected {}, got {}",
                    handle.meta.dimension.get(),
                    vector.dimension().get(),
                )));
            }
        }

        let hits = run_blocking_read(handle, move |kind| {
            kind.search_hybrid(&query)
                .map_err(|e| query_error_to_status(&e))
        })
        .await?;

        Ok(Response::new(hits_to_pb(&hits)))
    }

    async fn compact(
        &self,
        request: Request<pb::CompactRequest>,
    ) -> Result<Response<pb::CompactResponse>, Status> {
        let req = request.into_inner();

        let handle = self
            .registry
            .get(&req.collection)
            .ok_or_else(|| not_found(&req.collection))?;

        run_blocking(handle, |kind| {
            kind.compact().map_err(|e| server_error_to_status(&e))
        })
        .await?;

        Ok(Response::new(pb::CompactResponse {}))
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
