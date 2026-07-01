//! Typed async gRPC client for `EidosDB`.
//!
//! [`EidosClient`] wraps the generated tonic client and exposes methods that
//! speak in domain types ([`VectorId`], [`Embedding`], [`SearchQuery`], ...)
//! rather than wire types.  All wire translation is delegated to the
//! `eidosdb-proto` conversion layer, so this crate holds no protobuf logic of
//! its own beyond assembling and disassembling request and response envelopes.

use std::fmt;

use eidosdb_core::{Dimension, Embedding, Metric, VectorId};
use eidosdb_hnsw::HnswConfig;
use eidosdb_lexical::Document;
use eidosdb_proto::convert::{
    IndexTypeChoice, delete_by_filter_to_pb, hits_from_pb, hybrid_query_to_pb, index_type_from_pb,
    index_type_to_pb, metric_from_pb, metric_to_pb, point_to_pb, search_query_to_pb,
    vector_id_to_pb,
};
use eidosdb_proto::error::ConversionError;
use eidosdb_proto::pb;
use eidosdb_query::{Filter, HybridQuery, Payload, SearchHit, SearchQuery};
use tonic::transport::Channel;

/// Errors returned by [`EidosClient`] operations.
#[derive(Debug)]
pub enum ClientError {
    /// The transport failed to connect or carry the request.
    Transport(String),
    /// The server returned a gRPC error status.
    Status(tonic::Status),
    /// A request could not be encoded or a response could not be decoded.
    Conversion(ConversionError),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(msg) => write!(f, "transport error: {msg}"),
            Self::Status(status) => write!(f, "server error: {status}"),
            Self::Conversion(error) => write!(f, "conversion error: {error}"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<tonic::Status> for ClientError {
    fn from(status: tonic::Status) -> Self {
        Self::Status(status)
    }
}

impl From<tonic::transport::Error> for ClientError {
    fn from(error: tonic::transport::Error) -> Self {
        Self::Transport(error.to_string())
    }
}

impl From<ConversionError> for ClientError {
    fn from(error: ConversionError) -> Self {
        Self::Conversion(error)
    }
}

/// Builds a [`ClientError::Conversion`] from a domain-range message.
fn out_of_range(field: &str) -> ClientError {
    ClientError::Conversion(ConversionError::Domain(format!("{field} out of range")))
}

/// Narrows a `usize` to a `u32`, mapping overflow to a [`ClientError`].
fn narrow_u32(value: usize, field: &str) -> Result<u32, ClientError> {
    u32::try_from(value).map_err(|_| out_of_range(field))
}

/// Specification used to create a collection.
#[derive(Debug, Clone)]
pub struct CollectionSpec {
    /// Collection name.
    pub name: String,
    /// Distance metric.
    pub metric: Metric,
    /// Vector dimension.
    pub dimension: Dimension,
    /// Backing index type.
    pub index_type: IndexTypeChoice,
    /// HNSW parameters; ignored unless `index_type` is HNSW.
    pub hnsw: Option<HnswConfig>,
}

/// A read-only view of a collection's metadata and current size.
#[derive(Debug, Clone)]
pub struct CollectionMetaView {
    /// Collection name.
    pub name: String,
    /// Distance metric.
    pub metric: Metric,
    /// Vector dimension.
    pub dimension: Dimension,
    /// Backing index type.
    pub index_type: IndexTypeChoice,
    /// Number of vectors currently stored.
    pub count: u64,
}

/// A point to insert or update through a write method.
#[derive(Debug, Clone)]
pub struct PointInput {
    /// Vector identifier.
    pub id: VectorId,
    /// The embedding itself.
    pub embedding: Embedding,
    /// Optional document indexed for lexical search.
    pub document: Option<Document>,
    /// Optional structured payload.
    pub payload: Option<Payload>,
}

/// Decodes a wire [`pb::CollectionInfo`] into a [`CollectionMetaView`].
fn collection_info_to_view(info: pb::CollectionInfo) -> Result<CollectionMetaView, ClientError> {
    let metric = pb::Metric::try_from(info.metric)
        .map_err(|_| out_of_range("metric"))
        .and_then(|m| metric_from_pb(m).map_err(ClientError::Conversion))?;
    let index_type = pb::IndexType::try_from(info.index_type)
        .map_err(|_| out_of_range("index_type"))
        .and_then(|t| index_type_from_pb(t).map_err(ClientError::Conversion))?;
    let dimension =
        Dimension(usize::try_from(info.dimension).map_err(|_| out_of_range("dimension"))?);
    Ok(CollectionMetaView {
        name: info.name,
        metric,
        dimension,
        index_type,
        count: info.count,
    })
}

/// A typed async gRPC client for an `EidosDB` server.
pub struct EidosClient {
    inner: pb::eidos_db_client::EidosDbClient<Channel>,
}

impl EidosClient {
    /// Connects to an `EidosDB` server at `endpoint` (for example
    /// `http://127.0.0.1:50051`).
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] if the connection cannot be established.
    pub async fn connect(endpoint: impl Into<String>) -> Result<Self, ClientError> {
        let inner = pb::eidos_db_client::EidosDbClient::connect(endpoint.into()).await?;
        Ok(Self { inner })
    }

    /// Creates a new collection.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] if a parameter overflows the wire range or the
    /// server rejects the request.
    pub async fn create_collection(&mut self, spec: CollectionSpec) -> Result<(), ClientError> {
        let dimension = narrow_u32(spec.dimension.get(), "dimension")?;
        let hnsw_params = match spec.hnsw {
            Some(config) => Some(pb::HnswParams {
                m: narrow_u32(config.m, "m")?,
                ef_construction: narrow_u32(config.ef_construction, "ef_construction")?,
                ef_search: narrow_u32(config.ef_search, "ef_search")?,
                seed: config.seed,
            }),
            None => None,
        };
        let request = pb::CreateCollectionRequest {
            name: spec.name,
            metric: metric_to_pb(spec.metric) as i32,
            dimension,
            index_type: index_type_to_pb(spec.index_type) as i32,
            hnsw_params,
        };
        self.inner.create_collection(request).await?;
        Ok(())
    }

    /// Drops a collection, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn drop_collection(&mut self, name: &str) -> Result<bool, ClientError> {
        let response = self
            .inner
            .drop_collection(pb::DropCollectionRequest {
                name: name.to_string(),
            })
            .await?;
        Ok(response.into_inner().existed)
    }

    /// Lists all collections.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] if the server rejects the request or a response
    /// entry cannot be decoded.
    pub async fn list_collections(&mut self) -> Result<Vec<CollectionMetaView>, ClientError> {
        let response = self
            .inner
            .list_collections(pb::ListCollectionsRequest {})
            .await?;
        response
            .into_inner()
            .collections
            .into_iter()
            .map(collection_info_to_view)
            .collect()
    }

    /// Returns metadata for a single collection.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] with code `NOT_FOUND` if the collection
    /// does not exist.
    pub async fn describe_collection(
        &mut self,
        name: &str,
    ) -> Result<CollectionMetaView, ClientError> {
        let response = self
            .inner
            .describe_collection(pb::DescribeCollectionRequest {
                name: name.to_string(),
            })
            .await?;
        collection_info_to_view(response.into_inner())
    }

    /// Inserts or updates a single point.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn upsert(
        &mut self,
        collection: &str,
        id: VectorId,
        embedding: Embedding,
        document: Option<Document>,
        payload: Option<Payload>,
    ) -> Result<(), ClientError> {
        let point = point_to_pb(id, &embedding, document.as_ref(), payload.as_ref());
        self.inner
            .upsert(pb::UpsertRequest {
                collection: collection.to_string(),
                point: Some(point),
            })
            .await?;
        Ok(())
    }

    /// Inserts or updates a batch of points in one unary call, returning the
    /// number applied.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn batch_upsert(
        &mut self,
        collection: &str,
        points: Vec<PointInput>,
    ) -> Result<u64, ClientError> {
        let pb_points = points
            .into_iter()
            .map(|p| point_to_pb(p.id, &p.embedding, p.document.as_ref(), p.payload.as_ref()))
            .collect();
        let response = self
            .inner
            .batch_upsert(pb::BatchUpsertRequest {
                collection: collection.to_string(),
                points: pb_points,
            })
            .await?;
        Ok(response.into_inner().upserted)
    }

    /// Deletes a point, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn delete(&mut self, collection: &str, id: VectorId) -> Result<bool, ClientError> {
        let response = self
            .inner
            .delete(pb::DeleteRequest {
                collection: collection.to_string(),
                id: vector_id_to_pb(id),
            })
            .await?;
        Ok(response.into_inner().existed)
    }

    /// Deletes all points matching `filter`, returning the number deleted.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn delete_by_filter(
        &mut self,
        collection: &str,
        filter: Filter,
    ) -> Result<u64, ClientError> {
        let response = self
            .inner
            .delete_by_filter(delete_by_filter_to_pb(collection, &filter))
            .await?;
        Ok(response.into_inner().deleted)
    }

    /// Compacts a collection's vector index.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the request.
    pub async fn compact(&mut self, collection: &str) -> Result<(), ClientError> {
        self.inner
            .compact(pb::CompactRequest {
                collection: collection.to_string(),
            })
            .await?;
        Ok(())
    }

    /// Runs a dense vector search.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] if the query cannot be encoded, the server
    /// rejects it, or a hit cannot be decoded.
    pub async fn search(
        &mut self,
        collection: &str,
        query: SearchQuery,
    ) -> Result<Vec<SearchHit>, ClientError> {
        let request = search_query_to_pb(collection, &query)?;
        let response = self.inner.search(request).await?;
        hits_from_pb(response.into_inner()).map_err(ClientError::Conversion)
    }

    /// Runs a hybrid (dense + lexical) search fused by reciprocal rank fusion.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] if the query cannot be encoded, the server
    /// rejects it, or a hit cannot be decoded.
    pub async fn search_hybrid(
        &mut self,
        collection: &str,
        query: HybridQuery,
    ) -> Result<Vec<SearchHit>, ClientError> {
        let request = hybrid_query_to_pb(collection, &query)?;
        let response = self.inner.search_hybrid(request).await?;
        hits_from_pb(response.into_inner()).map_err(ClientError::Conversion)
    }

    /// Inserts or updates points as a client-streamed sequence of chunks,
    /// returning the total number applied.
    ///
    /// Each chunk becomes one streamed message; the collection is carried on
    /// every message and resolved by the server from the first.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Status`] if the server rejects the stream (for
    /// example an empty stream or a dimension mismatch).
    pub async fn bulk_upsert(
        &mut self,
        collection: &str,
        chunks: Vec<Vec<PointInput>>,
    ) -> Result<u64, ClientError> {
        let requests: Vec<pb::BulkUpsertRequest> = chunks
            .into_iter()
            .map(|chunk| pb::BulkUpsertRequest {
                collection: collection.to_string(),
                points: chunk
                    .into_iter()
                    .map(|p| {
                        point_to_pb(p.id, &p.embedding, p.document.as_ref(), p.payload.as_ref())
                    })
                    .collect(),
            })
            .collect();
        let response = self.inner.bulk_upsert(tokio_stream::iter(requests)).await?;
        Ok(response.into_inner().upserted)
    }
}
