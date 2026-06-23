//! Integration tests for the `EidosDB` gRPC service.
//!
//! Each test spins up an in-process server on an ephemeral port, connects a
//! tonic client, and exercises the four lifecycle RPCs.

use std::sync::Arc;

use eidosdb_core::VectorId;
use eidosdb_proto::pb;
use eidosdb_server::registry::Registry;
use eidosdb_server::service::EidosDbService;
use pb::eidos_db_client::EidosDbClient;
use pb::eidos_db_server::EidosDbServer;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::{Channel, Server};

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/// Starts an in-process gRPC server on an ephemeral port and returns a
/// connected client plus the `TempDir` that must be kept alive for the
/// duration of the test.
///
/// Pattern:
/// 1. Create a temporary data directory.
/// 2. Open a `Registry` rooted there.
/// 3. Bind `127.0.0.1:0` (port 0 = OS picks a free ephemeral port).
/// 4. Capture the real address **before** moving the listener.
/// 5. Wrap the listener in a `TcpListenerStream` and spawn
///    `Server::serve_with_incoming` — avoids any fixed-port race.
/// 6. Connect `EidosDbClient` and return `(client, tempdir)`.
///
/// Reused unchanged by B6 / B7 / B8 / C.
async fn start_server() -> (EidosDbClient<Channel>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(Registry::open(dir.path().to_path_buf()).expect("open registry"));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");

    let svc = EidosDbServer::new(EidosDbService::new(Arc::clone(&registry)));
    let incoming = TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("server error");
    });

    let client = EidosDbClient::connect(format!("http://{addr}"))
        .await
        .expect("connect");

    (client, dir)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_list_describe_drop() {
    let (mut client, _dir) = start_server().await;

    // Create.
    client
        .create_collection(pb::CreateCollectionRequest {
            name: "notes".into(),
            metric: pb::Metric::Cosine as i32,
            dimension: 3,
            index_type: pb::IndexType::Hnsw as i32,
            hnsw_params: None,
        })
        .await
        .expect("create");

    // List.
    let list = client
        .list_collections(pb::ListCollectionsRequest {})
        .await
        .expect("list")
        .into_inner();
    assert_eq!(list.collections.len(), 1);

    // Describe.
    let info = client
        .describe_collection(pb::DescribeCollectionRequest {
            name: "notes".into(),
        })
        .await
        .expect("describe")
        .into_inner();
    assert_eq!(info.dimension, 3);

    // Drop.
    let resp = client
        .drop_collection(pb::DropCollectionRequest {
            name: "notes".into(),
        })
        .await
        .expect("drop")
        .into_inner();
    assert!(resp.existed);
}

#[tokio::test]
async fn describe_unknown_is_not_found() {
    let (mut client, _dir) = start_server().await;

    let err = client
        .describe_collection(pb::DescribeCollectionRequest {
            name: "ghost".into(),
        })
        .await
        .expect_err("should fail");

    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ---------------------------------------------------------------------------
// B6 helpers
// ---------------------------------------------------------------------------

/// Creates a new HNSW collection with the given name and dimension.
async fn create_hnsw(client: &mut EidosDbClient<Channel>, name: &str, dim: u32) {
    client
        .create_collection(pb::CreateCollectionRequest {
            name: name.into(),
            metric: pb::Metric::Cosine as i32,
            dimension: dim,
            index_type: pb::IndexType::Hnsw as i32,
            hnsw_params: None,
        })
        .await
        .expect("create collection");
}

// ---------------------------------------------------------------------------
// B6 tests (upsert / batch_upsert)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn upsert_then_search_via_describe_count() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let id = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id,
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect("upsert");
    let info = client
        .describe_collection(pb::DescribeCollectionRequest {
            name: "notes".into(),
        })
        .await
        .expect("describe")
        .into_inner();
    assert_eq!(info.count, 1);
}

#[tokio::test]
async fn batch_upsert_loads_all_points() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "batch-col", 3).await;
    let points: Vec<_> = (0..50_u32)
        .map(|i| pb::Point {
            id: VectorId::new().as_uuid().to_string(),
            vector: vec![f32::from(u16::try_from(i).expect("fits u16")), 0.0, 0.0],
            document: None,
            payload: None,
        })
        .collect();
    let r = client
        .batch_upsert(pb::BatchUpsertRequest {
            collection: "batch-col".into(),
            points,
        })
        .await
        .expect("batch_upsert")
        .into_inner();
    assert_eq!(r.upserted, 50);
    let info = client
        .describe_collection(pb::DescribeCollectionRequest {
            name: "batch-col".into(),
        })
        .await
        .expect("describe")
        .into_inner();
    assert_eq!(info.count, 50);
}

#[tokio::test]
async fn upsert_wrong_dimension_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "dim-col", 3).await;
    let err = client
        .upsert(pb::UpsertRequest {
            collection: "dim-col".into(),
            point: Some(pb::Point {
                id: VectorId::new().as_uuid().to_string(),
                vector: vec![1.0, 2.0], // wrong: 2 components instead of 3
                document: None,
                payload: None,
            }),
        })
        .await
        .expect_err("should fail on wrong dimension");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn upsert_unknown_collection_is_not_found() {
    let (mut client, _dir) = start_server().await;
    let err = client
        .upsert(pb::UpsertRequest {
            collection: "ghost".into(),
            point: Some(pb::Point {
                id: VectorId::new().as_uuid().to_string(),
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect_err("should fail on unknown collection");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn upsert_missing_point_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "empty-pt", 3).await;
    let err = client
        .upsert(pb::UpsertRequest {
            collection: "empty-pt".into(),
            point: None,
        })
        .await
        .expect_err("should fail on missing point");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn create_unspecified_metric_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;

    let err = client
        .create_collection(pb::CreateCollectionRequest {
            name: "x".into(),
            metric: 0, // Metric::Unspecified
            dimension: 3,
            index_type: pb::IndexType::Flat as i32,
            hnsw_params: None,
        })
        .await
        .expect_err("should fail");

    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn create_zero_dimension_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;
    let err = client
        .create_collection(pb::CreateCollectionRequest {
            name: "z".into(),
            metric: pb::Metric::Cosine as i32,
            dimension: 0,
            index_type: pb::IndexType::Flat as i32,
            hnsw_params: None,
        })
        .await
        .expect_err("should reject zero dimension");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

// ---------------------------------------------------------------------------
// B7 tests (delete / compact)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_existing_returns_existed_true() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let id = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: id.clone(),
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect("upsert");
    let r = client
        .delete(pb::DeleteRequest {
            collection: "notes".into(),
            id,
        })
        .await
        .expect("delete")
        .into_inner();
    assert!(r.existed);
}

#[tokio::test]
async fn delete_absent_returns_existed_false() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let id = VectorId::new().as_uuid().to_string();
    let r = client
        .delete(pb::DeleteRequest {
            collection: "notes".into(),
            id,
        })
        .await
        .expect("delete")
        .into_inner();
    assert!(!r.existed);
}

#[tokio::test]
async fn delete_unknown_collection_is_not_found() {
    let (mut client, _dir) = start_server().await;
    let id = VectorId::new().as_uuid().to_string();
    let err = client
        .delete(pb::DeleteRequest {
            collection: "ghost".into(),
            id,
        })
        .await
        .expect_err("nf");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn compact_after_delete_succeeds() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let id = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: id.clone(),
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect("upsert");
    client
        .delete(pb::DeleteRequest {
            collection: "notes".into(),
            id,
        })
        .await
        .expect("delete");
    client
        .compact(pb::CompactRequest {
            collection: "notes".into(),
        })
        .await
        .expect("compact");
    let info = client
        .describe_collection(pb::DescribeCollectionRequest {
            name: "notes".into(),
        })
        .await
        .expect("describe")
        .into_inner();
    assert_eq!(info.count, 0);
}

#[tokio::test]
async fn compact_unknown_collection_is_not_found() {
    let (mut client, _dir) = start_server().await;
    let err = client
        .compact(pb::CompactRequest {
            collection: "ghost".into(),
        })
        .await
        .expect_err("nf");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ---------------------------------------------------------------------------
// B8 helpers + tests (search / search_hybrid + parity oracle)
// ---------------------------------------------------------------------------

/// Builds a payload carrying a single scalar text field.
fn text_payload(field: &str, value: &str) -> pb::Payload {
    let mut fields = std::collections::HashMap::new();
    fields.insert(
        field.to_string(),
        pb::FieldValue {
            kind: Some(pb::field_value::Kind::Scalar(pb::Value {
                kind: Some(pb::value::Kind::Text(value.to_string())),
            })),
        },
    );
    pb::Payload { fields }
}

/// Builds an equality filter on a scalar text field.
fn text_eq_filter(field: &str, value: &str) -> pb::Filter {
    pb::Filter {
        kind: Some(pb::filter::Kind::Eq(pb::Comparison {
            field: field.to_string(),
            value: Some(pb::Value {
                kind: Some(pb::value::Kind::Text(value.to_string())),
            }),
        })),
    }
}

#[tokio::test]
async fn search_returns_nearest_hit() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let target = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: target.clone(),
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect("upsert");
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: VectorId::new().as_uuid().to_string(),
                vector: vec![0.0, 1.0, 0.0],
                document: None,
                payload: None,
            }),
        })
        .await
        .expect("upsert2");
    let resp = client
        .search(pb::SearchRequest {
            collection: "notes".into(),
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            metric: None,
            filter: None,
        })
        .await
        .expect("search")
        .into_inner();
    assert_eq!(resp.hits.len(), 1);
    assert_eq!(resp.hits[0].id, target);
}

#[tokio::test]
async fn search_unknown_collection_is_not_found() {
    let (mut client, _dir) = start_server().await;
    let err = client
        .search(pb::SearchRequest {
            collection: "ghost".into(),
            vector: vec![1.0, 0.0, 0.0],
            k: 1,
            metric: None,
            filter: None,
        })
        .await
        .expect_err("nf");
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn search_wrong_dimension_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    let err = client
        .search(pb::SearchRequest {
            collection: "notes".into(),
            vector: vec![1.0, 2.0],
            k: 1,
            metric: None,
            filter: None,
        })
        .await
        .expect_err("dim");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn search_with_filter_excludes_non_matching() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;

    let note_id = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: note_id.clone(),
                vector: vec![1.0, 0.0, 0.0],
                document: None,
                payload: Some(text_payload("kind", "note")),
            }),
        })
        .await
        .expect("upsert note");
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: VectorId::new().as_uuid().to_string(),
                vector: vec![0.9, 0.1, 0.0],
                document: None,
                payload: Some(text_payload("kind", "task")),
            }),
        })
        .await
        .expect("upsert task");

    let resp = client
        .search(pb::SearchRequest {
            collection: "notes".into(),
            vector: vec![1.0, 0.0, 0.0],
            k: 5,
            metric: None,
            filter: Some(text_eq_filter("kind", "note")),
        })
        .await
        .expect("search")
        .into_inner();

    assert_eq!(resp.hits.len(), 1);
    assert_eq!(resp.hits[0].id, note_id);
}

#[tokio::test]
async fn server_search_matches_direct_collection_kind() {
    // ORACLE DE PARITE: searching through the gRPC layer must yield exactly the
    // same hits, in the same order, as querying a CollectionKind directly. The
    // HNSW seed is deterministic, so identical points produce identical graphs.
    use eidosdb_core::{Dimension, Embedding};
    use eidosdb_hnsw::HnswConfig;
    use eidosdb_query::SearchQuery;
    use eidosdb_server::collection_kind::CollectionKind;

    // 1. Deterministic point set (same ids/vectors on both sides).
    let points: Vec<(VectorId, Vec<f32>)> = (0..15u32)
        .map(|i| {
            let f = f32::from(u16::try_from(i).expect("fits u16"));
            (VectorId::new(), vec![f, 15.0 - f, 0.5 * f])
        })
        .collect();
    let query_vec = vec![3.0_f32, 12.0, 1.5];

    // 2. Direct: a CollectionKind HNSW on its own tempdir, default config.
    let direct_dir = tempfile::tempdir().expect("dir");
    let mut direct =
        CollectionKind::create_hnsw(direct_dir.path(), HnswConfig::default(), Dimension(3))
            .expect("direct");
    for (id, v) in &points {
        direct
            .upsert(*id, Embedding::new(v.clone()).expect("emb"), None, None)
            .expect("upsert");
    }
    let direct_hits = direct
        .search(&SearchQuery {
            embedding: Embedding::new(query_vec.clone()).expect("q"),
            k: 5,
            metric: None,
            filter: None,
        })
        .expect("direct search");
    let direct_ids: Vec<String> = direct_hits
        .iter()
        .map(|h| h.id.as_uuid().to_string())
        .collect();

    // 3. Server: same points via gRPC.
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;
    for (id, v) in &points {
        client
            .upsert(pb::UpsertRequest {
                collection: "notes".into(),
                point: Some(pb::Point {
                    id: id.as_uuid().to_string(),
                    vector: v.clone(),
                    document: None,
                    payload: None,
                }),
            })
            .await
            .expect("upsert");
    }
    let resp = client
        .search(pb::SearchRequest {
            collection: "notes".into(),
            vector: query_vec,
            k: 5,
            metric: None,
            filter: None,
        })
        .await
        .expect("search")
        .into_inner();
    let server_ids: Vec<String> = resp.hits.iter().map(|h| h.id.clone()).collect();

    // 4. Parity: same ids in the same order.
    assert_eq!(server_ids, direct_ids);
}

#[tokio::test]
async fn search_hybrid_combines_text_and_vector() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 2).await;

    let target = VectorId::new().as_uuid().to_string();
    client
        .upsert(pb::UpsertRequest {
            collection: "notes".into(),
            point: Some(pb::Point {
                id: target.clone(),
                vector: vec![1.0, 0.0],
                document: Some("hello world".into()),
                payload: None,
            }),
        })
        .await
        .expect("upsert");

    let resp = client
        .search_hybrid(pb::SearchHybridRequest {
            collection: "notes".into(),
            vector: vec![1.0, 0.0],
            text: Some("hello".into()),
            k: 5,
            filter: None,
            metric: None,
            rrf_k: 0.0,
            overfetch_factor: 0,
        })
        .await
        .expect("search_hybrid")
        .into_inner();

    assert_eq!(resp.hits.len(), 1);
    assert_eq!(resp.hits[0].id, target);
}

#[tokio::test]
async fn search_hybrid_wrong_vector_dimension_is_invalid_argument() {
    let (mut client, _dir) = start_server().await;
    create_hnsw(&mut client, "notes", 3).await;

    let err = client
        .search_hybrid(pb::SearchHybridRequest {
            collection: "notes".into(),
            vector: vec![1.0, 2.0],
            text: None,
            k: 1,
            filter: None,
            metric: None,
            rrf_k: 0.0,
            overfetch_factor: 0,
        })
        .await
        .expect_err("dim");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}
