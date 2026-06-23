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
