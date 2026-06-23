//! Integration tests for the `EidosDB` gRPC service.
//!
//! Each test spins up an in-process server on an ephemeral port, connects a
//! tonic client, and exercises the four lifecycle RPCs.

use std::sync::Arc;

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
