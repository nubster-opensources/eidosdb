//! Integration tests for [`eidosdb_client::EidosClient`].
//!
//! Each test starts an in-process `EidosDB` server on an ephemeral port (via the
//! `eidosdb-server` dev-dependency), then drives it through the typed client.

use std::sync::Arc;

use eidosdb_client::{CollectionSpec, EidosClient, PointInput};
use eidosdb_core::{Dimension, Embedding, Metric, VectorId};
use eidosdb_proto::convert::IndexTypeChoice;
use eidosdb_proto::pb::eidos_db_server::EidosDbServer;
use eidosdb_query::{FieldValue, Filter, Payload, SearchQuery, Value};
use eidosdb_server::{registry::Registry, service::EidosDbService};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;

/// Starts an in-process server and returns its endpoint plus the `TempDir` that
/// must be kept alive for the duration of the test.
async fn spawn_server() -> (String, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(Registry::open(dir.path().to_path_buf()).expect("registry"));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let svc = EidosDbServer::new(EidosDbService::new(registry));
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("serve");
    });
    (format!("http://{addr}"), dir)
}

/// Creates an HNSW collection of the given dimension via the client.
async fn create(client: &mut EidosClient, name: &str, dim: usize) {
    client
        .create_collection(CollectionSpec {
            name: name.to_string(),
            metric: Metric::Cosine,
            dimension: Dimension(dim),
            index_type: IndexTypeChoice::Hnsw,
            hnsw: None,
        })
        .await
        .expect("create collection");
}

#[tokio::test]
async fn create_and_list_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let collections = client.list_collections().await.expect("list");
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "notes");
    assert_eq!(collections[0].dimension, Dimension(3));
}

#[tokio::test]
async fn describe_and_drop_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let view = client.describe_collection("notes").await.expect("describe");
    assert_eq!(view.name, "notes");
    assert_eq!(view.count, 0);
    assert!(client.drop_collection("notes").await.expect("drop"));
    assert!(!client.drop_collection("notes").await.expect("drop again"));
}

#[tokio::test]
async fn upsert_then_describe_counts_one() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let id = VectorId::new();
    client
        .upsert(
            "notes",
            id,
            Embedding::new(vec![1.0, 0.0, 0.0]).expect("embedding"),
            None,
            None,
        )
        .await
        .expect("upsert");
    let view = client.describe_collection("notes").await.expect("describe");
    assert_eq!(view.count, 1);
    assert!(client.delete("notes", id).await.expect("delete"));
}

#[tokio::test]
async fn batch_upsert_and_compact_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let points: Vec<PointInput> = (0..20u32)
        .map(|i| PointInput {
            id: VectorId::new(),
            embedding: Embedding::new(vec![f32::from(u16::try_from(i).expect("u16")), 0.0, 0.0])
                .expect("embedding"),
            document: None,
            payload: None,
        })
        .collect();
    let count = client.batch_upsert("notes", points).await.expect("batch");
    assert_eq!(count, 20);
    client.compact("notes").await.expect("compact");
    let view = client.describe_collection("notes").await.expect("describe");
    assert_eq!(view.count, 20);
}

#[tokio::test]
async fn search_returns_nearest_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let target = VectorId::new();
    client
        .upsert(
            "notes",
            target,
            Embedding::new(vec![1.0, 0.0, 0.0]).expect("embedding"),
            None,
            None,
        )
        .await
        .expect("upsert");
    client
        .upsert(
            "notes",
            VectorId::new(),
            Embedding::new(vec![0.0, 1.0, 0.0]).expect("embedding"),
            None,
            None,
        )
        .await
        .expect("upsert2");
    let hits = client
        .search(
            "notes",
            SearchQuery {
                embedding: Embedding::new(vec![1.0, 0.0, 0.0]).expect("query"),
                k: 1,
                metric: None,
                filter: None,
            },
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, target);
}

#[tokio::test]
async fn bulk_upsert_streams_chunks_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;
    let chunks: Vec<Vec<PointInput>> = (0..3u32)
        .map(|c| {
            (0..10u32)
                .map(|i| PointInput {
                    id: VectorId::new(),
                    embedding: Embedding::new(vec![
                        f32::from(u16::try_from(c * 10 + i).expect("u16")),
                        0.0,
                        0.0,
                    ])
                    .expect("embedding"),
                    document: None,
                    payload: None,
                })
                .collect()
        })
        .collect();
    let count = client.bulk_upsert("notes", chunks).await.expect("bulk");
    assert_eq!(count, 30);
    let view = client.describe_collection("notes").await.expect("describe");
    assert_eq!(view.count, 30);
}

#[tokio::test]
async fn describe_unknown_is_not_found_via_client() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    let err = client
        .describe_collection("ghost")
        .await
        .expect_err("should not be found");
    match err {
        eidosdb_client::ClientError::Status(status) => {
            assert_eq!(status.code(), tonic::Code::NotFound);
        }
        other => panic!("expected Status error, got {other}"),
    }
}

#[tokio::test]
async fn delete_by_filter_removes_matching_points() {
    let (endpoint, _dir) = spawn_server().await;
    let mut client = EidosClient::connect(endpoint).await.expect("connect");
    create(&mut client, "notes", 3).await;

    let source_field = |source: &str| {
        let mut fields = std::collections::BTreeMap::new();
        fields.insert(
            "source".to_string(),
            FieldValue::Scalar(Value::Text(source.into())),
        );
        Payload::new(fields).expect("payload")
    };

    let wiki = VectorId::new();
    let blog = VectorId::new();
    client
        .upsert(
            "notes",
            wiki,
            Embedding::new(vec![1.0, 0.0, 0.0]).unwrap(),
            None,
            Some(source_field("wiki")),
        )
        .await
        .expect("wiki");
    client
        .upsert(
            "notes",
            blog,
            Embedding::new(vec![0.0, 1.0, 0.0]).unwrap(),
            None,
            Some(source_field("blog")),
        )
        .await
        .expect("blog");

    let deleted = client
        .delete_by_filter(
            "notes",
            Filter::Eq("source".into(), Value::Text("wiki".into())),
        )
        .await
        .expect("delete_by_filter");
    assert_eq!(deleted, 1);
    assert_eq!(
        client
            .describe_collection("notes")
            .await
            .expect("describe")
            .count,
        1
    );
}
