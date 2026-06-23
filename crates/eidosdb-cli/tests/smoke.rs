//! Smoke tests for the `eidos` CLI logic.
//!
//! Each test starts an in-process `EidosDB` server, then drives the CLI through
//! its testable [`run`] entry point and asserts on the JSON it returns.

use std::sync::Arc;

use eidosdb_cli::cli::{Cli, Command, IndexTypeArg, MetricArg, run};
use eidosdb_core::VectorId;
use eidosdb_proto::pb::eidos_db_server::EidosDbServer;
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

/// Runs a single command against `endpoint` and returns the JSON result.
async fn run_command(endpoint: &str, command: Command) -> serde_json::Value {
    run(Cli {
        endpoint: endpoint.to_string(),
        command,
    })
    .await
    .expect("command succeeds")
}

async fn create_notes(endpoint: &str, dimension: u32) {
    run_command(
        endpoint,
        Command::CreateCollection {
            name: "notes".to_string(),
            metric: MetricArg::Cosine,
            dimension,
            index_type: IndexTypeArg::Hnsw,
            m: None,
            ef_construction: None,
            ef_search: None,
            seed: None,
        },
    )
    .await;
}

#[tokio::test]
async fn create_then_list_outputs_one_collection() {
    let (endpoint, _dir) = spawn_server().await;
    create_notes(&endpoint, 3).await;
    let listed = run_command(&endpoint, Command::List).await;
    let array = listed.as_array().expect("list is an array");
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["name"], "notes");
    assert_eq!(array[0]["dimension"], 3);
    assert_eq!(array[0]["index_type"], "hnsw");
}

#[tokio::test]
async fn upsert_then_describe_reports_count() {
    let (endpoint, _dir) = spawn_server().await;
    create_notes(&endpoint, 3).await;
    let id = VectorId::new().as_uuid().to_string();
    let upserted = run_command(
        &endpoint,
        Command::Upsert {
            collection: "notes".to_string(),
            id: id.clone(),
            vector: "1.0,0.0,0.0".to_string(),
            document: None,
            payload: Some(r#"{"kind":"note","rank":1}"#.to_string()),
        },
    )
    .await;
    assert_eq!(upserted["upserted"], id);
    let described = run_command(
        &endpoint,
        Command::Describe {
            name: "notes".to_string(),
        },
    )
    .await;
    assert_eq!(described["count"], 1);
}

#[tokio::test]
async fn search_outputs_expected_hit() {
    let (endpoint, _dir) = spawn_server().await;
    create_notes(&endpoint, 3).await;
    let target = VectorId::new().as_uuid().to_string();
    run_command(
        &endpoint,
        Command::Upsert {
            collection: "notes".to_string(),
            id: target.clone(),
            vector: "1.0,0.0,0.0".to_string(),
            document: None,
            payload: None,
        },
    )
    .await;
    run_command(
        &endpoint,
        Command::Upsert {
            collection: "notes".to_string(),
            id: VectorId::new().as_uuid().to_string(),
            vector: "0.0,1.0,0.0".to_string(),
            document: None,
            payload: None,
        },
    )
    .await;
    let result = run_command(
        &endpoint,
        Command::Search {
            collection: "notes".to_string(),
            vector: "1.0,0.0,0.0".to_string(),
            k: 1,
            metric: None,
            filter: None,
        },
    )
    .await;
    let hits = result["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["id"], target);
}

#[tokio::test]
async fn search_with_eq_filter_excludes_non_matching() {
    let (endpoint, _dir) = spawn_server().await;
    create_notes(&endpoint, 3).await;
    let note_id = VectorId::new().as_uuid().to_string();
    run_command(
        &endpoint,
        Command::Upsert {
            collection: "notes".to_string(),
            id: note_id.clone(),
            vector: "1.0,0.0,0.0".to_string(),
            document: None,
            payload: Some(r#"{"kind":"note"}"#.to_string()),
        },
    )
    .await;
    run_command(
        &endpoint,
        Command::Upsert {
            collection: "notes".to_string(),
            id: VectorId::new().as_uuid().to_string(),
            vector: "0.9,0.1,0.0".to_string(),
            document: None,
            payload: Some(r#"{"kind":"task"}"#.to_string()),
        },
    )
    .await;
    let result = run_command(
        &endpoint,
        Command::Search {
            collection: "notes".to_string(),
            vector: "1.0,0.0,0.0".to_string(),
            k: 5,
            metric: None,
            filter: Some(r#"{"field":"kind","eq":"note"}"#.to_string()),
        },
    )
    .await;
    let hits = result["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["id"], note_id);
}
