//! Command-line surface for the `eidos` client.
//!
//! Parsing ([`Cli`]) is separated from execution ([`run`]) so the behaviour can
//! be tested in-process against an embedded server without spawning a subprocess.
//! Every command prints a JSON value on success.

use std::collections::BTreeMap;
use std::fmt;

use clap::{Parser, Subcommand, ValueEnum};
use eidosdb_client::{CollectionMetaView, CollectionSpec, EidosClient};
use eidosdb_core::{Dimension, Embedding, Metric, VectorId};
use eidosdb_hnsw::HnswConfig;
use eidosdb_lexical::Document;
use eidosdb_proto::convert::IndexTypeChoice;
use eidosdb_query::{FieldValue, Filter, HybridQuery, Payload, SearchHit, SearchQuery, Value};
use serde_json::{Value as Json, json};

/// Top-level command-line interface for `eidos`.
#[derive(Parser, Debug)]
#[command(name = "eidos", about = "Command-line client for EidosDB")]
pub struct Cli {
    /// Server endpoint, for example `http://127.0.0.1:50051`.
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    pub endpoint: String,
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The set of subcommands `eidos` understands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a collection.
    CreateCollection {
        /// Collection name.
        #[arg(long)]
        name: String,
        /// Distance metric.
        #[arg(long, value_enum)]
        metric: MetricArg,
        /// Vector dimension.
        #[arg(long)]
        dimension: u32,
        /// Backing index type.
        #[arg(long = "index-type", value_enum)]
        index_type: IndexTypeArg,
        /// HNSW links per node.
        #[arg(long)]
        m: Option<u32>,
        /// HNSW construction candidate list size.
        #[arg(long = "ef-construction")]
        ef_construction: Option<u32>,
        /// HNSW query candidate list size.
        #[arg(long = "ef-search")]
        ef_search: Option<u32>,
        /// HNSW RNG seed.
        #[arg(long)]
        seed: Option<u64>,
    },
    /// List all collections.
    List,
    /// Describe a single collection.
    Describe {
        /// Collection name.
        #[arg(long)]
        name: String,
    },
    /// Drop a collection.
    Drop {
        /// Collection name.
        #[arg(long)]
        name: String,
    },
    /// Insert or update a single point.
    Upsert {
        /// Target collection.
        #[arg(long)]
        collection: String,
        /// Point identifier (UUID).
        #[arg(long)]
        id: String,
        /// Vector components as a comma-separated list.
        #[arg(long)]
        vector: String,
        /// Optional document text for lexical search.
        #[arg(long)]
        document: Option<String>,
        /// Optional payload as a flat JSON object.
        #[arg(long)]
        payload: Option<String>,
    },
    /// Delete a point.
    Delete {
        /// Target collection.
        #[arg(long)]
        collection: String,
        /// Point identifier (UUID).
        #[arg(long)]
        id: String,
    },
    /// Compact a collection's vector index.
    Compact {
        /// Target collection.
        #[arg(long)]
        collection: String,
    },
    /// Run a dense vector search.
    Search {
        /// Target collection.
        #[arg(long)]
        collection: String,
        /// Query vector as a comma-separated list.
        #[arg(long)]
        vector: String,
        /// Number of hits to return.
        #[arg(long)]
        k: u32,
        /// Optional metric override.
        #[arg(long, value_enum)]
        metric: Option<MetricArg>,
        /// Optional equality filter as `{"field":"f","eq":value}`.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Run a hybrid (dense + lexical) search.
    SearchHybrid {
        /// Target collection.
        #[arg(long)]
        collection: String,
        /// Optional query vector as a comma-separated list.
        #[arg(long)]
        vector: Option<String>,
        /// Optional lexical query text.
        #[arg(long)]
        text: Option<String>,
        /// Number of hits to return.
        #[arg(long)]
        k: u32,
        /// Optional equality filter as `{"field":"f","eq":value}`.
        #[arg(long)]
        filter: Option<String>,
    },
}

/// Metric choice exposed on the command line.
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum MetricArg {
    /// Cosine similarity.
    Cosine,
    /// Raw dot product.
    DotProduct,
    /// Euclidean (L2) distance.
    Euclidean,
}

impl From<MetricArg> for Metric {
    fn from(arg: MetricArg) -> Self {
        match arg {
            MetricArg::Cosine => Self::Cosine,
            MetricArg::DotProduct => Self::DotProduct,
            MetricArg::Euclidean => Self::Euclidean,
        }
    }
}

/// Index-type choice exposed on the command line.
#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum IndexTypeArg {
    /// Flat brute-force index.
    Flat,
    /// HNSW graph index.
    Hnsw,
}

impl From<IndexTypeArg> for IndexTypeChoice {
    fn from(arg: IndexTypeArg) -> Self {
        match arg {
            IndexTypeArg::Flat => Self::Flat,
            IndexTypeArg::Hnsw => Self::Hnsw,
        }
    }
}

/// Errors surfaced by the CLI.
#[derive(Debug)]
pub enum CliError {
    /// The underlying client failed.
    Client(eidosdb_client::ClientError),
    /// A JSON value could not be parsed or rendered.
    Json(serde_json::Error),
    /// The user supplied an invalid argument.
    Usage(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(error) => write!(f, "client error: {error}"),
            Self::Json(error) => write!(f, "json error: {error}"),
            Self::Usage(message) => write!(f, "usage error: {message}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<eidosdb_client::ClientError> for CliError {
    fn from(error: eidosdb_client::ClientError) -> Self {
        Self::Client(error)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Narrows a `u32` argument to a `usize`.
fn to_usize(value: u32) -> Result<usize, CliError> {
    usize::try_from(value).map_err(|_| CliError::Usage("value out of range".to_string()))
}

/// Parses a comma-separated list of floats into a vector.
fn parse_vector(raw: &str) -> Result<Vec<f32>, CliError> {
    raw.split(',')
        .map(|component| {
            component
                .trim()
                .parse::<f32>()
                .map_err(|_| CliError::Usage(format!("invalid vector component: {component:?}")))
        })
        .collect()
}

/// Parses a UUID string into a [`VectorId`].
fn parse_id(raw: &str) -> Result<VectorId, CliError> {
    let uuid = uuid::Uuid::parse_str(raw)
        .map_err(|_| CliError::Usage(format!("invalid UUID: {raw:?}")))?;
    Ok(VectorId::from_uuid(uuid))
}

/// Maps a scalar JSON value to a domain [`Value`].
fn json_to_value(value: &Json) -> Result<Value, CliError> {
    match value {
        Json::String(s) => Ok(Value::Text(s.clone())),
        Json::Bool(b) => Ok(Value::Bool(*b)),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float(f))
            } else {
                Err(CliError::Usage("number out of range".to_string()))
            }
        }
        Json::Null | Json::Array(_) | Json::Object(_) => Err(CliError::Usage(
            "payload values must be scalars or scalar arrays".to_string(),
        )),
    }
}

/// Maps a JSON field (scalar or array of scalars) to a domain [`FieldValue`].
fn json_to_field_value(value: &Json) -> Result<FieldValue, CliError> {
    match value {
        Json::Array(items) => items
            .iter()
            .map(json_to_value)
            .collect::<Result<Vec<_>, _>>()
            .map(FieldValue::Array),
        other => json_to_value(other).map(FieldValue::Scalar),
    }
}

/// Parses a flat JSON object into a domain [`Payload`].
fn parse_payload(raw: &str) -> Result<Payload, CliError> {
    let parsed: Json = serde_json::from_str(raw)?;
    let object = parsed
        .as_object()
        .ok_or_else(|| CliError::Usage("payload must be a JSON object".to_string()))?;
    let mut fields = BTreeMap::new();
    for (key, value) in object {
        fields.insert(key.clone(), json_to_field_value(value)?);
    }
    Payload::new(fields).map_err(|error| CliError::Usage(error.to_string()))
}

/// Parses an equality filter of the form `{"field":"f","eq":value}`.
fn parse_filter(raw: &str) -> Result<Filter, CliError> {
    let parsed: Json = serde_json::from_str(raw)?;
    let object = parsed
        .as_object()
        .ok_or_else(|| CliError::Usage("filter must be a JSON object".to_string()))?;
    let field = object
        .get("field")
        .and_then(Json::as_str)
        .ok_or_else(|| CliError::Usage("filter requires a string \"field\"".to_string()))?;
    let eq = object.get("eq").ok_or_else(|| {
        CliError::Usage("filter supports only {\"field\":..,\"eq\":..}".to_string())
    })?;
    Ok(Filter::Eq(field.to_string(), json_to_value(eq)?))
}

/// Renders a domain [`Metric`] as a lowercase string.
fn metric_label(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot-product",
        Metric::Euclidean => "euclidean",
    }
}

/// Renders an [`IndexTypeChoice`] as a lowercase string.
fn index_type_label(index_type: IndexTypeChoice) -> &'static str {
    match index_type {
        IndexTypeChoice::Flat => "flat",
        IndexTypeChoice::Hnsw => "hnsw",
    }
}

/// Renders a collection metadata view as JSON.
fn view_to_json(view: &CollectionMetaView) -> Json {
    json!({
        "name": view.name,
        "metric": metric_label(view.metric),
        "dimension": view.dimension.get(),
        "index_type": index_type_label(view.index_type),
        "count": view.count,
    })
}

/// Renders a domain [`Value`] as JSON.
fn value_to_json(value: &Value) -> Json {
    match value {
        Value::Text(s) => json!(s),
        Value::Integer(i) => json!(i),
        Value::Float(f) => json!(f),
        Value::Bool(b) => json!(b),
    }
}

/// Renders a domain [`Payload`] as a JSON object.
fn payload_to_json(payload: &Payload) -> Json {
    let mut object = serde_json::Map::new();
    for (key, field) in payload.iter() {
        let value = match field {
            FieldValue::Scalar(scalar) => value_to_json(scalar),
            FieldValue::Array(items) => Json::Array(items.iter().map(value_to_json).collect()),
        };
        object.insert(key.clone(), value);
    }
    Json::Object(object)
}

/// Renders a search hit as JSON.
fn hit_to_json(hit: &SearchHit) -> Json {
    json!({
        "id": hit.id.as_uuid().to_string(),
        "score": hit.score.0,
        "payload": hit.payload.as_ref().map(payload_to_json),
    })
}

/// Builds the HNSW configuration for a create request, filling unset fields
/// from the defaults.
fn build_hnsw(
    metric: Metric,
    m: Option<u32>,
    ef_construction: Option<u32>,
    ef_search: Option<u32>,
    seed: Option<u64>,
) -> Result<HnswConfig, CliError> {
    let defaults = HnswConfig::default();
    Ok(HnswConfig {
        metric,
        m: m.map(to_usize).transpose()?.unwrap_or(defaults.m),
        ef_construction: ef_construction
            .map(to_usize)
            .transpose()?
            .unwrap_or(defaults.ef_construction),
        ef_search: ef_search
            .map(to_usize)
            .transpose()?
            .unwrap_or(defaults.ef_search),
        seed: seed.unwrap_or(defaults.seed),
    })
}

/// Executes a `CreateCollection` command.
async fn handle_create(
    client: &mut EidosClient,
    name: String,
    metric: MetricArg,
    dimension: u32,
    index_type: IndexTypeArg,
    hnsw_args: (Option<u32>, Option<u32>, Option<u32>, Option<u64>),
) -> Result<Json, CliError> {
    let metric = Metric::from(metric);
    let index_type = IndexTypeChoice::from(index_type);
    let (m, ef_construction, ef_search, seed) = hnsw_args;
    let hnsw = if matches!(index_type, IndexTypeChoice::Hnsw) {
        Some(build_hnsw(metric, m, ef_construction, ef_search, seed)?)
    } else {
        None
    };
    client
        .create_collection(CollectionSpec {
            name: name.clone(),
            metric,
            dimension: Dimension(to_usize(dimension)?),
            index_type,
            hnsw,
        })
        .await?;
    Ok(json!({ "created": name }))
}

/// Executes an `Upsert` command.
async fn handle_upsert(
    client: &mut EidosClient,
    collection: String,
    id: String,
    vector: String,
    document: Option<String>,
    payload: Option<String>,
) -> Result<Json, CliError> {
    let id = parse_id(&id)?;
    let embedding =
        Embedding::new(parse_vector(&vector)?).map_err(|e| CliError::Usage(e.to_string()))?;
    let document = document
        .map(|text| Document::new(text).map_err(|e| CliError::Usage(e.to_string())))
        .transpose()?;
    let payload = payload.as_deref().map(parse_payload).transpose()?;
    client
        .upsert(&collection, id, embedding, document, payload)
        .await?;
    Ok(json!({ "upserted": id.as_uuid().to_string() }))
}

/// Executes a `Search` command.
async fn handle_search(
    client: &mut EidosClient,
    collection: String,
    vector: String,
    k: u32,
    metric: Option<MetricArg>,
    filter: Option<String>,
) -> Result<Json, CliError> {
    let embedding =
        Embedding::new(parse_vector(&vector)?).map_err(|e| CliError::Usage(e.to_string()))?;
    let filter = filter.as_deref().map(parse_filter).transpose()?;
    let hits = client
        .search(
            &collection,
            SearchQuery {
                embedding,
                k: to_usize(k)?,
                metric: metric.map(Metric::from),
                filter,
            },
        )
        .await?;
    Ok(json!({ "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>() }))
}

/// Executes a `SearchHybrid` command.
async fn handle_search_hybrid(
    client: &mut EidosClient,
    collection: String,
    vector: Option<String>,
    text: Option<String>,
    k: u32,
    filter: Option<String>,
) -> Result<Json, CliError> {
    let vector = vector
        .as_deref()
        .map(parse_vector)
        .transpose()?
        .map(Embedding::new)
        .transpose()
        .map_err(|e| CliError::Usage(e.to_string()))?;
    let filter = filter.as_deref().map(parse_filter).transpose()?;
    let hits = client
        .search_hybrid(
            &collection,
            HybridQuery {
                vector,
                text,
                k: to_usize(k)?,
                filter,
                metric: None,
                rrf_k: eidosdb_query::DEFAULT_RRF_K,
                overfetch_factor: eidosdb_query::DEFAULT_OVERFETCH_FACTOR,
            },
        )
        .await?;
    Ok(json!({ "hits": hits.iter().map(hit_to_json).collect::<Vec<_>>() }))
}

/// Connects to the server and executes the parsed command, returning the JSON
/// result to print.
///
/// # Errors
///
/// Returns [`CliError`] for invalid arguments, transport or server failures, or
/// JSON rendering errors.
pub async fn run(cli: Cli) -> Result<Json, CliError> {
    let mut client = EidosClient::connect(cli.endpoint).await?;
    match cli.command {
        Command::CreateCollection {
            name,
            metric,
            dimension,
            index_type,
            m,
            ef_construction,
            ef_search,
            seed,
        } => {
            handle_create(
                &mut client,
                name,
                metric,
                dimension,
                index_type,
                (m, ef_construction, ef_search, seed),
            )
            .await
        }
        Command::List => {
            let views = client.list_collections().await?;
            Ok(Json::Array(views.iter().map(view_to_json).collect()))
        }
        Command::Describe { name } => {
            let view = client.describe_collection(&name).await?;
            Ok(view_to_json(&view))
        }
        Command::Drop { name } => {
            let existed = client.drop_collection(&name).await?;
            Ok(json!({ "dropped": existed }))
        }
        Command::Upsert {
            collection,
            id,
            vector,
            document,
            payload,
        } => handle_upsert(&mut client, collection, id, vector, document, payload).await,
        Command::Delete { collection, id } => {
            let id = parse_id(&id)?;
            let existed = client.delete(&collection, id).await?;
            Ok(json!({ "deleted": existed }))
        }
        Command::Compact { collection } => {
            client.compact(&collection).await?;
            Ok(json!({ "compacted": collection }))
        }
        Command::Search {
            collection,
            vector,
            k,
            metric,
            filter,
        } => handle_search(&mut client, collection, vector, k, metric, filter).await,
        Command::SearchHybrid {
            collection,
            vector,
            text,
            k,
            filter,
        } => handle_search_hybrid(&mut client, collection, vector, text, k, filter).await,
    }
}
