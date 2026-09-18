//! Fine-tuning HTTP surface (`/api/finetune/*`) — Unsloth integration.
//!
//! Relocated out of Core (`apps/core/src/server/finetune.rs`) into this crate so
//! it can run BOTH in-process (Core merges [`routes`] into its router) and
//! out-of-process (the `ryu-finetune` control-plane sidecar in `main.rs` serves
//! the same router). It owns *what runs* (a fine-tune job on this node's GPU or a
//! remote Ryu Cloud GPU node) and the durable job record; the actual training
//! happens in the out-of-process Python worker (`apps-store/finetune/sidecar`),
//! which this surface reaches over one HTTP contract at [`FinetuneCtx::unsloth_url`]
//! (`RYU_UNSLOTH_URL`, default `http://127.0.0.1:8086`).
//!
//! The router is built with its own state ([`FinetuneCtx`]) inside this crate so it
//! returns a state-less, mergeable `Router<()>`. The routes are declared relative
//! to `/api/finetune` (the host nests this service at that prefix), while the
//! OpenAPI annotations keep the full external paths — mirroring `ryu-teams` and
//! `ryu-research`.

use std::net::{IpAddr, SocketAddr};

use axum::{
    body::Body,
    extract::{Extension, Path, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{from_fn, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use ryu_model_catalog::device::DeviceInfo;
use ryu_model_catalog::installed::{self, InstalledModel};
use ryu_model_format::ModelFormat;

use crate::adapters::{self, InstalledAdapter};
use crate::store::{
    FinetuneJob, FinetuneStore, FinetuneTenantContext, FinetuneTenantScope,
    StartIdempotencyClaim,
};

pub const CALLER_USER_ID_HEADER: &str = "x-ryu-caller-user-id";
pub const CALLER_ORG_ID_HEADER: &str = "x-ryu-caller-org-id";

pub fn tenant_from_headers(
    headers: &HeaderMap,
    node_id: &str,
    managed_node: bool,
) -> Result<FinetuneTenantContext, StatusCode> {
    let owner_user_id = server_header(headers, CALLER_USER_ID_HEADER)?;
    let org_id = server_header(headers, CALLER_ORG_ID_HEADER)?;
    if org_id.is_some() && owner_user_id.is_none() {
        return Err(StatusCode::FORBIDDEN);
    }
    if owner_user_id.is_none() && managed_node {
        return Err(StatusCode::FORBIDDEN);
    }
    if node_id.trim().is_empty() {
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    Ok(FinetuneTenantContext {
        owner_user_id,
        org_id,
        node_id: node_id.to_owned(),
    })
}

fn server_header(headers: &HeaderMap, name: &str) -> Result<Option<String>, StatusCode> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| StatusCode::FORBIDDEN)?.trim();
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(Some(value.to_owned()))
}

async fn attach_tenant(
    mut request: Request,
    next: Next,
    node_id: &str,
    managed_node: bool,
) -> Response {
    let tenant = match tenant_from_headers(request.headers(), node_id, managed_node) {
        Ok(tenant) => tenant,
        Err(status) => return status.into_response(),
    };
    request.extensions_mut().insert(tenant);
    next.run(request).await
}

/// Default base URL of the Python Unsloth training worker (overridable via
/// `RYU_UNSLOTH_URL`). The `@ryu/finetune` app's manifest binds the worker on
/// this same loopback port (`8086`).
pub const DEFAULT_UNSLOTH_URL: &str = "http://127.0.0.1:8086";
/// Optional comma-separated list of exact HTTPS origins allowed for remote GPU
/// dispatch. When set, a remote target must match one of these origins after
/// normalization. This gives operators a strict egress boundary in addition to
/// the built-in private-address checks.
pub const REMOTE_ALLOWLIST_ENV: &str = "RYU_FINETUNE_REMOTE_ALLOWLIST";

const MAX_REMOTE_URL_CHARS: usize = 2048;
const MAX_CONTROL_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_CONTROL_REQUEST_BYTES: usize = 4 * 1024 * 1024;

fn disallowed_remote_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, ..] = ip.octets();
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_broadcast()
                || a == 0
                || a == 10
                || (a == 100 && (64..=127).contains(&b))
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 168)
                || (a == 192 && b == 0)
                || (a == 198 && (18..=19).contains(&b))
                || a >= 224
        }
        IpAddr::V6(ip) => {
            if let Some(ipv4) = ip.to_ipv4() {
                return disallowed_remote_ip(IpAddr::V4(ipv4));
            }
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

fn remote_origin(url: &reqwest::Url) -> String {
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let host = if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host
    };
    let mut origin = format!("{}://{host}", url.scheme());
    if let Some(port) = url.port() {
        origin.push(':');
        origin.push_str(&port.to_string());
    }
    origin
}

fn normalize_remote_base_url(raw: &str) -> Result<String, String> {
    let candidate = raw.trim().trim_end_matches('/');
    if candidate.is_empty() || candidate.chars().count() > MAX_REMOTE_URL_CHARS {
        return Err(format!(
            "remote.url must be a non-empty URL of at most {MAX_REMOTE_URL_CHARS} characters"
        ));
    }
    let url = reqwest::Url::parse(candidate)
        .map_err(|_| "remote.url must be a valid absolute URL".to_owned())?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("remote.url must not contain userinfo".to_owned());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("remote.url must not contain a query or fragment".to_owned());
    }
    if url.path() != "" && url.path() != "/" {
        return Err("remote.url must be a bare origin without a path".to_owned());
    }
    let host = url
        .host_str()
        .map(str::to_ascii_lowercase)
        .filter(|host| !host.is_empty())
        .ok_or_else(|| "remote.url must contain a host".to_owned())?;
    let ip_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(&host);
    let loopback_dev = cfg!(debug_assertions)
        && url.scheme() == "http"
        && ip_host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if url.scheme() != "https" && !loopback_dev {
        return Err("remote.url must use https".to_owned());
    }
    if host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host == "metadata.google.internal"
        || host == "metadata"
    {
        return Err("remote.url must not target a local or metadata host".to_owned());
    }
    if let Ok(ip) = ip_host.parse::<IpAddr>() {
        if disallowed_remote_ip(ip) && !loopback_dev {
            return Err(
                "remote.url must not target a private, loopback, or link-local address".to_owned(),
            );
        }
    }
    Ok(remote_origin(&url))
}

fn validate_remote_base_url(raw: &str) -> Result<String, String> {
    let origin = normalize_remote_base_url(raw)?;
    if let Ok(configured) = std::env::var(REMOTE_ALLOWLIST_ENV) {
        let configured = configured.trim();
        if !configured.is_empty()
            && !configured
                .split(',')
                .filter_map(|entry| normalize_remote_base_url(entry).ok())
                .any(|allowed| allowed == origin)
        {
            return Err(format!(
                "remote.url is not in the configured {REMOTE_ALLOWLIST_ENV} allowlist"
            ));
        }
    }
    Ok(origin)
}

/// Resolve a remote hostname once, reject private answers, and pin the safe
/// answer into a redirect-disabled client. This closes the DNS-rebinding gap
/// left by validating only the URL text: every proxy request uses the same
/// public address that passed the check, and a redirect cannot move the bearer
/// to another host.
async fn remote_client_for_base(ctx: &FinetuneCtx, base: &str) -> Result<reqwest::Client, String> {
    let url = reqwest::Url::parse(base)
        .map_err(|_| "remote.url must be a valid absolute URL".to_owned())?;
    let host = url
        .host_str()
        .ok_or_else(|| "remote.url must contain a host".to_owned())?;
    let ip_host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if ip_host.parse::<IpAddr>().is_ok() {
        return Ok(ctx.remote_client.clone());
    }

    let port = url.port_or_known_default().unwrap_or(443);
    let mut addresses = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        tokio::net::lookup_host((host, port)),
    )
    .await
    .map_err(|_| "remote hostname resolution timed out".to_owned())?
    .map_err(|_| "remote hostname could not be resolved".to_owned())?;
    let address: SocketAddr = addresses
        .find(|address| !disallowed_remote_ip(address.ip()))
        .ok_or_else(|| "remote hostname resolved only to private or local addresses".to_owned())?;

    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .resolve(host, address)
        .build()
        .map_err(|_| "safe remote HTTP client could not be created".to_owned())
}

/// This app's manifest id. It is also the namespace half of every event id below —
/// Core re-checks on each emit that the authenticated caller IS this plugin and that
/// the event is declared in this manifest, so the two must stay in step.
const PLUGIN_ID: &str = "@ryu/finetune";

/// Raised when the worker (or a remote GPU node) has accepted a job.
const EVENT_JOB_STARTED: &str = "@ryu/finetune#job.started";

/// Raised on the poll that first observes a job finish training.
const EVENT_JOB_SUCCEEDED: &str = "@ryu/finetune#job.succeeded";

/// Raised on the poll that first observes a job end in failure.
const EVENT_JOB_FAILED: &str = "@ryu/finetune#job.failed";

/// Router state for the fine-tuning HTTP surface: the durable [`FinetuneStore`],
/// an un-timed HTTP client (the adapter→GGUF merge is long-running, so no short
/// timeout), and the base URL of the Python training worker. Cheap to clone
/// (`Arc`s inside). This replaces Core's `ServerState` — the finetune handlers
/// touched only `state.finetune` + `state.client`, so this three-field state is a
/// faithful, decoupled substitute.
#[derive(Clone)]
pub struct FinetuneCtx {
    pub store: FinetuneStore,
    pub client: reqwest::Client,
    /// A redirect-disabled client for the local worker. The worker bearer must
    /// never follow an operator-configured redirect to a different host.
    worker_client: reqwest::Client,
    pub unsloth_url: String,
    /// The shared secret Core injects into both finetune sidecars. The control
    /// plane must re-stamp it when it calls the Python worker; the worker's
    /// protected routes fail closed without it.
    pub worker_token: Option<String>,
    /// A client with redirects disabled for remote-node proxying. Following a
    /// redirect could otherwise send a stored remote bearer to another host.
    remote_client: reqwest::Client,
    /// Raises the app events this crate declares in `manifest.json`. Built once
    /// here rather than at each call site so the plugin id can never drift between
    /// emits, and off the same client so it shares the connection pool. Training is
    /// the slowest thing Ryu does and nothing else on the node can see it finish —
    /// these events are how a plugin hook or a workflow learns without polling
    /// `/api/finetune/list` forever.
    pub events: ryu_app_events::EventEmitter,
}

impl FinetuneCtx {
    /// Build a context. `unsloth_url` falls back to [`DEFAULT_UNSLOTH_URL`] when
    /// empty; the trailing slash is trimmed so `worker("/finetune")` composes
    /// cleanly.
    pub fn new(
        store: FinetuneStore,
        client: reqwest::Client,
        unsloth_url: impl Into<String>,
    ) -> Self {
        let mut url = unsloth_url.into();
        if url.trim().is_empty() {
            url = DEFAULT_UNSLOTH_URL.to_string();
        }
        let url = url.trim().trim_end_matches('/').to_string();
        let events = ryu_app_events::EventEmitter::with_client(PLUGIN_ID, client.clone());
        let worker_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("redirect-disabled finetune worker client must build");
        let remote_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("redirect-disabled finetune remote client must build");
        Self {
            store,
            client,
            worker_client,
            unsloth_url: url,
            worker_token: None,
            remote_client,
            events,
        }
    }

    /// Attach the per-plugin bearer used for control-plane → worker requests.
    /// Empty values are treated as absent so a misconfigured standalone binary
    /// cannot accidentally send an empty `Authorization` header.
    pub fn with_worker_token(mut self, token: Option<String>) -> Self {
        self.worker_token = token
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        self
    }

    /// Absolute URL of a Python worker endpoint (`path` starts with `/`).
    fn worker(&self, path: &str) -> String {
        format!("{}{path}", self.unsloth_url)
    }

    fn worker_request(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.worker_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }
}

/// Read JSON from a worker/remote response without allowing a hostile endpoint
/// to make the control plane buffer an arbitrary body in memory.
async fn response_json_limited(resp: reqwest::Response) -> anyhow::Result<Value> {
    if resp
        .content_length()
        .is_some_and(|length| length > MAX_CONTROL_RESPONSE_BYTES as u64)
    {
        anyhow::bail!("finetune control response exceeds {MAX_CONTROL_RESPONSE_BYTES} bytes");
    }
    let mut bytes = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if bytes.len().saturating_add(chunk.len()) > MAX_CONTROL_RESPONSE_BYTES {
            anyhow::bail!("finetune control response exceeds {MAX_CONTROL_RESPONSE_BYTES} bytes");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&bytes)?)
}

/// Build the `/api/finetune/*` router with its own state baked in, returning a
/// state-less `Router<()>` the host nests at `/api/finetune`.
pub fn routes(ctx: FinetuneCtx) -> Router<()> {
    let node_id = ctx.store.node_id().to_owned();
    let managed_node = std::env::var("RYU_MANAGED_NODE").ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        )
    });
    Router::new()
        .route("/capability", get(capability))
        .route(
            "/start",
            post(start).layer(axum::extract::DefaultBodyLimit::max(
                MAX_CONTROL_REQUEST_BYTES,
            )),
        )
        .route("/list", get(list))
        .route("/adapters", get(list_adapters))
        .route(
            "/merge",
            post(merge).layer(axum::extract::DefaultBodyLimit::max(
                MAX_CONTROL_REQUEST_BYTES,
            )),
        )
        .route("/:id", get(get_job).delete(cancel))
        .route("/:id/stream", get(stream))
        .with_state(ctx)
        .layer(from_fn(move |request: Request, next: Next| {
            let node_id = node_id.clone();
            async move { attach_tenant(request, next, &node_id, managed_node).await }
        }))
}

/// The OpenAPI sub-document for the fine-tuning surface, merged into Core's spec
/// when the `finetune` feature is enabled.
pub fn openapi() -> utoipa::openapi::OpenApi {
    <FinetuneApiDoc as utoipa::OpenApi>::openapi()
}

/// The document Core imports. `components(schemas(...))` is what turns each
/// `request_body = T` into a resolvable `#/components/schemas/T` entry: without it
/// the operation still carries a `$ref`, but the target is missing and Core's
/// `resolve_ref` yields nothing — a derived write tool with zero visible arguments.
/// utoipa 5 also auto-collects schemas reachable from the annotated paths, so these
/// rows are belt-and-braces; they are listed explicitly anyway so the registration
/// is greppable and cannot be silently lost to an attribute edit.
///
/// `DatasetSpec`/`LoraSpec`/`TrainingSpec`/`RemoteTarget` are reachable only
/// TRANSITIVELY, through fields of [`StartJobBody`] — the transitive half of the
/// graph is the part that breaks builds when a derive is missed.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(capability, start, list, get_job, cancel, list_adapters, merge, stream),
    components(schemas(
        DatasetSpec,
        LoraSpec,
        MergeBody,
        RemoteTarget,
        StartJobBody,
        TrainingSpec,
    ))
)]
struct FinetuneApiDoc;

// ── Request bodies ──────────────────────────────────────────────────────────
//
// These types describe the wire shape; they are deliberately NOT used as axum
// extractors. `start` and `merge` are proxies: the body is forwarded verbatim to
// the Python worker (`apps-store/finetune/sidecar`, whose pydantic models are the
// contract of record) and, for a remote job, on to another node's Core. Putting a
// Rust struct in the extract path would make this crate a gatekeeper for a schema
// it does not own — every worker-side field addition would then need a Rust
// release to stop being silently dropped. So the handlers keep `Json<Value>` and
// the annotation carries the type, which is the half Core reads.
//
// They mirror `FinetuneRequest`/`MergeRequest` in `ryu_unsloth/server.py` field for
// field; change them together.

/// Request body for `POST /api/finetune/start`.
// Everything below is `//`, not `///`, ON PURPOSE: utoipa lifts a struct's doc
// comment into the schema's own `description`, so internal rationale written as
// `///` ships to the model alongside the arguments.
//
// The FIELD docs below, by contrast, are not decoration — utoipa lifts them
// verbatim into each property's `description`, and they are the only prose the
// model reads when it decides how to call the derived `start` tool.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct StartJobBody {
    /// Hugging Face repo id of the base model to fine-tune — ideally an
    /// `unsloth/*-bnb-4bit` build, which is what fits on a consumer GPU.
    pub base_model_id: String,
    /// Optional stable retry key. Reusing it returns the original accepted job;
    /// using it with a different request body is rejected.
    #[serde(default, alias = "idempotencyKey")]
    pub idempotency_key: Option<String>,
    /// The training data. Supply either inline `samples` or a `path` to a
    /// `.json`/`.jsonl` file with the same row shapes.
    pub dataset: DatasetSpec,
    /// Stem for the saved adapter directory, e.g. `my-tone-v1`. Derived from the
    /// base model when absent.
    #[serde(default)]
    pub output_name: Option<String>,
    /// LoRA adapter shape. Every field is optional; omit the whole object to train
    /// with the worker's defaults.
    // `#[schema(inline)]` — NOT a doc comment: everything above IS lifted into the
    // schema and read by the model, and this rationale is not for it. An
    // `Option<Struct>` renders as `oneOf: [null, <schema>]`, and Core follows only a
    // `$ref` at the TOP of a node — a ref buried in that wrapper reaches the model
    // as an opaque pointer. Inlined, it sees the real sub-fields.
    #[serde(default)]
    #[schema(inline)]
    pub lora: Option<LoraSpec>,
    /// Training hyper-parameters. Every field is optional; omit the whole object to
    /// train with the worker's defaults.
    #[serde(default)]
    #[schema(inline)]
    pub training: Option<TrainingSpec>,
    /// Where to train: `local` (this node's GPU, the default) or `remote` (a Ryu
    /// Cloud GPU node, which then also needs `remote`).
    #[serde(default)]
    pub target: Option<String>,
    /// The GPU node to train on. Required when `target` is `remote`, ignored
    /// otherwise.
    #[serde(default)]
    #[schema(inline)]
    pub remote: Option<RemoteTarget>,
}

/// The training data for a fine-tune job. One of `samples` or `path` must be
/// present — the worker rejects a dataset with neither.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct DatasetSpec {
    /// Row shape: `chat` (default) — `{"messages":[{role,content}]}`; `alpaca` —
    /// `{instruction,input?,output}`; or `text` — `{"text":"..."}`.
    #[serde(default)]
    pub format: Option<String>,
    /// The rows themselves, whose shape follows `format`. Left untyped because the
    /// three accepted row shapes are genuinely different objects.
    #[serde(default)]
    pub samples: Option<Vec<Value>>,
    /// Absolute path to a `.json` or `.jsonl` file of rows, as an alternative to
    /// sending them inline.
    #[serde(default)]
    pub path: Option<String>,
}

/// LoRA adapter shape (all optional — the worker fills its own defaults).
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct LoraSpec {
    /// Rank of the adapter. Higher trains more capacity and costs more VRAM.
    #[serde(default)]
    pub r: Option<u32>,
    /// LoRA alpha (scaling). Conventionally equal to, or twice, `r`.
    #[serde(default)]
    pub alpha: Option<u32>,
    /// Dropout applied to the adapter during training, 0.0–1.0.
    #[serde(default)]
    pub dropout: Option<f32>,
    /// Which projection modules to adapt, e.g. `["q_proj","v_proj"]`.
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
}

/// Training hyper-parameters (all optional — the worker fills its own defaults).
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct TrainingSpec {
    /// Passes over the dataset. Fractional values are allowed.
    #[serde(default)]
    pub epochs: Option<f32>,
    /// Hard cap on optimizer steps, which overrides `epochs` when set.
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// Learning rate, e.g. `0.0002`.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    /// Per-device batch size.
    #[serde(default)]
    pub batch_size: Option<u32>,
    /// Gradient-accumulation steps — raises the effective batch size without
    /// costing more VRAM.
    #[serde(default)]
    pub grad_accum: Option<u32>,
    /// Token context length each row is truncated to.
    #[serde(default)]
    pub max_seq_length: Option<u32>,
    /// Load the base model in 4-bit. On by default in the worker; this is the knob
    /// that decides whether a large model fits at all.
    #[serde(default)]
    pub load_in_4bit: Option<bool>,
    /// Seed, for a reproducible run.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// The remote Ryu node a `target: "remote"` job is dispatched to.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct RemoteTarget {
    /// Base URL of the remote node's Core, e.g. `https://gpu.example.com`.
    pub url: String,
    /// Bearer token for that node, when it requires one.
    #[serde(default)]
    pub token: Option<String>,
}

/// Request body for `POST /api/finetune/merge`.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct MergeBody {
    /// Name of an adapter directory under the worker's output dir, as listed by
    /// `GET /api/finetune/adapters`. Supply this or `adapter_path`.
    #[serde(default)]
    pub adapter_name: Option<String>,
    /// Absolute path to an adapter directory, for one that lives outside the
    /// worker's output dir. Supply this or `adapter_name`.
    #[serde(default)]
    pub adapter_path: Option<String>,
    /// Stem for the merged `.gguf`, which is also the name it is registered under
    /// as an installed model.
    #[serde(default)]
    pub output_name: Option<String>,
    /// The base model this adapter was trained on, recorded as provenance on the
    /// resulting installed model.
    #[serde(default)]
    pub base_model_id: Option<String>,
    /// GGUF quantization, e.g. `q4_k_m` (default), `q8_0`, or `f16`.
    #[serde(default)]
    pub quantization_method: Option<String>,
    /// Context length baked into the merged GGUF.
    #[serde(default)]
    pub max_seq_length: Option<u32>,
}

// ── Worker (Python Unsloth) HTTP proxy helpers ──────────────────────────────
// These replace Core's `sidecar::providers::unsloth::*` — the surface now targets
// `ctx.unsloth_url` directly instead of Core's hardcoded provider base URL.

/// Fetch the worker's hardware probe (`GET /health`). Used by `/api/finetune/capability`.
async fn worker_health(ctx: &FinetuneCtx) -> anyhow::Result<Value> {
    let url = ctx.worker("/health");
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /health returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// Start a fine-tune job on the worker (`POST /finetune`).
async fn worker_start(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune");
    let resp = ctx
        .worker_request(ctx.worker_client.post(&url))
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    let json = response_json_limited(resp).await?;
    if !status.is_success() {
        let err = json
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        anyhow::bail!("unsloth /finetune failed ({status}): {err}");
    }
    Ok(json)
}

/// All in-process job snapshots from the worker (`GET /finetune`).
async fn worker_list(ctx: &FinetuneCtx) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune");
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// One job snapshot from the worker (`GET /finetune/{id}`).
async fn worker_get(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune/{id} returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// Cancel a worker job (`DELETE /finetune/{id}`).
async fn worker_cancel(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx
        .worker_request(ctx.worker_client.delete(&url))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth cancel returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// Merge a trained adapter into a GGUF on the worker (`POST /finetune/merge`).
async fn worker_merge(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune/merge");
    let resp = ctx
        .worker_request(ctx.worker_client.post(&url))
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    let json = response_json_limited(resp).await?;
    if !status.is_success() {
        let err = json
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        anyhow::bail!("unsloth /merge failed ({status}): {err}");
    }
    Ok(json)
}

/// URL of the worker's SSE progress stream for a job.
fn worker_stream_url(ctx: &FinetuneCtx, id: &str) -> String {
    ctx.worker(&format!("/finetune/{id}/stream"))
}

// ── GPU gate ────────────────────────────────────────────────────────────────

/// Whether this node can train locally, plus a human reason when it cannot.
/// Heuristic: a discrete (non-unified) GPU detected by `nvidia-smi`. Unsloth
/// training requires an NVIDIA CUDA GPU; Apple unified memory and CPU-only boxes
/// cannot train (they fall back to a remote node — Unit 5).
fn local_capability(dev: &DeviceInfo) -> (bool, String) {
    if dev.gpu_name.is_some() && !dev.unified_memory {
        return (true, String::new());
    }
    let reason = if dev.unified_memory {
        "Apple Silicon / unified memory detected — Unsloth training needs an NVIDIA CUDA GPU. \
         Use a remote GPU node instead."
            .to_string()
    } else if dev.gpu_name.is_none() {
        "No NVIDIA GPU detected — Unsloth training needs a CUDA GPU. Use a remote GPU node instead."
            .to_string()
    } else {
        "This GPU is not supported for training — use a remote GPU node instead.".to_string()
    };
    (false, reason)
}

// ── App events ──────────────────────────────────────────────────────────────

/// Raise `event`, detached. A fan-out runs every subscribing hook and starts every
/// matching workflow, so it takes as long as the slowest consumer — it must never
/// sit inside the `/start` request that produced it, nor inside the per-job refresh
/// loop `/list` runs. Emitting is best-effort, so there is no outcome to await.
fn spawn_event(ctx: &FinetuneCtx, event: &'static str, payload: Value) {
    let events = ctx.events.clone();
    tokio::spawn(async move { events.emit(event, payload).await });
}

/// [`spawn_event`] plus an optional user-facing notification raised alongside the
/// fan-out (the Inbox shows a finished-job row with the Fine-tune icon).
fn spawn_event_with_notify(
    ctx: &FinetuneCtx,
    event: &'static str,
    payload: Value,
    notify: Option<ryu_app_events::NotifyHint>,
) {
    let events = ctx.events.clone();
    tokio::spawn(async move { events.emit_with_notify(event, payload, notify).await });
}

/// Announce a job the worker (or the remote node) actually accepted. Gated on a
/// non-empty id: without one there is nothing a consumer could poll, stream or
/// cancel, and announcing an unaddressable job is worse than announcing nothing.
fn announce_job_started(ctx: &FinetuneCtx, job: &FinetuneJob) {
    if job.id.is_empty() {
        return;
    }
    spawn_event(
        ctx,
        EVENT_JOB_STARTED,
        // No `state`: the worker answers the start call before its training thread
        // has necessarily flipped the job off `queued`, so the value here is a race,
        // and a `job.started` payload reading `queued` only invites a consumer to
        // branch on it. `GET /api/finetune/{job_id}` is the live state.
        json!({
            "job_id": job.id,
            "base_model": job.base_model,
            "output_name": job.output_name,
            "target": job.target,
            "created_at": job.created_at,
        }),
    );
}

/// Announce a job that reached a terminal state. `prior` is the record as it stood
/// before [`FinetuneStore::sync_from_snapshot`] claimed the transition, so this runs
/// on the poll that first observed the finish and never again.
///
/// Cancellation is deliberately silent: it is user-initiated, and [`cancel_value`]
/// writes `cancelled` through a path that raises nothing — the person who asked for
/// it already has the answer in their response.
fn announce_job_finished(
    ctx: &FinetuneCtx,
    prior: &FinetuneJob,
    state: &str,
    adapter_name: Option<&str>,
    output_ref: Option<&str>,
    error: Option<&str>,
) {
    match state {
        "succeeded" => spawn_event_with_notify(
            ctx,
            EVENT_JOB_SUCCEEDED,
            json!({
                "job_id": prior.id,
                "base_model": prior.base_model,
                "output_name": prior.output_name,
                "target": prior.target,
                // The stem the adapter was indexed under is exactly what `POST
                // /api/finetune/merge` takes as `adapter_name`, so "when a fine-tune
                // finishes, merge it into a GGUF" needs nothing beyond this payload.
                "adapter_name": adapter_name,
                "adapter_path": output_ref,
            }),
            // A training job finishing is the classic long-job-done notification.
            Some(
                ryu_app_events::NotifyHint::info(
                    format!(
                        "Fine-tune “{}” finished",
                        prior.output_name.as_deref().unwrap_or("job")
                    ),
                    Some(format!("Adapter ready for {}", prior.base_model)),
                )
                .with_level("success"),
            ),
        ),
        "failed" => spawn_event_with_notify(
            ctx,
            EVENT_JOB_FAILED,
            json!({
                "job_id": prior.id,
                "base_model": prior.base_model,
                "output_name": prior.output_name,
                "target": prior.target,
                "error": error,
            }),
            Some(
                ryu_app_events::NotifyHint::info(
                    format!(
                        "Fine-tune “{}” failed",
                        prior.output_name.as_deref().unwrap_or("job")
                    ),
                    error.map(str::to_owned),
                )
                .with_level("error"),
            ),
        ),
        // `queued` / `running` / `cancelled` are not finishes.
        _ => {}
    }
}

// ── Handlers ────────────────────────────────────────────────────────────────

/// `GET /api/finetune/capability` — what this node can train, for the desktop's
/// gating UI. Both the host GPU and the Python runtime must be ready before
/// local training is advertised: the worker's `/health` reports whether its
/// CUDA capability and optional training dependencies are actually available.
#[utoipa::path(
    get,
    path = "/api/finetune/capability",
    tag = "Finetune",
    summary = "what this node can train, for the desktop's gating UI",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn capability(State(ctx): State<FinetuneCtx>) -> impl IntoResponse {
    Json(capability_value(&ctx).await)
}

/// Shared capability probe value.
pub async fn capability_value(ctx: &FinetuneCtx) -> Value {
    let dev = DeviceInfo::detect();
    let sidecar = worker_health(ctx).await.ok();
    let (can_local, reason) = effective_local_capability(local_capability(&dev), sidecar.as_ref());
    json!({
        "can_train_local": can_local,
        "gpu": dev.gpu_name,
        "vram_bytes": dev.vram_bytes,
        "vram_human": dev.vram_human,
        "unified_memory": dev.unified_memory,
        "os": dev.os,
        "reason": reason,
        "sidecar": sidecar,
    })
}

/// A local GPU is necessary but not sufficient: the Python worker also needs a
/// supported CUDA runtime and the optional Unsloth training dependencies. Keep
/// this pure so the UI gate can be tested with fake hardware/health payloads.
fn effective_local_capability(
    (hardware_ready, hardware_reason): (bool, String),
    sidecar: Option<&Value>,
) -> (bool, String) {
    if !hardware_ready {
        return (false, hardware_reason);
    }

    let Some(sidecar) = sidecar else {
        return (
            false,
            "The Unsloth worker is unavailable — start it or use a remote GPU node.".to_string(),
        );
    };
    if sidecar
        .get("can_finetune")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (true, String::new());
    }

    let reason = sidecar
        .get("reason")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("The Unsloth worker cannot train on this node.");
    (false, format!("{reason}. Use a remote GPU node instead."))
}

/// `POST /api/finetune/start` — start a fine-tune job. Gates local training on the
/// GPU, proxies the request to the worker, and records the job. Body is forwarded
/// after canonicalizing the bounded model/output fields, plus an optional `target`
/// (`local` | `remote`).
#[utoipa::path(
    post,
    path = "/api/finetune/start",
    tag = "Finetune",
    summary = "start a fine-tune job (local GPU or remote node)",
    request_body = StartJobBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn start(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let header_key = match request_idempotency_key(&headers) {
        Ok(key) => key,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, Json(json!({ "error": error }))).into_response()
        }
    };
    match dispatch_with_idempotency_for(&ctx, &tenant, body, header_key.as_deref()).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err((code, err)) => (code, Json(err)).into_response(),
    }
}

/// Start a fine-tune job (local or remote), returning the worker/remote response
/// JSON on success or a `(status, error-json)` on failure.
pub async fn dispatch(ctx: &FinetuneCtx, body: Value) -> Result<Value, (StatusCode, Value)> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    dispatch_with_idempotency_for(ctx, &tenant, body, None).await
}

/// Start with an optional HTTP `Idempotency-Key`. The companion bridge also may
/// carry the same key as `idempotency_key` in its JSON body because it cannot add
/// arbitrary HTTP headers; a mismatch is rejected rather than choosing one key.
pub async fn dispatch_with_idempotency(
    ctx: &FinetuneCtx,
    body: Value,
    header_key: Option<&str>,
) -> Result<Value, (StatusCode, Value)> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    dispatch_with_idempotency_for(ctx, &tenant, body, header_key).await
}

pub async fn dispatch_with_idempotency_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    body: Value,
    header_key: Option<&str>,
) -> Result<Value, (StatusCode, Value)> {
    let snake_body_key = body.get("idempotency_key");
    let camel_body_key = body.get("idempotencyKey");
    if snake_body_key.is_some() && camel_body_key.is_some() && snake_body_key != camel_body_key {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "idempotency_key and idempotencyKey must match" }),
        ));
    }
    let body_key = match snake_body_key.or(camel_body_key) {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(
            normalize_idempotency_key(value)
                .map_err(|error| (StatusCode::BAD_REQUEST, json!({ "error": error })))?,
        ),
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "idempotency_key must be a string" }),
            ));
        }
    };
    let header_key = header_key
        .map(normalize_idempotency_key)
        .transpose()
        .map_err(|error| (StatusCode::BAD_REQUEST, json!({ "error": error })))?;
    if body_key.is_some() && header_key.is_some() && body_key != header_key {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "idempotency key in the header and body must match" }),
        ));
    }
    let idempotency_key = header_key.or(body_key);
    let mut request_body = body;
    if let Some(object) = request_body.as_object_mut() {
        object.remove("idempotency_key");
        object.remove("idempotencyKey");
    }
    let request_fingerprint = if idempotency_key.is_some() {
        Some(
            request_fingerprint(&request_body)
                .map_err(|error| (StatusCode::BAD_REQUEST, json!({ "error": error })))?,
        )
    } else {
        None
    };
    if let (Some(key), Some(fingerprint)) = (&idempotency_key, &request_fingerprint) {
        match ctx
            .store
            .claim_start(key, fingerprint)
            .await
            .map_err(|error| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({ "error": format!("could not claim finetune retry key: {error:#}") }),
                )
            })? {
            StartIdempotencyClaim::New => {}
            StartIdempotencyClaim::Replay(response) => return Ok(response),
            StartIdempotencyClaim::InProgress => {
                return Err((
                    StatusCode::CONFLICT,
                    json!({ "error": "another request with this idempotency key is in progress" }),
                ));
            }
            StartIdempotencyClaim::Conflict => {
                return Err((
                    StatusCode::CONFLICT,
                    json!({ "error": "idempotency key was already used for a different request" }),
                ));
            }
        }
    }

    let result = dispatch_unkeyed(ctx, tenant, request_body, idempotency_key.as_deref()).await;
    if let (Some(key), Some(fingerprint)) = (&idempotency_key, &request_fingerprint) {
        match &result {
            Ok(response) => {
                if let Err(error) = ctx.store.complete_start(key, fingerprint, response).await {
                    tracing::error!(%error, "could not persist finetune idempotency response");
                }
            }
            Err(_) => {
                let _ = ctx.store.release_start(key, fingerprint).await;
            }
        }
    }
    result
}

fn normalize_idempotency_key(raw: &str) -> Result<String, String> {
    let key = raw.trim();
    if key.is_empty()
        || key.len() > 128
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
    {
        return Err(
            "idempotency key must be 1-128 ASCII letters, digits, '.', '_' ':' or '-'".to_owned(),
        );
    }
    Ok(key.to_owned())
}

fn request_idempotency_key(headers: &HeaderMap) -> Result<Option<String>, String> {
    let values = headers.get_all("idempotency-key");
    if values.iter().count() > 1 {
        return Err("only one idempotency-key header is allowed".to_owned());
    }
    values
        .iter()
        .next()
        .map(|value| {
            value
                .to_str()
                .map_err(|_| "idempotency-key must be valid ASCII".to_owned())
                .and_then(normalize_idempotency_key)
        })
        .transpose()
}

fn request_fingerprint(body: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(body)
        .map_err(|error| format!("could not fingerprint request: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

async fn dispatch_unkeyed(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    body: Value,
    idempotency_key: Option<&str>,
) -> Result<Value, (StatusCode, Value)> {
    let base_model = body
        .get("base_model_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if base_model.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "missing `base_model_id`" }),
        ));
    }
    if !valid_base_model_id(&base_model) {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "base_model_id must be a Hugging Face namespace/model id" }),
        ));
    }
    if let Some(output_name) = body.get("output_name") {
        if !output_name.is_null() {
            let Some(output_name) = output_name.as_str().map(str::trim) else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    json!({ "error": "output_name must be a filename-safe string" }),
                ));
            };
            if !valid_output_name(output_name) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    json!({ "error": "output_name must use ASCII letters, digits, '.', '_' or '-' and be at most 128 characters" }),
                ));
            }
        }
    }

    // Forward the same canonical values we validated. Otherwise a harmless
    // surrounding space would pass Core's trimmed check but fail later in the
    // Python worker, while the durable record describes a different request.
    let mut body = body;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "base_model_id".to_owned(),
            Value::String(base_model.clone()),
        );
        if let Some(Value::String(output_name)) = object.get_mut("output_name") {
            *output_name = output_name.trim().to_owned();
        }
    }

    let target = body
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .trim()
        .to_string();

    match target.as_str() {
        "remote" => {
            return dispatch_remote(ctx, tenant, &body, base_model, idempotency_key).await
        }
        "local" => {}
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "target must be `local` or `remote`" }),
            ));
        }
    }

    // Gate local training on the node's GPU.
    let dev = DeviceInfo::detect();
    let (can_local, reason) = local_capability(&dev);
    if !can_local {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": reason, "can_train_local": false }),
        ));
    }

    match worker_start(ctx, &body).await {
        Ok(resp) => {
            let job_id = match response_job_id(&resp) {
                Ok(id) => id,
                Err(error) => {
                    return Err((StatusCode::BAD_GATEWAY, json!({ "error": error })));
                }
            };
            let job_state = response_job_state(&resp);
            let output_name = body
                .get("output_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let now = chrono::Utc::now().to_rfc3339();
            let job = FinetuneJob {
                id: job_id,
                tenant: FinetuneTenantScope::from_context(tenant),
                base_model,
                output_name,
                state: job_state,
                target,
                remote_url: None,
                remote_token: None,
                output_ref: None,
                error: None,
                created_at: now.clone(),
                updated_at: now,
            };
            match ctx.store.record_if_absent_for(tenant, &job).await {
                Ok(true) => {}
                Ok(false) => {
                    return Err((
                        StatusCode::BAD_GATEWAY,
                        json!({ "error": "worker returned a duplicate job_id" }),
                    ));
                }
                Err(e) => tracing::warn!("recording finetune job failed: {e:#}"),
            }
            announce_job_started(ctx, &job);
            Ok(resp)
        }
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            json!({
                "error": format!("{e:#}"),
                "hint": "Install the Unsloth fine-tuning tool from the Store, or run `bun run dev:unsloth`.",
            }),
        )),
    }
}

/// Dispatch a job to a remote Ryu Cloud GPU node (Unit 5). The desktop supplies
/// the target node's connection as `body.remote = { url, token }`; we forward the
/// job to that node's Core (forcing it to train *locally* there), then record it
/// with the remote coordinates so `get`/`stream`/`cancel` proxy back to it.
async fn dispatch_remote(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    body: &Value,
    base_model: String,
    idempotency_key: Option<&str>,
) -> Result<Value, (StatusCode, Value)> {
    let remote = body.get("remote");
    let url = remote
        .and_then(|r| r.get("url"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned();
    let token = match remote.and_then(|value| value.get("token")) {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => {
            let value = value.trim();
            if !valid_remote_token(value) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    json!({ "error": "remote.token must be a non-empty token of at most 4096 characters" }),
                ));
            }
            Some(value.to_owned())
        }
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "remote.token must be a string" }),
            ));
        }
    };
    if url.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "remote target needs `remote.url`" }),
        ));
    }
    let url = match validate_remote_base_url(&url) {
        Ok(url) => url,
        Err(error) => {
            return Err((StatusCode::BAD_REQUEST, json!({ "error": error })));
        }
    };

    // Forward verbatim but force the remote to train locally (it is the GPU node)
    // and drop our remote envelope so it doesn't recurse.
    let mut fwd = body.clone();
    if let Some(obj) = fwd.as_object_mut() {
        obj.insert("target".into(), json!("local"));
        obj.remove("remote");
    }

    let endpoint = format!("{url}/api/finetune/start");
    let remote_client = match remote_client_for_base(ctx, &url).await {
        Ok(client) => client,
        Err(error) => return Err((StatusCode::BAD_GATEWAY, json!({ "error": error }))),
    };
    let mut req = remote_client.post(&endpoint).json(&fwd);
    if let Some(key) = idempotency_key {
        req = req.header("Idempotency-Key", key);
    }
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let json_body = response_json_limited(resp)
                .await
                .unwrap_or_else(|_| json!({}));
            if !status.is_success() {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    json!({
                        "error": format!("remote node returned {status}"),
                        "detail": json_body,
                    }),
                ));
            }
            let job_id = match response_job_id(&json_body) {
                Ok(id) => id,
                Err(error) => {
                    return Err((StatusCode::BAD_GATEWAY, json!({ "error": error })));
                }
            };
            let job_state = response_job_state(&json_body);
            let output_name = body
                .get("output_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let now = chrono::Utc::now().to_rfc3339();
            let job = FinetuneJob {
                id: job_id,
                tenant: FinetuneTenantScope::from_context(tenant),
                base_model,
                output_name,
                state: job_state,
                target: "remote".to_string(),
                remote_url: Some(url),
                remote_token: token,
                output_ref: None,
                error: None,
                created_at: now.clone(),
                updated_at: now,
            };
            match ctx.store.record_if_absent_for(tenant, &job).await {
                Ok(true) => {}
                Ok(false) => {
                    return Err((
                        StatusCode::BAD_GATEWAY,
                        json!({ "error": "remote node returned a duplicate job_id" }),
                    ));
                }
                Err(e) => tracing::warn!("recording remote finetune job failed: {e:#}"),
            }
            // The GPU node runs `dispatch` for the forwarded job and announces it
            // against ITS OWN record; this announces the job as THIS node knows it.
            // Two Cores, two job ids, two events — not a double-emit.
            announce_job_started(ctx, &job);
            Ok(json_body)
        }
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            json!({ "error": format!("remote node unreachable: {e}") }),
        )),
    }
}

async fn remote_of_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    id: &str,
) -> Result<Option<(String, Option<String>)>, String> {
    match ctx.store.get_for(tenant, id).await {
        Ok(Some(job)) if job.target == "remote" => {
            let url = job
                .remote_url
                .ok_or_else(|| format!("remote finetune job '{id}' has no remote URL"))?;
            let normalized = validate_remote_base_url(&url).map_err(|error| {
                tracing::warn!(job_id = %id, %error, "unsafe persisted remote finetune URL");
                format!("remote finetune job '{id}' has an unsafe remote URL")
            })?;
            let token = job
                .remote_token
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty());
            Ok(Some((normalized, token)))
        }
        Ok(_) => Ok(None),
        Err(error) => Err(format!("could not load finetune job '{id}': {error:#}")),
    }
}

fn response_job_id(response: &Value) -> Result<String, String> {
    response
        .get("job_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| valid_job_id(id))
        .map(str::to_owned)
        .ok_or_else(|| {
            "remote or worker response did not contain a valid job_id (ASCII letters, digits, '.', '_' or '-' only; max 128 characters)".to_owned()
        })
}

/// Job ids are interpolated into worker and remote URL paths. Keep the accepted
/// alphabet deliberately narrower than a generic URL segment so a compromised
/// worker or remote node cannot turn its response into a path/query injection.
fn valid_job_id(id: &str) -> bool {
    !matches!(id, "." | "..")
        && !id.is_empty()
        && id.chars().count() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// The worker passes this directly to the model loader. Restrict it to the
/// Hugging Face `namespace/model` form so an authenticated caller cannot turn
/// the training sidecar into an arbitrary local-path or URL loader.
fn valid_base_model_id(id: &str) -> bool {
    let mut parts = id.split('/');
    let Some(namespace) = parts.next() else {
        return false;
    };
    let Some(model) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    [namespace, model].into_iter().all(|part| {
        !part.is_empty()
            && part.chars().count() <= 128
            && part
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    })
}

fn valid_output_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= 128
        && name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn valid_remote_token(token: &str) -> bool {
    !token.is_empty() && token.len() <= 4096 && !token.chars().any(char::is_control)
}

fn response_job_state(response: &Value) -> String {
    response
        .get("state")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|state| {
            matches!(
                *state,
                "queued" | "running" | "succeeded" | "failed" | "cancelled"
            )
        })
        .unwrap_or("running")
        .to_owned()
}

/// Mirror a worker snapshot's mutable fields back into the persisted record so the
/// store stays current (and terminal jobs survive a Core/worker restart).
///
/// This is also the ONLY place a finish is ever noticed: the worker owns the
/// training and nothing here polls it in the background, so a job's terminal state
/// becomes known — and its event fires — on the next `/list` or `/:id` read.
async fn persist_from_snapshot_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    id: &str,
    snap: &Value,
) {
    let job_state = snap.get("state").and_then(Value::as_str).unwrap_or("");
    if job_state.is_empty() {
        return;
    }
    let output_ref = snap.get("output_dir").and_then(Value::as_str);
    let error = snap.get("error").and_then(Value::as_str);
    let now = chrono::Utc::now().to_rfc3339();
    // One round-trip that writes the new state AND hands back the record it
    // replaced, so `prior.state != job_state` identifies this call as the one that
    // moved the job (see `sync_from_snapshot` on why the compare must ride the
    // write). `prior` also carries the fields a poll never touches, so the adapter
    // index below needs no second read.
    let prior = match ctx
        .store
        .sync_from_snapshot_for(tenant, id, job_state, output_ref, error, &now)
        .await
    {
        Ok(prior) => prior,
        Err(e) => {
            tracing::warn!("syncing finetune job {id} failed: {e:#}");
            return;
        }
    };
    // An id we have no record of: the worker still remembers a job this node never
    // stored (or stored under another profile). Nothing to update, nothing to own.
    let Some(prior) = prior else {
        return;
    };

    // On success, index the produced adapter (Unit 3). Idempotent on stem, and run
    // on EVERY succeeded poll rather than only the transition, so a catalog entry
    // lost to a failed write or an out-of-band delete is restored by the next read.
    let mut adapter_name = None;
    if job_state == "succeeded" {
        if let Some(out) = output_ref {
            let stem = prior.output_name.clone().unwrap_or_else(|| {
                std::path::Path::new(out)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| id.to_string())
            });
            if let Err(e) = adapters::record(InstalledAdapter {
                stem: stem.clone(),
                base_model: prior.base_model.clone(),
                job_id: id.to_string(),
                path: out.to_string(),
                created_at: now.clone(),
            }) {
                tracing::warn!("indexing adapter for job {id} failed: {e:#}");
            }
            adapter_name = Some(stem);
        }
    }

    // Announce only the move, and only after the adapter is on the index — a hook
    // that reacts by merging must find the adapter already listed.
    if prior.state != job_state {
        announce_job_finished(
            ctx,
            &prior,
            job_state,
            adapter_name.as_deref(),
            output_ref,
            error,
        );
    }
}

/// `GET /api/finetune/list` — the durable job list. Refreshes each job's state
/// from the worker when reachable (so running jobs show live state), then returns
/// the persisted records.
#[utoipa::path(
    get,
    path = "/api/finetune/list",
    tag = "Finetune",
    summary = "the durable job list (overlaid with live worker state)",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn list(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
) -> impl IntoResponse {
    match list_value_for(&ctx, &tenant).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response(),
    }
}

/// Shared job-list logic (`{ jobs: [...] }`). Overlays live worker snapshots onto
/// the durable store.
pub async fn list_value(ctx: &FinetuneCtx) -> Result<Value, String> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    list_value_for(ctx, &tenant).await
}

pub async fn list_value_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
) -> Result<Value, String> {
    let owned_ids: std::collections::HashSet<String> = ctx
        .store
        .list_for(tenant)
        .await
        .map_err(|e| format!("{e:#}"))?
        .into_iter()
        .map(|job| job.id)
        .collect();
    if let Ok(Value::Array(snaps)) = worker_list(ctx).await {
        for snap in &snaps {
            if let Some(id) = snap.get("id").and_then(Value::as_str) {
                if owned_ids.contains(id) {
                    persist_from_snapshot_for(ctx, tenant, id, snap).await;
                }
            }
        }
    }
    ctx.store
        .list_for(tenant)
        .await
        .map(|jobs| json!({ "jobs": jobs }))
        .map_err(|e| format!("{e:#}"))
}

/// `GET /api/finetune/:id` — one job. Prefers the worker's live snapshot (and
/// persists it); falls back to the stored record when the worker is unreachable.
#[utoipa::path(
    get,
    path = "/api/finetune/{id}",
    tag = "Finetune",
    summary = "one job (live worker snapshot, else stored record)",
    params(("id" = String, Path)),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn get_job(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
    Path(id): Path<String>,
) -> Response {
    match get_value_for(&ctx, &tenant, &id).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared single-job snapshot. Prefers the worker's (or remote node's) live
/// snapshot, persisting it; falls back to the stored record.
pub async fn get_value(ctx: &FinetuneCtx, id: &str) -> Result<Value, (StatusCode, Value)> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    get_value_for(ctx, &tenant, id).await
}

pub async fn get_value_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    id: &str,
) -> Result<Value, (StatusCode, Value)> {
    if !valid_job_id(id) {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "job id contains invalid characters" }),
        ));
    }
    match ctx.store.get_for(tenant, id).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                json!({ "error": format!("unknown job '{id}'") }),
            ))
        }
        Err(error) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": format!("{error:#}") }),
            ));
        }
    }
    let remote = match remote_of_for(ctx, tenant, id).await {
        Ok(remote) => remote,
        Err(error) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, json!({ "error": error })));
        }
    };
    if let Some((base, token)) = remote {
        // Remote job: proxy the snapshot from the remote node's Core.
        let client = remote_client_for_base(ctx, &base).await.ok();
        let Some(client) = client else {
            return match ctx.store.get_for(tenant, id).await {
                Ok(Some(job)) => serde_json::to_value(job).map_err(|error| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        json!({ "error": format!("{error:#}") }),
                    )
                }),
                Ok(None) => Err((
                    StatusCode::NOT_FOUND,
                    json!({ "error": format!("unknown job '{id}'") }),
                )),
                Err(error) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({ "error": format!("{error:#}") }),
                )),
            };
        };
        let mut req = client.get(format!("{base}/api/finetune/{id}"));
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(snap) = response_json_limited(resp).await {
                    persist_from_snapshot_for(ctx, tenant, id, &snap).await;
                    return Ok(snap);
                }
            }
        }
        // Remote unreachable — fall through to the stored record below.
    } else if let Ok(snap) = worker_get(ctx, id).await {
        persist_from_snapshot_for(ctx, tenant, id, &snap).await;
        return Ok(snap);
    }
    match ctx.store.get_for(tenant, id).await {
        Ok(Some(job)) => serde_json::to_value(job).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": format!("{e:#}") }),
            )
        }),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            json!({ "error": format!("unknown job '{id}'") }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "error": format!("{e:#}") }),
        )),
    }
}

/// `DELETE /api/finetune/:id` — cooperative cancel. Proxies to the worker and
/// marks the stored record cancelled.
#[utoipa::path(
    delete,
    path = "/api/finetune/{id}",
    tag = "Finetune",
    summary = "cooperative cancel (proxied to the worker/remote node)",
    params(("id" = String, Path)),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn cancel(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
    Path(id): Path<String>,
) -> Response {
    match cancel_value_for(&ctx, &tenant, &id).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared cooperative-cancel. Proxies to the worker (or remote node) and marks the
/// stored record cancelled.
pub async fn cancel_value(ctx: &FinetuneCtx, id: &str) -> Result<Value, (StatusCode, Value)> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    cancel_value_for(ctx, &tenant, id).await
}

pub async fn cancel_value_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    id: &str,
) -> Result<Value, (StatusCode, Value)> {
    if !valid_job_id(id) {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "job id contains invalid characters" }),
        ));
    }
    match ctx.store.get_for(tenant, id).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return Err((
                StatusCode::NOT_FOUND,
                json!({ "error": format!("unknown job '{id}'") }),
            ))
        }
        Err(error) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": format!("{error:#}") }),
            ));
        }
    }
    match remote_of_for(ctx, tenant, id).await {
        Err(error) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, json!({ "error": error })));
        }
        Ok(Some((base, token))) => {
            let client = match remote_client_for_base(ctx, &base).await {
                Ok(client) => client,
                Err(error) => return Err((StatusCode::BAD_GATEWAY, json!({ "error": error }))),
            };
            let mut req = client.delete(format!("{base}/api/finetune/{id}"));
            if let Some(t) = &token {
                req = req.bearer_auth(t);
            }
            return match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body = response_json_limited(resp)
                        .await
                        .unwrap_or_else(|_| json!({ "cancelling": true }));
                    let now = chrono::Utc::now().to_rfc3339();
                    let _ = ctx
                        .store
                        .update_state_for(tenant, id, "cancelled", None, None, &now)
                        .await;
                    Ok(body)
                }
                Ok(resp) => Err((
                    StatusCode::BAD_GATEWAY,
                    json!({ "error": format!("remote node returned {}", resp.status()) }),
                )),
                Err(e) => Err((
                    StatusCode::BAD_GATEWAY,
                    json!({ "error": format!("remote node unreachable: {e}") }),
                )),
            };
        }
        Ok(None) => {}
    }
    match worker_cancel(ctx, id).await {
        Ok(resp) => {
            let now = chrono::Utc::now().to_rfc3339();
            let _ = ctx
                .store
                .update_state_for(tenant, id, "cancelled", None, None, &now)
                .await;
            Ok(resp)
        }
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            json!({ "error": format!("{e:#}") }),
        )),
    }
}

/// `GET /api/finetune/adapters` — the installed trained adapters (Unit 3), with
/// provenance (base model + producing job).
#[utoipa::path(
    get,
    path = "/api/finetune/adapters",
    tag = "Finetune",
    summary = "the installed trained adapters, with provenance",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn list_adapters(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
) -> impl IntoResponse {
    let job_ids: std::collections::HashSet<String> = ctx
        .store
        .list_for(&tenant)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|job| job.id)
        .collect();
    let adapters = adapters::load_present()
        .into_iter()
        .filter(|adapter| job_ids.contains(&adapter.job_id))
        .collect::<Vec<_>>();
    Json(json!({ "adapters": adapters }))
}

/// `POST /api/finetune/merge` — merge a trained adapter into a GGUF (Unit 4), then
/// register it as an installed model so it is selectable as the active chat model
/// via the existing `POST /api/models/active` (llama.cpp) path. Body:
/// `{ adapter_name | adapter_path, output_name?, base_model_id?, quantization_method? }`.
#[utoipa::path(
    post,
    path = "/api/finetune/merge",
    tag = "Finetune",
    summary = "merge a trained adapter into a GGUF + register it",
    request_body = MergeBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn merge(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
    Json(body): Json<Value>,
) -> Response {
    match merge_value_for(&ctx, &tenant, body).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared adapter→GGUF merge. Registers the merged GGUF as an installed model on
/// success (idempotent, into the shared `${RYU_DIR}/installed-models.json`).
pub async fn merge_value(ctx: &FinetuneCtx, body: Value) -> Result<Value, (StatusCode, Value)> {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    merge_value_for(ctx, &tenant, body).await
}

pub async fn merge_value_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    body: Value,
) -> Result<Value, (StatusCode, Value)> {
    if body.get("adapter_name").and_then(Value::as_str).is_none()
        && body.get("adapter_path").and_then(Value::as_str).is_none()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "need `adapter_name` or `adapter_path`" }),
        ));
    }

    let adapter_name = body.get("adapter_name").and_then(Value::as_str);
    let adapter_path = body.get("adapter_path").and_then(Value::as_str);
    let adapter = adapters::load_present().into_iter().find(|adapter| {
        adapter_name.is_some_and(|name| name == adapter.stem)
            || adapter_path.is_some_and(|path| path == adapter.path)
    });
    let allowed = match adapter {
        Some(adapter) => ctx
            .store
            .get_for(tenant, &adapter.job_id)
            .await
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({ "error": format!("{error:#}") }),
                )
            })?
            .is_some(),
        None => false,
    };
    if !allowed && (tenant.owner_user_id.is_some() || tenant.org_id.is_some()) {
        return Err((
            StatusCode::NOT_FOUND,
            json!({ "error": "adapter not found for this tenant" }),
        ));
    }

    match worker_merge(ctx, &body).await {
        Ok(resp) => {
            // Register the merged GGUF so it shows up as an installed model.
            if let (Some(stem), Some(_path)) = (
                resp.get("stem").and_then(Value::as_str),
                resp.get("gguf_path").and_then(Value::as_str),
            ) {
                let base = resp
                    .get("base_model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let model = InstalledModel {
                    repo_id: base.clone(),
                    filename: format!("{stem}.gguf"),
                    stem: stem.to_string(),
                    size_bytes: resp.get("size_bytes").and_then(Value::as_u64),
                    format: ModelFormat::Gguf,
                    mmproj: None,
                    // Provenance: this GGUF is a merged fine-tune of `base`.
                    finetune_base: Some(base),
                };
                if let Err(e) = installed::record(model) {
                    tracing::warn!("recording merged model '{stem}' failed: {e:#}");
                }
            }
            Ok(resp)
        }
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            json!({ "error": format!("{e:#}") }),
        )),
    }
}

/// `GET /api/finetune/:id/stream` — proxy the worker's SSE progress stream straight
/// through as `text/event-stream` (no re-parsing of frames).
#[utoipa::path(
    get,
    path = "/api/finetune/{id}/stream",
    tag = "Finetune",
    summary = "proxy the worker's SSE progress stream",
    params(("id" = String, Path)),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn stream(
    State(ctx): State<FinetuneCtx>,
    Extension(tenant): Extension<FinetuneTenantContext>,
    Path(id): Path<String>,
) -> Response {
    stream_response_for(&ctx, &tenant, &id).await
}

/// Shared SSE proxy for a job's progress stream. Streams the worker's (or remote
/// node's) `text/event-stream` frames through verbatim.
pub async fn stream_response(ctx: &FinetuneCtx, id: &str) -> Response {
    let tenant = FinetuneTenantContext::local(ctx.store.node_id());
    stream_response_for(ctx, &tenant, id).await
}

pub async fn stream_response_for(
    ctx: &FinetuneCtx,
    tenant: &FinetuneTenantContext,
    id: &str,
) -> Response {
    if !valid_job_id(id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "job id contains invalid characters" })),
        )
            .into_response();
    }
    match ctx.store.get_for(tenant, id).await {
        Ok(Some(_)) => {}
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": format!("unknown job '{id}'") })),
            )
                .into_response()
        }
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("{error:#}") })),
            )
                .into_response()
        }
    }
    // Remote jobs stream from the remote node's Core; local jobs from the worker.
    let (url, token, client, remote) = match remote_of_for(ctx, tenant, id).await {
        Ok(Some((base, token))) => {
            let client = match remote_client_for_base(ctx, &base).await {
                Ok(client) => client,
                Err(error) => {
                    return (StatusCode::BAD_GATEWAY, Json(json!({ "error": error })))
                        .into_response();
                }
            };
            (
                format!("{base}/api/finetune/{id}/stream"),
                token,
                client,
                true,
            )
        }
        Ok(None) => (
            worker_stream_url(ctx, id),
            None,
            ctx.worker_client.clone(),
            false,
        ),
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": error })),
            )
                .into_response();
        }
    };
    let mut req = client.get(&url);
    if !remote {
        req = ctx.worker_request(req);
    }
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    match req.send().await {
        Ok(resp) if resp.status().is_success() => Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from_stream(resp.bytes_stream()))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
        Ok(resp) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("finetune stream returned {}", resp.status()) })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("finetune source not reachable: {e}") })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_remote_base_url, response_job_id, valid_base_model_id, valid_output_name,
    };
    use serde_json::json;

    // ── OpenAPI document ───────────────────────────────────────────────────────

    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration
    /// rather than against a second list that could drift from it.
    fn openapi_manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The manifest sidecar whose HTTP surface this router serves: the one that
    /// declares an `http.mount`. Selected BY mount rather than by index because an app
    /// may declare a second, mountless sidecar (finetune already does), and
    /// `sidecars[0]` would then quietly start asserting against the wrong process.
    fn mounted_sidecar() -> serde_json::Value {
        openapi_manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten
    /// into the form the OpenAPI document uses (absolute, in `{param}` form).
    ///
    /// The two forms differ ON PURPOSE — the router registers paths relative to the
    /// mount because Core nests it there, while the `#[utoipa::path]` annotations carry
    /// the absolute EXTERNAL path a caller actually hits. Normalise here; do not
    /// "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_is_served_and_non_empty() {
        // The doc is no longer dead code: Core fetches it to derive tools.
        assert!(!super::openapi().paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield. Core's `ext_api::lower` keeps only the
        // document operations the manifest ALSO declares, so a declared route with no
        // `#[utoipa::path]` annotation is a tool that silently never exists — nothing
        // errors, the agent simply cannot call it. (The other direction is harmless: an
        // annotated path the manifest does not declare is dropped by the same filter.)
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = super::openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    // ── Request-body schemas ───────────────────────────────────────────────────

    /// The one pointer Core reads to give a derived write tool its arguments.
    fn body_schema(wire: &serde_json::Value, path: &str, method: &str) -> serde_json::Value {
        wire.pointer(&format!(
            "/paths/{}/{method}/requestBody/content/application~1json/schema",
            path.replace('/', "~1")
        ))
        .unwrap_or_else(|| panic!("{method} {path} must declare a JSON request body"))
        .clone()
    }

    #[test]
    fn post_routes_document_their_request_body() {
        // The regression this locks down: both annotations here used to say
        // `request_body = serde_json::Value`, which serialises to an untyped schema.
        // Core derives a tool per operation and fills `input_schema` from THIS node,
        // so an untyped body produced a tool the model could discover, could call,
        // and could never pass a single argument to — discoverable and useless, with
        // nothing logged to explain it. Training is the most expensive thing this
        // node does; an agent that cannot name a base model cannot start one.
        //
        // A `$ref` is the CORRECT and expected shape, not a near-miss: Core's
        // `openapi_import::resolve_ref` resolves it against `components.schemas`
        // before reading `properties`. So accept either a ref or inlined properties;
        // asserting "inlined" would fail on a healthy document.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, method) in [
            ("/api/finetune/start", "post"), // start -> StartJobBody
            ("/api/finetune/merge", "post"), // merge -> MergeBody
        ] {
            let schema = body_schema(&wire, path, method);
            assert!(
                schema.get("$ref").is_some() || schema.get("properties").is_some(),
                "a derived write tool for {method} {path} would have no arguments: {schema}"
            );
        }
    }

    #[test]
    fn every_request_body_ref_resolves_against_components() {
        // The half of the retrofit that a `$ref`-shaped assertion alone cannot see:
        // a `$ref` pointing at a schema that was never registered in
        // `components(schemas(...))` looks identical in the operation and still
        // yields zero arguments once Core tries to resolve it. Walk every request
        // body in the document and check the target actually exists and carries
        // properties.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let paths = wire["paths"].as_object().expect("paths must be an object");
        let mut checked = 0usize;
        for (path, item) in paths {
            for (method, op) in item.as_object().expect("a path item is an object") {
                let Some(schema) = op.pointer("/requestBody/content/application~1json/schema")
                else {
                    continue;
                };
                let Some(reference) = schema.get("$ref").and_then(|r| r.as_str()) else {
                    // Inlined schemas are fine as long as they describe something.
                    // The failure this catches in practice is `request_body =
                    // Option<T>`, which utoipa renders as a nullable `oneOf` wrapper:
                    // Core resolves only a TOP-LEVEL `$ref`, so the wrapper reaches the
                    // importer unresolved and contributes no properties at all.
                    assert!(
                        schema.get("properties").is_some(),
                        "{method} {path} has a request-body schema Core cannot read \
                         (a `oneOf` here means `request_body = Option<T>` — use the \
                         plain type): {schema}"
                    );
                    checked += 1;
                    continue;
                };
                let name = reference
                    .strip_prefix("#/components/schemas/")
                    .unwrap_or_else(|| {
                        panic!("unexpected ref form '{reference}' at {method} {path}")
                    });
                let target = wire
                    .pointer(&format!("/components/schemas/{name}"))
                    .unwrap_or_else(|| {
                        panic!(
                            "{method} {path} refs '{name}' but it is missing from \
                             components.schemas — add it to components(schemas(..))"
                        )
                    });
                assert!(
                    target.get("properties").is_some(),
                    "{method} {path} refs '{name}', which has no properties: {target}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 2,
            "expected both write routes to carry a body schema, saw {checked}"
        );
    }

    #[test]
    fn a_nested_struct_argument_is_self_describing() {
        // `StartJobBody::training` is an `Option<TrainingSpec>`. utoipa wraps that in
        // `oneOf: [null, …]`, and Core resolves a `$ref` only at the TOP of a node —
        // so a ref nested inside the wrapper would reach the model as an opaque
        // pointer. `#[schema(inline)]` is what makes the real hyper-parameters
        // visible; this test fails the moment someone removes it.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let training = wire
            .pointer("/components/schemas/StartJobBody/properties/training")
            .expect("StartJobBody must document `training`");
        let variants = training["oneOf"]
            .as_array()
            .expect("an optional struct field is a oneOf wrapper");
        let object = variants
            .iter()
            .find(|v| v["type"] == "object")
            .expect("the non-null variant must be an inlined object, not a $ref");
        for field in ["epochs", "learning_rate", "max_steps", "load_in_4bit"] {
            assert!(
                object["properties"].get(field).is_some(),
                "TrainingSpec::{field} is invisible to the model: {object}"
            );
        }
    }

    #[test]
    fn a_required_struct_argument_stays_resolvable() {
        // `dataset` is NOT optional, so utoipa emits a bare `$ref` at the top of the
        // property node — the one shape Core's one-level nested resolution does
        // follow. No `#[schema(inline)]` needed; what IS needed is the component
        // entry it points at, which this asserts end to end.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let reference = wire
            .pointer("/components/schemas/StartJobBody/properties/dataset/$ref")
            .and_then(|r| r.as_str())
            .expect("`dataset` must be a top-level $ref Core can follow");
        assert_eq!(reference, "#/components/schemas/DatasetSpec");
        assert!(
            wire.pointer("/components/schemas/DatasetSpec/properties/samples")
                .is_some(),
            "DatasetSpec is not registered, so the ref dangles and `dataset` is opaque"
        );
        // `samples` is a `Vec<serde_json::Value>`. utoipa inlines `Value` as an
        // any-schema; if it ever emitted a NAMED component instead, the ref would sit
        // two hops deep (property → items) — past the single level Core expands
        // inside an already-resolved object — and reach the model as an opaque
        // pointer that every other assertion here would still pass.
        assert!(
            wire.pointer("/components/schemas/DatasetSpec/properties/samples/items/$ref")
                .is_none(),
            "`samples` items became a $ref Core will not expand"
        );
    }

    #[test]
    fn schema_descriptions_carry_no_internal_rationale() {
        // utoipa lifts a STRUCT's doc comment into the schema's own `description`,
        // exactly as it lifts field docs into property descriptions — so a `///`
        // paragraph explaining why a type is not the axum extractor would ship to the
        // model as part of the tool. The convention that prevents it: one `///` line
        // naming the body, and every rationale paragraph below it demoted to `//`.
        // Wrapped prose is fine — the tell is VOCABULARY, so this greps for the Rust
        // implementation words that only ever appear in rationale, never in something
        // written for a caller.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let schemas = wire["components"]["schemas"]
            .as_object()
            .expect("components.schemas must be an object");
        for (name, schema) in schemas {
            let mut descriptions = vec![schema.get("description")];
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                descriptions.extend(props.values().map(|p| p.get("description")));
            }
            for description in descriptions
                .into_iter()
                .flatten()
                .filter_map(|d| d.as_str())
            {
                for leak in ["axum", "utoipa", "extractor", "Deserialize", "serde_json"] {
                    assert!(
                        !description.contains(leak),
                        "{name} ships the word '{leak}' to the model in a schema \
                         description — demote that rationale from `///` to `//`: \
                         {description:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn local_capability_requires_a_ready_worker() {
        let hardware = (true, String::new());
        assert_eq!(
            super::effective_local_capability(
                hardware.clone(),
                Some(&serde_json::json!({
                    "can_finetune": true,
                    "reason": ""
                }))
            ),
            (true, String::new())
        );

        let (ready, reason) = super::effective_local_capability(
            hardware.clone(),
            Some(&serde_json::json!({
                "can_finetune": false,
                "reason": "unsloth is not installed"
            })),
        );
        assert!(!ready);
        assert!(reason.contains("unsloth is not installed"));

        let (ready, reason) = super::effective_local_capability(hardware, None);
        assert!(!ready);
        assert!(reason.contains("worker is unavailable"));
    }

    #[test]
    fn body_field_docs_reach_the_schema_as_argument_descriptions() {
        // Doc comments on the body-struct fields are the whole payoff of the
        // retrofit: they are the only prose the model reads when choosing arguments.
        // utoipa lifts them into `description`, so a future edit that drops them
        // silently degrades tool-call quality with no compile error.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let base = wire
            .pointer("/components/schemas/StartJobBody/properties/base_model_id/description")
            .and_then(|d| d.as_str())
            .unwrap_or_default();
        assert!(
            base.contains("Hugging Face repo id"),
            "StartJobBody::base_model_id lost its doc comment, got {base:?}"
        );
    }

    #[test]
    fn only_the_body_carrying_routes_declare_a_request_body() {
        // The other direction of the same bug. `capability`/`list`/`adapters`, the
        // single-job read, the SSE stream, and the DELETE cancel take no JSON body at
        // all — their handlers have no `Json` extractor. Declaring one would document
        // something the endpoint never reads, and (before the retrofit) an untyped one
        // at that.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for (path, method) in [
            ("/api/finetune/capability", "get"),
            ("/api/finetune/list", "get"),
            ("/api/finetune/adapters", "get"),
            ("/api/finetune/{id}", "get"),
            ("/api/finetune/{id}", "delete"),
            ("/api/finetune/{id}/stream", "get"),
        ] {
            let op = wire
                .pointer(&format!("/paths/{}/{method}", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{method} {path} must have an operation"));
            assert!(
                op.get("requestBody").is_none(),
                "{method} {path} takes no body but the document declares one"
            );
        }
        // …and the id the per-job routes DO take must still be an argument.
        for path in ["/api/finetune/{id}", "/api/finetune/{id}/stream"] {
            let op = wire
                .pointer(&format!("/paths/{}/get", path.replace('/', "~1")))
                .expect("a GET operation");
            assert!(
                op.get("parameters").is_some(),
                "{path} must still document its path id"
            );
        }
    }

    #[test]
    fn remote_targets_reject_private_hosts_and_url_smuggling() {
        assert!(normalize_remote_base_url("https://127.0.0.1").is_err());
        assert!(normalize_remote_base_url("https://169.254.169.254").is_err());
        assert!(normalize_remote_base_url("https://gpu.example.test/path").is_err());
        assert!(normalize_remote_base_url("https://user:pass@gpu.example.test").is_err());
        assert!(normalize_remote_base_url("https://gpu.example.test/?token=secret").is_err());
        assert_eq!(
            normalize_remote_base_url("https://GPU.Example.Test:8443/").unwrap(),
            "https://gpu.example.test:8443"
        );
    }

    #[test]
    fn worker_response_must_contain_a_bounded_job_id() {
        assert_eq!(
            response_job_id(&json!({ "job_id": "job-1" })).unwrap(),
            "job-1"
        );
        assert!(response_job_id(&json!({})).is_err());
        assert!(response_job_id(&json!({ "job_id": "   " })).is_err());
        assert!(response_job_id(&json!({ "job_id": "x".repeat(129) })).is_err());
        for id in [
            "../etc/passwd",
            "job?redirect=1",
            "job/child",
            "job%2Fchild",
            ".",
            "..",
        ] {
            assert!(
                response_job_id(&json!({ "job_id": id })).is_err(),
                "path-significant job id must be rejected: {id}"
            );
        }
    }

    #[test]
    fn model_and_output_names_cannot_become_loader_or_path_inputs() {
        assert!(valid_base_model_id("unsloth/llama-3-8b"));
        for id in [
            "/etc/passwd",
            "file:///tmp/model",
            "../model",
            "org/model/extra",
        ] {
            assert!(!valid_base_model_id(id), "unsafe model id accepted: {id}");
        }
        assert!(valid_output_name("adapter-v1"));
        for name in ["../adapter", ".", "adapter/name", "adapter\0name"] {
            assert!(
                !valid_output_name(name),
                "unsafe output name accepted: {name:?}"
            );
        }
    }
}
