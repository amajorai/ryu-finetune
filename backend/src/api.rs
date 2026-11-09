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
    FinetuneJob, FinetuneStore, FinetuneTenantContext, FinetuneTenantScope, StartIdempotencyClaim,
};

mod bodies;
mod capability;
mod dispatch;
mod ids;
mod jobs;
mod merge;
mod remote;
mod remote_url;
mod stream;
#[cfg(test)]
mod tests;
mod worker;
mod worker_reason;

pub use bodies::*;
pub use capability::*;
pub use dispatch::*;
pub use ids::*;
pub use jobs::*;
pub use merge::*;
use remote::*;
use remote_url::*;
pub use stream::*;
use worker::*;
use worker_reason::*;

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
/// Bound on establishing a TCP/TLS connection to the worker or a remote node.
/// Streams and the long-running merge still have no total deadline; this only
/// stops a blackholed host from pinning a handler forever.
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Total deadline for short control calls (health, start, list, get, cancel).
/// The worker answers these immediately (training runs on its own thread), so a
/// call still pending after this is a hung peer, not slow work.
const CONTROL_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

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
    /// `<data dir>/models/adapters`, the folder the worker creates each local
    /// job's `<artifact_id>/` adapter directory in. The adapter catalog records
    /// only paths this resolves to (see `owned_adapter_dir`).
    pub adapters_root: std::path::PathBuf,
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
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .expect("redirect-disabled finetune worker client must build");
        let remote_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
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
            adapters_root: ryu_sidecar_runtime::models_root()
                .join(ryu_sidecar_runtime::layout::Modality::Adapters.dir_name()),
        }
    }

    /// Point the adapter catalog at a different adapters folder (tests; a worker
    /// started with a non-default `RYU_UNSLOTH_ADAPTERS_DIR`).
    #[must_use]
    pub fn with_adapters_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.adapters_root = root.into();
        self
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
