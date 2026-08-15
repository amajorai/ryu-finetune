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

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

use ryu_model_catalog::device::DeviceInfo;
use ryu_model_catalog::installed::{self, InstalledModel};
use ryu_model_format::ModelFormat;

use crate::adapters::{self, InstalledAdapter};
use crate::store::{FinetuneJob, FinetuneStore};

/// Default base URL of the Python Unsloth training worker (overridable via
/// `RYU_UNSLOTH_URL`). The `@ryu/finetune` app's manifest binds the worker on
/// this same loopback port (`8086`).
pub const DEFAULT_UNSLOTH_URL: &str = "http://127.0.0.1:8086";

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
    pub unsloth_url: String,
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
        Self {
            store,
            client,
            unsloth_url: url,
            events,
        }
    }

    /// Absolute URL of a Python worker endpoint (`path` starts with `/`).
    fn worker(&self, path: &str) -> String {
        format!("{}{path}", self.unsloth_url)
    }
}

/// Build the `/api/finetune/*` router with its own state baked in, returning a
/// state-less `Router<()>` the host nests at `/api/finetune`.
pub fn routes(ctx: FinetuneCtx) -> Router<()> {
    Router::new()
        .route("/capability", get(capability))
        .route("/start", post(start))
        .route("/list", get(list))
        .route("/adapters", get(list_adapters))
        .route("/merge", post(merge))
        .route("/:id", get(get_job).delete(cancel))
        .route("/:id/stream", get(stream))
        .with_state(ctx)
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
    let resp = ctx.client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /health returned {}", resp.status());
    }
    Ok(resp.json::<Value>().await?)
}

/// Start a fine-tune job on the worker (`POST /finetune`).
async fn worker_start(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune");
    let resp = ctx.client.post(&url).json(body).send().await?;
    let status = resp.status();
    let json = resp.json::<Value>().await?;
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
    let resp = ctx.client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune returned {}", resp.status());
    }
    Ok(resp.json::<Value>().await?)
}

/// One job snapshot from the worker (`GET /finetune/{id}`).
async fn worker_get(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx.client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune/{id} returned {}", resp.status());
    }
    Ok(resp.json::<Value>().await?)
}

/// Cancel a worker job (`DELETE /finetune/{id}`).
async fn worker_cancel(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx.client.delete(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth cancel returned {}", resp.status());
    }
    Ok(resp.json::<Value>().await?)
}

/// Merge a trained adapter into a GGUF on the worker (`POST /finetune/merge`).
async fn worker_merge(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune/merge");
    let resp = ctx.client.post(&url).json(body).send().await?;
    let status = resp.status();
    let json = resp.json::<Value>().await?;
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
        "succeeded" => spawn_event(
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
        ),
        "failed" => spawn_event(
            ctx,
            EVENT_JOB_FAILED,
            json!({
                "job_id": prior.id,
                "base_model": prior.base_model,
                "output_name": prior.output_name,
                "target": prior.target,
                "error": error,
            }),
        ),
        // `queued` / `running` / `cancelled` are not finishes.
        _ => {}
    }
}

// ── Handlers ────────────────────────────────────────────────────────────────

/// `GET /api/finetune/capability` — what this node can train, for the desktop's
/// gating UI. Combines the device probe (authoritative for the *local* gate) with
/// the worker's `/health` (authoritative for CUDA-capability + whether the
/// training deps are installed), when the worker is reachable.
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
    let (can_local, reason) = local_capability(&dev);
    let sidecar = worker_health(ctx).await.ok();
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

/// `POST /api/finetune/start` — start a fine-tune job. Gates local training on the
/// GPU, proxies the request to the worker, and records the job. Body is forwarded
/// verbatim to the worker plus an optional `target` (`local` | `remote`).
#[utoipa::path(
    post,
    path = "/api/finetune/start",
    tag = "Finetune",
    summary = "start a fine-tune job (local GPU or remote node)",
    request_body = StartJobBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn start(State(ctx): State<FinetuneCtx>, Json(body): Json<Value>) -> Response {
    match dispatch(&ctx, body).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err((code, err)) => (code, Json(err)).into_response(),
    }
}

/// Start a fine-tune job (local or remote), returning the worker/remote response
/// JSON on success or a `(status, error-json)` on failure.
pub async fn dispatch(ctx: &FinetuneCtx, body: Value) -> Result<Value, (StatusCode, Value)> {
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

    let target = body
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .to_string();

    if target == "remote" {
        return dispatch_remote(ctx, &body, base_model).await;
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
            let job_id = resp
                .get("job_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let job_state = resp
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("running")
                .to_string();
            let output_name = body
                .get("output_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let now = chrono::Utc::now().to_rfc3339();
            let job = FinetuneJob {
                id: job_id,
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
            if let Err(e) = ctx.store.record(&job).await {
                tracing::warn!("recording finetune job failed: {e:#}");
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
    body: &Value,
    base_model: String,
) -> Result<Value, (StatusCode, Value)> {
    let remote = body.get("remote");
    let url = remote
        .and_then(|r| r.get("url"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .trim_end_matches('/')
        .to_string();
    let token = remote
        .and_then(|r| r.get("token"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if url.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "remote target needs `remote.url`" }),
        ));
    }

    // Forward verbatim but force the remote to train locally (it is the GPU node)
    // and drop our remote envelope so it doesn't recurse.
    let mut fwd = body.clone();
    if let Some(obj) = fwd.as_object_mut() {
        obj.insert("target".into(), json!("local"));
        obj.remove("remote");
    }

    let endpoint = format!("{url}/api/finetune/start");
    let mut req = ctx.client.post(&endpoint).json(&fwd);
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }
    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            let json_body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
            if !status.is_success() {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    json!({
                        "error": format!("remote node returned {status}"),
                        "detail": json_body,
                    }),
                ));
            }
            let job_id = json_body
                .get("job_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let job_state = json_body
                .get("state")
                .and_then(Value::as_str)
                .unwrap_or("running")
                .to_string();
            let output_name = body
                .get("output_name")
                .and_then(Value::as_str)
                .map(str::to_string);
            let now = chrono::Utc::now().to_rfc3339();
            let job = FinetuneJob {
                id: job_id,
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
            if let Err(e) = ctx.store.record(&job).await {
                tracing::warn!("recording remote finetune job failed: {e:#}");
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

/// If `id` is a remote job, return its `(url, token)` for proxying. `None` for a
/// local job or an unknown id.
async fn remote_of(ctx: &FinetuneCtx, id: &str) -> Option<(String, Option<String>)> {
    match ctx.store.get(id).await {
        Ok(Some(job)) if job.target == "remote" => job.remote_url.map(|u| (u, job.remote_token)),
        _ => None,
    }
}

/// Mirror a worker snapshot's mutable fields back into the persisted record so the
/// store stays current (and terminal jobs survive a Core/worker restart).
///
/// This is also the ONLY place a finish is ever noticed: the worker owns the
/// training and nothing here polls it in the background, so a job's terminal state
/// becomes known — and its event fires — on the next `/list` or `/:id` read.
async fn persist_from_snapshot(ctx: &FinetuneCtx, id: &str, snap: &Value) {
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
        .sync_from_snapshot(id, job_state, output_ref, error, &now)
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
pub async fn list(State(ctx): State<FinetuneCtx>) -> impl IntoResponse {
    match list_value(&ctx).await {
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
    if let Ok(Value::Array(snaps)) = worker_list(ctx).await {
        for snap in &snaps {
            if let Some(id) = snap.get("id").and_then(Value::as_str) {
                persist_from_snapshot(ctx, id, snap).await;
            }
        }
    }
    ctx.store
        .list()
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
pub async fn get_job(State(ctx): State<FinetuneCtx>, Path(id): Path<String>) -> Response {
    match get_value(&ctx, &id).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared single-job snapshot. Prefers the worker's (or remote node's) live
/// snapshot, persisting it; falls back to the stored record.
pub async fn get_value(ctx: &FinetuneCtx, id: &str) -> Result<Value, (StatusCode, Value)> {
    if let Some((base, token)) = remote_of(ctx, id).await {
        // Remote job: proxy the snapshot from the remote node's Core.
        let mut req = ctx.client.get(format!("{base}/api/finetune/{id}"));
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(snap) = resp.json::<Value>().await {
                    persist_from_snapshot(ctx, id, &snap).await;
                    return Ok(snap);
                }
            }
        }
        // Remote unreachable — fall through to the stored record below.
    } else if let Ok(snap) = worker_get(ctx, id).await {
        persist_from_snapshot(ctx, id, &snap).await;
        return Ok(snap);
    }
    match ctx.store.get(id).await {
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
pub async fn cancel(State(ctx): State<FinetuneCtx>, Path(id): Path<String>) -> Response {
    match cancel_value(&ctx, &id).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared cooperative-cancel. Proxies to the worker (or remote node) and marks the
/// stored record cancelled.
pub async fn cancel_value(ctx: &FinetuneCtx, id: &str) -> Result<Value, (StatusCode, Value)> {
    if let Some((base, token)) = remote_of(ctx, id).await {
        let mut req = ctx.client.delete(format!("{base}/api/finetune/{id}"));
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        return match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                let body = resp
                    .json::<Value>()
                    .await
                    .unwrap_or_else(|_| json!({ "cancelling": true }));
                let now = chrono::Utc::now().to_rfc3339();
                let _ = ctx
                    .store
                    .update_state(id, "cancelled", None, None, &now)
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
    match worker_cancel(ctx, id).await {
        Ok(resp) => {
            let now = chrono::Utc::now().to_rfc3339();
            let _ = ctx
                .store
                .update_state(id, "cancelled", None, None, &now)
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
pub async fn list_adapters() -> impl IntoResponse {
    Json(json!({ "adapters": adapters::load_present() }))
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
pub async fn merge(State(ctx): State<FinetuneCtx>, Json(body): Json<Value>) -> Response {
    match merge_value(&ctx, body).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err((code, body)) => (code, Json(body)).into_response(),
    }
}

/// Shared adapter→GGUF merge. Registers the merged GGUF as an installed model on
/// success (idempotent, into the shared `${RYU_DIR}/installed-models.json`).
pub async fn merge_value(ctx: &FinetuneCtx, body: Value) -> Result<Value, (StatusCode, Value)> {
    if body.get("adapter_name").and_then(Value::as_str).is_none()
        && body.get("adapter_path").and_then(Value::as_str).is_none()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "need `adapter_name` or `adapter_path`" }),
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
pub async fn stream(State(ctx): State<FinetuneCtx>, Path(id): Path<String>) -> Response {
    stream_response(&ctx, &id).await
}

/// Shared SSE proxy for a job's progress stream. Streams the worker's (or remote
/// node's) `text/event-stream` frames through verbatim.
pub async fn stream_response(ctx: &FinetuneCtx, id: &str) -> Response {
    // Remote jobs stream from the remote node's Core; local jobs from the worker.
    let (url, token) = match remote_of(ctx, id).await {
        Some((base, token)) => (format!("{base}/api/finetune/{id}/stream"), token),
        None => (worker_stream_url(ctx, id), None),
    };
    let mut req = ctx.client.get(&url);
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
}
