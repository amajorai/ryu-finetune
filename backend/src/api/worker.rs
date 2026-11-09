//! HTTP helpers for the Python Unsloth worker.

use super::*;

// ── Worker (Python Unsloth) HTTP proxy helpers ──────────────────────────────
// These replace Core's `sidecar::providers::unsloth::*` — the surface now targets
// `ctx.unsloth_url` directly instead of Core's hardcoded provider base URL.

/// Fetch the worker's hardware probe (`GET /health`). Used by `/api/finetune/capability`.
pub(super) async fn worker_health(ctx: &FinetuneCtx) -> anyhow::Result<Value> {
    let url = ctx.worker("/health");
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /health returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// Start a fine-tune job on the worker (`POST /finetune`).
pub(super) async fn worker_start(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune");
    let resp = ctx
        .worker_request(ctx.worker_client.post(&url))
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    let json = response_json_limited(resp).await?;
    if !status.is_success() {
        return Err(WorkerRefusal::from_body(status, &json, "/finetune").into());
    }
    Ok(json)
}

/// All in-process job snapshots from the worker (`GET /finetune`).
pub(super) async fn worker_list(ctx: &FinetuneCtx) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune");
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// One job snapshot from the worker (`GET /finetune/{id}`).
pub(super) async fn worker_get(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx
        .worker_request(ctx.worker_client.get(&url))
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("unsloth /finetune/{id} returned {}", resp.status());
    }
    Ok(response_json_limited(resp).await?)
}

/// Cancel a worker job (`DELETE /finetune/{id}`).
pub(super) async fn worker_cancel(ctx: &FinetuneCtx, id: &str) -> anyhow::Result<Value> {
    let url = ctx.worker(&format!("/finetune/{id}"));
    let resp = ctx
        .worker_request(ctx.worker_client.delete(&url))
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let json = response_json_limited(resp).await.unwrap_or(Value::Null);
        return Err(WorkerRefusal::from_body(status, &json, "cancel").into());
    }
    Ok(response_json_limited(resp).await?)
}

/// Merge a trained adapter into a GGUF on the worker (`POST /finetune/merge`).
pub(super) async fn worker_merge(ctx: &FinetuneCtx, body: &Value) -> anyhow::Result<Value> {
    let url = ctx.worker("/finetune/merge");
    let resp = ctx
        .worker_request(ctx.worker_client.post(&url))
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    let json = response_json_limited(resp).await?;
    if !status.is_success() {
        return Err(WorkerRefusal::from_body(status, &json, "/finetune/merge").into());
    }
    Ok(json)
}

/// URL of the worker's SSE progress stream for a job.
pub(super) fn worker_stream_url(ctx: &FinetuneCtx, id: &str) -> String {
    ctx.worker(&format!("/finetune/{id}/stream"))
}

/// Shown when the worker could not be reached or answered without a sentence of
/// its own (the transport detail goes to the log).
pub(super) const WORKER_UNAVAILABLE_MESSAGE: &str = "The fine-tuning tool could not be reached.";
/// Shown when the worker refused a request without saying why.
pub(super) const WORKER_REFUSAL_MESSAGE: &str =
    "The fine-tuning tool could not complete the request.";
const INSTALL_HINT: &str =
    "Install the Unsloth fine-tuning tool from the Store, or run `bun run dev:unsloth`.";

/// The worker answered a request with an error status. `Display` is the worker's
/// own `error` sentence, unchanged, so the Companion's localizer can match it;
/// the status and route stay here for tracing.
#[derive(Debug)]
pub(super) struct WorkerRefusal {
    pub(super) status: reqwest::StatusCode,
    pub(super) route: &'static str,
    pub(super) message: Option<String>,
}

impl WorkerRefusal {
    pub(super) fn from_body(
        status: reqwest::StatusCode,
        body: &Value,
        route: &'static str,
    ) -> Self {
        let message = body
            .get("error")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .map(str::to_owned);
        Self {
            status,
            route,
            message,
        }
    }
}

impl std::fmt::Display for WorkerRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message.as_deref().unwrap_or(WORKER_REFUSAL_MESSAGE))
    }
}

impl std::error::Error for WorkerRefusal {}

/// The `(status, body)` a handler returns when a worker call failed: the worker's
/// own sentence when it refused, else a fixed sentence plus the install hint.
pub(super) fn worker_failure(error: &anyhow::Error) -> (StatusCode, Value) {
    if let Some(refusal) = error.downcast_ref::<WorkerRefusal>() {
        tracing::warn!(
            status = %refusal.status,
            route = refusal.route,
            "unsloth worker refused: {refusal}"
        );
        return (
            StatusCode::BAD_GATEWAY,
            json!({ "error": refusal.to_string() }),
        );
    }
    tracing::warn!("unsloth worker unavailable: {error:#}");
    (
        StatusCode::BAD_GATEWAY,
        json!({ "error": WORKER_UNAVAILABLE_MESSAGE, "hint": INSTALL_HINT }),
    )
}
