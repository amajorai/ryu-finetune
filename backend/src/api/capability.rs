//! The local GPU gate, app events and the capability handler.

use super::*;

// ── GPU gate ────────────────────────────────────────────────────────────────

/// Whether this node can train locally, plus a human reason when it cannot.
/// Heuristic: a discrete (non-unified) GPU detected by `nvidia-smi`. Unsloth
/// training requires an NVIDIA CUDA GPU; Apple unified memory and CPU-only boxes
/// cannot train (they fall back to a remote node — Unit 5).
pub(super) fn local_capability(dev: &DeviceInfo) -> (bool, String) {
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
pub(super) fn spawn_event(ctx: &FinetuneCtx, event: &'static str, payload: Value) {
    let events = ctx.events.clone();
    tokio::spawn(async move { events.emit(event, payload).await });
}

/// [`spawn_event`] plus an optional user-facing notification raised alongside the
/// fan-out (the Inbox shows a finished-job row with the Fine-tune icon).
pub(super) fn spawn_event_with_notify(
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
pub(super) fn announce_job_started(ctx: &FinetuneCtx, job: &FinetuneJob) {
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
pub(super) fn announce_job_finished(
    ctx: &FinetuneCtx,
    prior: &FinetuneJob,
    state: &str,
    adapter_name: Option<&str>,
    adapter_path: Option<&str>,
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
                // `None` for a remote job: its adapter stays on the remote node
                // and is never indexed (or mergeable) here.
                "adapter_name": adapter_name,
                "adapter_path": adapter_path,
            }),
            // A training job finishing is the classic long-job-done notification.
            Some(crate::fanout_messages::job_finished(
                prior.output_name.as_deref(),
                &prior.base_model,
            )),
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
            Some(crate::fanout_messages::job_failed(
                prior.output_name.as_deref(),
                error,
            )),
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
pub(super) fn effective_local_capability(
    (hardware_ready, hardware_reason): (bool, String),
    sidecar: Option<&Value>,
) -> (bool, String) {
    if !hardware_ready {
        return (false, hardware_reason);
    }

    let Some(sidecar) = sidecar else {
        return (false, LOCAL_WORKER_DOWN_MESSAGE.to_string());
    };
    if sidecar
        .get("can_finetune")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (true, String::new());
    }

    (false, worker_reason(sidecar))
}
