//! Listing, reading and cancelling jobs, and persisting worker snapshots.

use super::*;

/// Mirror a worker snapshot's mutable fields back into the persisted record so the
/// store stays current (and terminal jobs survive a Core/worker restart).
///
/// This is also the ONLY place a finish is ever noticed: the worker owns the
/// training and nothing here polls it in the background, so a job's terminal state
/// becomes known — and its event fires — on the next `/list` or `/:id` read.
pub(super) async fn persist_from_snapshot_for(
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

    // On success, index the produced adapter (Unit 3). Idempotent on the job, and
    // run on EVERY succeeded poll rather than only the transition, so a catalog
    // entry lost to a failed write or an out-of-band delete is restored by the
    // next read.
    //
    // Only a LOCAL job is indexed, and only at the directory this process
    // allocated for it at dispatch (`<adapters root>/<artifact_id>`). The
    // snapshot's `output_dir` is never trusted as the catalog path: a remote
    // node's snapshot is caller-chosen (any public host can answer as the
    // "remote"), and recording its path would let that caller list and merge
    // another job's adapter under its own job id.
    let mut adapter_name = None;
    // What the success event reports as the adapter's location: the indexed
    // directory for a local job, the remote node's own path for a remote one.
    let mut adapter_path = if prior.target == "local" {
        None
    } else {
        output_ref.map(str::to_owned)
    };
    if job_state == "succeeded" {
        if let Some(path) = owned_adapter_dir(ctx, &prior, output_ref) {
            adapter_path = Some(path.to_string_lossy().to_string());
            let stem = prior.artifact_id.clone().unwrap_or_default();
            if let Err(e) = adapters::record(InstalledAdapter {
                stem: stem.clone(),
                name: prior.output_name.clone(),
                base_model: prior.base_model.clone(),
                job_id: id.to_string(),
                path: path.to_string_lossy().to_string(),
                created_at: now.clone(),
            }) {
                tracing::warn!("indexing adapter for job {id} failed: {e:#}");
            }
            adapter_name = Some(stem);
        } else if output_ref.is_some() && prior.target == "local" {
            tracing::warn!(
                "job {id} reported an adapter outside its allocated directory; not indexing it"
            );
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
            adapter_path.as_deref(),
            error,
        );
    }
}

/// The adapter directory a succeeded snapshot may be indexed under: this local
/// job's own `<adapters root>/<artifact_id>`, and only when the snapshot reports
/// exactly that directory. Remote jobs, jobs without an allocated id, and any
/// other reported path yield `None`.
pub(super) fn owned_adapter_dir(
    ctx: &FinetuneCtx,
    job: &FinetuneJob,
    output_ref: Option<&str>,
) -> Option<std::path::PathBuf> {
    if job.target != "local" {
        return None;
    }
    let artifact_id = job.artifact_id.as_deref()?;
    let expected = ctx.adapters_root.join(artifact_id);
    let reported = std::path::Path::new(output_ref?);
    let same = match (reported.canonicalize(), expected.canonicalize()) {
        (Ok(reported), Ok(allocated)) => reported == allocated,
        _ => false,
    };
    same.then_some(expected)
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
        let mut req = client
            .get(format!("{base}/api/finetune/{id}"))
            .timeout(CONTROL_REQUEST_TIMEOUT);
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
            let mut req = client
                .delete(format!("{base}/api/finetune/{id}"))
                .timeout(CONTROL_REQUEST_TIMEOUT);
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
        Err(e) => Err(worker_failure(&e)),
    }
}
