//! Installed adapters and the adapter-to-GGUF merge.

use super::*;

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
/// success (idempotent, into the model catalog's shared `installed-models.json`).
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

    let selector = |key: &str| {
        body.get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
    };
    let adapter_name = selector("adapter_name");
    let adapter_path = selector("adapter_path");
    // Resolve the adapter ONLY among catalog records produced by this tenant's own
    // jobs. A name and a path that name different adapters are refused rather than
    // letting the worker pick one, and the worker is then sent the owned record's
    // path alone, so it merges exactly the adapter that was authorized.
    let owned_jobs: std::collections::HashSet<String> = ctx
        .store
        .list_for(tenant)
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({ "error": format!("{error:#}") }),
            )
        })?
        .into_iter()
        .map(|job| job.id)
        .collect();
    let adapter = match adapters::resolve_owned(
        adapters::load_present(),
        &owned_jobs,
        adapter_name,
        adapter_path,
    ) {
        Ok(adapter) => adapter,
        Err(adapters::ResolveError::Ambiguous) => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "adapter_name and adapter_path name different adapters" }),
            ));
        }
    };
    let tenant_scoped = tenant.owner_user_id.is_some() || tenant.org_id.is_some();
    let requested_name = match body.get("output_name") {
        None | Some(Value::Null) => None,
        Some(Value::String(name)) if valid_output_name(name.trim()) => Some(name.trim().to_owned()),
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "output_name must use ASCII letters, digits, '.', '_' or '-' and be at most 128 characters" }),
            ));
        }
    };
    let mut body = body;
    // The merged GGUF lands in the node-wide chat-model folder, so its stem is
    // server-generated: the requested name (or the adapter's) plus a random
    // suffix. One caller can never pick another tenant's merged model, or any
    // installed chat GGUF, as its output; the worker additionally refuses to
    // overwrite an existing file.
    let merged_stem = merged_model_stem(
        requested_name
            .as_deref()
            .or_else(|| adapter.as_ref().and_then(|a| a.name.as_deref()))
            .unwrap_or("merged"),
    );
    if let Some(object) = body.as_object_mut() {
        object.insert("output_name".to_owned(), Value::String(merged_stem));
    }
    match adapter {
        Some(adapter) => {
            if let Some(object) = body.as_object_mut() {
                object.remove("adapter_name");
                object.insert("adapter_path".to_owned(), Value::String(adapter.path));
            }
        }
        // A tenant-scoped caller may merge only its own adapters. An unbound
        // personal node keeps its single principal's access to any adapter under
        // the worker's output root (the worker still confines the path).
        None if tenant_scoped => {
            return Err((
                StatusCode::NOT_FOUND,
                json!({ "error": "adapter not found for this tenant" }),
            ));
        }
        None => {}
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
        Err(e) => Err(worker_failure(&e)),
    }
}
