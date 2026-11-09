//! Starting a fine-tune job: idempotency and the local dispatch.

use super::*;

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

pub(super) fn normalize_idempotency_key(raw: &str) -> Result<String, String> {
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

pub(super) fn request_idempotency_key(headers: &HeaderMap) -> Result<Option<String>, String> {
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

pub(super) fn request_fingerprint(body: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(body)
        .map_err(|error| format!("could not fingerprint request: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(super) async fn dispatch_unkeyed(
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

    let training_method = body
        .pointer("/training/method")
        .and_then(Value::as_str)
        .unwrap_or("sft")
        .trim()
        .to_owned();
    if !matches!(training_method.as_str(), "sft" | "dpo") {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "training.method must be `sft` or `dpo`" }),
        ));
    }
    let dataset_format = body
        .pointer("/dataset/format")
        .and_then(Value::as_str)
        .unwrap_or("chat")
        .trim()
        .to_owned();
    let dataset_method_matches = (training_method == "dpo" && dataset_format == "preference")
        || (training_method == "sft" && dataset_format != "preference");
    if !dataset_method_matches {
        return Err((
            StatusCode::BAD_REQUEST,
            json!({ "error": "DPO requires dataset.format `preference`; SFT uses `chat`, `alpaca`, or `text`" }),
        ));
    }
    if let Some(beta_value) = body
        .pointer("/training/beta")
        .filter(|value| !value.is_null())
    {
        let valid_beta = beta_value
            .as_f64()
            .is_some_and(|beta| beta.is_finite() && beta > 0.0 && beta <= 1.0);
        if training_method == "dpo" && !valid_beta {
            return Err((
                StatusCode::BAD_REQUEST,
                json!({ "error": "training.beta must be greater than 0 and at most 1" }),
            ));
        }
    }

    // Forward the same canonical values we validated. Otherwise a harmless
    // surrounding space would pass Core's trimmed check but fail later in the
    // Python worker, while the durable record describes a different request.
    let mut body = body;
    if let Some(object) = body.as_object_mut() {
        // The adapter directory id is allocated by THIS process per local job
        // (below), never accepted from the caller.
        object.remove("artifact_id");
        object.insert(
            "base_model_id".to_owned(),
            Value::String(base_model.clone()),
        );
        if let Some(Value::String(output_name)) = object.get_mut("output_name") {
            *output_name = output_name.trim().to_owned();
        }
        if let Some(dataset) = object.get_mut("dataset").and_then(Value::as_object_mut) {
            dataset.insert("format".to_owned(), Value::String(dataset_format));
        }
        if let Some(training) = object.get_mut("training").and_then(Value::as_object_mut) {
            training.insert("method".to_owned(), Value::String(training_method.clone()));
        }
    }

    let target = body
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("local")
        .trim()
        .to_string();

    match target.as_str() {
        "remote" => return dispatch_remote(ctx, tenant, &body, base_model, idempotency_key).await,
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

    // A server-generated, per-job artifact id names the adapter directory, so two
    // tenants (or two jobs) that pick the same human-readable `output_name` can
    // never write into, replace, or displace each other's adapter. `output_name`
    // stays display metadata only.
    let artifact_id = new_artifact_id();
    let mut body = body;
    if let Some(object) = body.as_object_mut() {
        object.insert("artifact_id".to_owned(), Value::String(artifact_id.clone()));
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
                method: training_method.clone(),
                output_name,
                state: job_state,
                target,
                remote_url: None,
                remote_token: None,
                output_ref: None,
                artifact_id: Some(artifact_id),
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
        Err(e) => Err(worker_failure(&e)),
    }
}
