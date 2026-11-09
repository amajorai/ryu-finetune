//! Dispatching a job to a remote Ryu Cloud GPU node.

use super::*;

/// Dispatch a job to a remote Ryu Cloud GPU node (Unit 5). The desktop supplies
/// the target node's connection as `body.remote = { url, token }`; we forward the
/// job to that node's Core (forcing it to train *locally* there), then record it
/// with the remote coordinates so `get`/`stream`/`cancel` proxy back to it.
pub(super) async fn dispatch_remote(
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
    let mut req = remote_client
        .post(&endpoint)
        .timeout(CONTROL_REQUEST_TIMEOUT)
        .json(&fwd);
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
                method: body
                    .pointer("/training/method")
                    .and_then(Value::as_str)
                    .unwrap_or("sft")
                    .to_owned(),
                output_name,
                state: job_state,
                target: "remote".to_string(),
                remote_url: Some(url),
                remote_token: token,
                output_ref: None,
                artifact_id: None,
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

pub(super) async fn remote_of_for(
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

pub(super) fn response_job_id(response: &Value) -> Result<String, String> {
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
