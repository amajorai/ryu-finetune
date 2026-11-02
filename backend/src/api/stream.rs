//! The SSE progress-stream proxy.

use super::*;

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
