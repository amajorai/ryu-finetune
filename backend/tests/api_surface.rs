//! Network-touching coverage for the `/api/finetune/*` surface.
//!
//! The Python Unsloth worker and any remote Ryu Cloud GPU node are both reached
//! over HTTP; every test here fronts them with a throwaway **axum mock server**
//! bound to `127.0.0.1:0` (kernel-assigned port, loopback only — hermetic, no
//! real network). Unreachable-source branches point the client at
//! `http://127.0.0.1:1` (nothing listens → immediate connection refused). The
//! public `*_value` functions carry the logic; the thin axum handler wrappers are
//! exercised through the real `routes()` Router via `tower::ServiceExt::oneshot`.
//!
//! Isolation: each store gets its own temp `finetune.db` (`FinetuneStore::open`,
//! never `open_default`). The process-global `DATA_DIR` OnceLock is set once to a
//! per-binary temp dir; because adapter/installed-model writes accumulate there
//! across tests, every catalog assertion matches its OWN unique stem (`any(...)`),
//! never a total count.

use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::Path,
    http::{header, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use ryu_finetune::api::{
    cancel_value, capability_value, dispatch, get_value, list_value, merge_value, routes,
    stream_response, FinetuneCtx,
};
use ryu_finetune::{FinetuneJob, FinetuneStore};

// ── Test scaffolding ────────────────────────────────────────────────────────

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique(tag: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ryu-finetune-{tag}-{}-{nanos}-{seq}", std::process::id())
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(unique(tag));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn temp_store() -> FinetuneStore {
    FinetuneStore::open(temp_dir("db").join("finetune.db")).unwrap()
}

/// A URL with nothing listening — connection is refused immediately.
const DEAD_URL: &str = "http://127.0.0.1:1";

fn ctx_for(store: FinetuneStore, url: impl Into<String>) -> FinetuneCtx {
    FinetuneCtx::new(store, reqwest::Client::new(), url)
}

/// Bind a mock server on an ephemeral loopback port and serve `router` on a
/// background task. Returns its base URL (`http://127.0.0.1:<port>`). The task is
/// aborted when the test runtime shuts down.
async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

fn local_job(id: &str) -> FinetuneJob {
    FinetuneJob {
        id: id.to_owned(),
        base_model: "unsloth/llama-3-8b".to_owned(),
        output_name: None,
        state: "queued".to_owned(),
        target: "local".to_owned(),
        remote_url: None,
        remote_token: None,
        output_ref: None,
        error: None,
        created_at: "2024-01-01T00:00:00Z".to_owned(),
        updated_at: "2024-01-01T00:00:00Z".to_owned(),
    }
}

fn remote_job(id: &str, url: &str) -> FinetuneJob {
    FinetuneJob {
        target: "remote".to_owned(),
        remote_url: Some(url.to_owned()),
        remote_token: Some("node-token".to_owned()),
        ..local_job(id)
    }
}

// ── capability_value: worker /health reachable vs not ───────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capability_includes_sidecar_health_when_worker_reachable() {
    let worker = spawn(Router::new().route(
        "/health",
        get(|| async { Json(json!({ "cuda": true, "unsloth_installed": true })) }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);

    let v = capability_value(&ctx).await;
    // Device probe keys are always present (values are hardware-dependent).
    assert!(v.get("can_train_local").unwrap().is_boolean());
    assert!(v.get("os").is_some());
    assert!(v.get("reason").is_some());
    // The worker's /health payload is threaded through as `sidecar`.
    assert_eq!(v["sidecar"]["cuda"], json!(true));
    assert_eq!(v["sidecar"]["unsloth_installed"], json!(true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capability_sidecar_is_null_when_worker_unreachable() {
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let v = capability_value(&ctx).await;
    assert!(v["sidecar"].is_null(), "sidecar should be null: {v}");
    assert!(v.get("can_train_local").unwrap().is_boolean());
}

// Worker /health returns a non-2xx → treated as unreachable (sidecar null).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capability_sidecar_is_null_when_health_errors() {
    let worker = spawn(Router::new().route(
        "/health",
        get(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let v = capability_value(&ctx).await;
    assert!(v["sidecar"].is_null());
}

// ── dispatch → remote (not GPU-gated, so fully drivable) ────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatch_remote_forwards_and_records_the_job() {
    // The remote node echoes what it received so we can prove the envelope was
    // rewritten to force local training + strip the `remote` block.
    async fn start(Json(body): Json<Value>) -> Response {
        // Guard: the forwarded body must force `target=local` and drop `remote`.
        assert_eq!(body.get("target").and_then(Value::as_str), Some("local"));
        assert!(body.get("remote").is_none());
        Json(json!({ "job_id": "remote-1", "state": "running" })).into_response()
    }
    let node = spawn(Router::new().route("/api/finetune/start", post(start))).await;

    let store = temp_store();
    let ctx = ctx_for(store.clone(), DEAD_URL); // worker unused on the remote path
    let body = json!({
        "base_model_id": "unsloth/llama-3-8b",
        "output_name": "my-out",
        "target": "remote",
        "remote": { "url": format!("{node}/"), "token": "node-token" },
    });

    let resp = dispatch(&ctx, body).await.expect("remote dispatch ok");
    assert_eq!(resp["job_id"], json!("remote-1"));

    let job = store.get("remote-1").await.unwrap().expect("job recorded");
    assert_eq!(job.target, "remote");
    assert_eq!(job.state, "running");
    assert_eq!(job.output_name.as_deref(), Some("my-out"));
    // Trailing slash on the supplied url is trimmed before persisting.
    assert_eq!(job.remote_url.as_deref(), Some(node.as_str()));
    assert_eq!(job.remote_token.as_deref(), Some("node-token"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatch_remote_maps_node_error_to_bad_gateway() {
    let node = spawn(Router::new().route(
        "/api/finetune/start",
        post(|| async {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "no gpu" })),
            )
        }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let body = json!({
        "base_model_id": "m",
        "target": "remote",
        "remote": { "url": node },
    });
    let (code, err) = dispatch(&ctx, body).await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(err["error"]
        .as_str()
        .unwrap()
        .contains("remote node returned"));
    assert_eq!(err["detail"]["error"], json!("no gpu"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatch_remote_unreachable_node_is_bad_gateway() {
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let body = json!({
        "base_model_id": "m",
        "target": "remote",
        "remote": { "url": DEAD_URL },
    });
    let (code, err) = dispatch(&ctx, body).await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(err["error"].as_str().unwrap().contains("unreachable"));
}

// ── get_value ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_prefers_live_worker_snapshot() {
    let worker = spawn(Router::new().route(
        "/finetune/:id",
        get(|Path(id): Path<String>| async move {
            Json(json!({ "id": id, "state": "running", "output_dir": null }))
        }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let v = get_value(&ctx, "job-x").await.expect("ok");
    assert_eq!(v["state"], json!("running"));
}

// succeeded snapshot must persist state + index the produced adapter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_succeeded_persists_and_indexes_adapter() {
    ensure_data_dir();
    let out_dir = temp_dir("adapter-out");
    let out = out_dir.to_string_lossy().to_string();
    let stem = format!("stem-{}", COUNTER.fetch_add(1, Ordering::Relaxed));

    let store = temp_store();
    let mut j = local_job("done-1");
    j.output_name = Some(stem.clone());
    store.record(&j).await.unwrap();

    let out_for_worker = out.clone();
    let worker = spawn(Router::new().route(
        "/finetune/:id",
        get(move |Path(id): Path<String>| {
            let out = out_for_worker.clone();
            async move { Json(json!({ "id": id, "state": "succeeded", "output_dir": out })) }
        }),
    ))
    .await;
    let ctx = ctx_for(store.clone(), worker);

    let v = get_value(&ctx, "done-1").await.expect("ok");
    assert_eq!(v["state"], json!("succeeded"));

    // Persisted back into the store.
    let job = store.get("done-1").await.unwrap().unwrap();
    assert_eq!(job.state, "succeeded");
    assert_eq!(job.output_ref.as_deref(), Some(out.as_str()));

    // Indexed into the adapter catalog under the job's output_name stem.
    let present = ryu_finetune::adapters::load_present();
    assert!(
        present.iter().any(|a| a.stem == stem),
        "adapter {stem} not indexed: {present:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_falls_back_to_store_when_worker_down() {
    let store = temp_store();
    store.record(&local_job("stored-1")).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let v = get_value(&ctx, "stored-1").await.expect("stored record");
    assert_eq!(v["id"], json!("stored-1"));
    assert_eq!(v["state"], json!("queued"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_unknown_id_is_not_found() {
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let (code, err) = get_value(&ctx, "nope").await.unwrap_err();
    assert_eq!(code, StatusCode::NOT_FOUND);
    assert!(err["error"].as_str().unwrap().contains("nope"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_remote_job_proxies_the_node() {
    let node = spawn(Router::new().route(
        "/api/finetune/:id",
        get(|Path(id): Path<String>| async move { Json(json!({ "id": id, "state": "running" })) }),
    ))
    .await;
    let store = temp_store();
    store.record(&remote_job("rj-1", &node)).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let v = get_value(&ctx, "rj-1").await.expect("ok");
    assert_eq!(v["state"], json!("running"));
}

// Remote node unreachable → the proxy attempt falls through to the stored record.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_value_remote_unreachable_falls_back_to_store() {
    let store = temp_store();
    store.record(&remote_job("rj-2", DEAD_URL)).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let v = get_value(&ctx, "rj-2").await.expect("stored fallback");
    assert_eq!(v["id"], json!("rj-2"));
    // The persisted secret never surfaces in the response.
    assert!(v.get("remote_token").is_none());
}

// ── list_value ──────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_value_overlays_worker_snapshots_onto_store() {
    ensure_data_dir();
    let out_dir = temp_dir("list-out");
    let out = out_dir.to_string_lossy().to_string();

    let store = temp_store();
    store.record(&local_job("a")).await.unwrap();
    store.record(&local_job("b")).await.unwrap();

    let out_for_worker = out.clone();
    let worker = spawn(Router::new().route(
        "/finetune",
        get(move || {
            let out = out_for_worker.clone();
            async move {
                Json(json!([
                    { "id": "a", "state": "running" },
                    { "id": "b", "state": "succeeded", "output_dir": out },
                ]))
            }
        }),
    ))
    .await;
    let ctx = ctx_for(store, worker);

    let v = list_value(&ctx).await.expect("ok");
    let jobs = v["jobs"].as_array().unwrap();
    assert_eq!(jobs.len(), 2);
    let state_of = |id: &str| {
        jobs.iter()
            .find(|j| j["id"] == json!(id))
            .map(|j| j["state"].as_str().unwrap().to_owned())
            .unwrap()
    };
    assert_eq!(state_of("a"), "running");
    assert_eq!(state_of("b"), "succeeded");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_value_returns_store_when_worker_down() {
    let store = temp_store();
    store.record(&local_job("only")).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let v = list_value(&ctx).await.expect("ok");
    assert_eq!(v["jobs"].as_array().unwrap().len(), 1);
}

// ── cancel_value ────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_value_local_marks_cancelled() {
    let store = temp_store();
    store.record(&local_job("c1")).await.unwrap();
    let worker = spawn(Router::new().route(
        "/finetune/:id",
        delete(|| async { Json(json!({ "cancelling": true })) }),
    ))
    .await;
    let ctx = ctx_for(store.clone(), worker);
    let v = cancel_value(&ctx, "c1").await.expect("ok");
    assert_eq!(v["cancelling"], json!(true));
    assert_eq!(store.get("c1").await.unwrap().unwrap().state, "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_value_local_worker_error_is_bad_gateway() {
    let worker = spawn(Router::new().route(
        "/finetune/:id",
        delete(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let (code, _) = cancel_value(&ctx, "c2").await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_value_remote_marks_cancelled() {
    let node = spawn(Router::new().route(
        "/api/finetune/:id",
        delete(|| async { Json(json!({ "ok": true })) }),
    ))
    .await;
    let store = temp_store();
    store.record(&remote_job("rc-1", &node)).await.unwrap();
    let ctx = ctx_for(store.clone(), DEAD_URL);
    let v = cancel_value(&ctx, "rc-1").await.expect("ok");
    assert_eq!(v["ok"], json!(true));
    assert_eq!(store.get("rc-1").await.unwrap().unwrap().state, "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_value_remote_node_error_is_bad_gateway() {
    let node = spawn(Router::new().route(
        "/api/finetune/:id",
        delete(|| async { (StatusCode::BAD_REQUEST, "nope") }),
    ))
    .await;
    let store = temp_store();
    store.record(&remote_job("rc-2", &node)).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let (code, err) = cancel_value(&ctx, "rc-2").await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(err["error"]
        .as_str()
        .unwrap()
        .contains("remote node returned"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_value_remote_unreachable_is_bad_gateway() {
    let store = temp_store();
    store.record(&remote_job("rc-3", DEAD_URL)).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let (code, err) = cancel_value(&ctx, "rc-3").await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(err["error"].as_str().unwrap().contains("unreachable"));
}

// ── merge_value ───────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_value_requires_adapter_name_or_path() {
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let (code, err) = merge_value(&ctx, json!({})).await.unwrap_err();
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(err["error"].as_str().unwrap().contains("adapter_name"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_value_registers_merged_model_on_success() {
    install_test_catalog_host();
    let stem = format!("merged-{}", COUNTER.fetch_add(1, Ordering::Relaxed));
    let stem_for_worker = stem.clone();
    let worker = spawn(Router::new().route(
        "/finetune/merge",
        post(move || {
            let stem = stem_for_worker.clone();
            async move {
                Json(json!({
                    "stem": stem,
                    "gguf_path": format!("/tmp/{stem}.gguf"),
                    "base_model": "unsloth/llama-3-8b",
                    "size_bytes": 4096,
                }))
            }
        }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let v = merge_value(&ctx, json!({ "adapter_name": "adp" }))
        .await
        .expect("ok");
    assert_eq!(v["stem"], json!(stem));

    // `load_present` prunes records whose weights are gone, so materialize the
    // registered stem's GGUF at the catalog's on-disk convention
    // (`<ryu_dir>/models/<stem>.gguf`) before asserting it was recorded.
    let catalog_dir = catalog_host_dir();
    let models_dir = catalog_dir.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::write(models_dir.join(format!("{stem}.gguf")), b"gguf").unwrap();

    let models = ryu_model_catalog::installed::load_present();
    assert!(
        models
            .iter()
            .any(|m| m.stem == stem && m.finetune_base.as_deref() == Some("unsloth/llama-3-8b")),
        "merged model {stem} not registered with provenance: {models:?}"
    );
}

// A worker response missing `stem`/`gguf_path` still succeeds but registers nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_value_without_stem_skips_registration() {
    let worker = spawn(Router::new().route(
        "/finetune/merge",
        post(|| async { Json(json!({ "ok": true })) }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let v = merge_value(&ctx, json!({ "adapter_path": "/tmp/adp" }))
        .await
        .expect("ok");
    assert_eq!(v["ok"], json!(true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_value_worker_error_is_bad_gateway() {
    let worker = spawn(Router::new().route(
        "/finetune/merge",
        post(|| async {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": "merge blew up" })),
            )
        }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let (code, err) = merge_value(&ctx, json!({ "adapter_name": "adp" }))
        .await
        .unwrap_err();
    assert_eq!(code, StatusCode::BAD_GATEWAY);
    assert!(err["error"].as_str().unwrap().contains("merge blew up"));
}

// ── stream_response ─────────────────────────────────────────────────────────

fn sse_ok() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from("data: {\"step\":1}\n\n"))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_local_worker_success_is_event_stream() {
    let worker =
        spawn(Router::new().route("/finetune/:id/stream", get(|| async { sse_ok() }))).await;
    let ctx = ctx_for(temp_store(), worker);
    let resp = stream_response(&ctx, "s1").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_worker_non_success_is_bad_gateway() {
    let worker = spawn(Router::new().route(
        "/finetune/:id/stream",
        get(|| async { (StatusCode::NOT_FOUND, "gone") }),
    ))
    .await;
    let ctx = ctx_for(temp_store(), worker);
    let resp = stream_response(&ctx, "s2").await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_worker_unreachable_is_bad_gateway() {
    let ctx = ctx_for(temp_store(), DEAD_URL);
    let resp = stream_response(&ctx, "s3").await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_remote_job_success() {
    let node =
        spawn(Router::new().route("/api/finetune/:id/stream", get(|| async { sse_ok() }))).await;
    let store = temp_store();
    store.record(&remote_job("rs-1", &node)).await.unwrap();
    let ctx = ctx_for(store, DEAD_URL);
    let resp = stream_response(&ctx, "rs-1").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── Thin handler wrappers via the real Router (tower::oneshot) ───────────────────

async fn oneshot(
    router: Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut b = Request::builder().method(method).uri(uri);
    let req = match body {
        Some(v) => {
            b = b.header(header::CONTENT_TYPE, "application/json");
            b.body(Body::from(v.to_string())).unwrap()
        }
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = router.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, value)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn router_wrappers_route_and_respond() {
    // Worker unreachable so every handler takes its store/fallback branch.
    let ctx = ctx_for(temp_store(), DEAD_URL);

    // GET /capability → 200 with the probe shape.
    let (code, v) = oneshot(routes(ctx.clone()), "GET", "/capability", None).await;
    assert_eq!(code, StatusCode::OK);
    assert!(v.get("can_train_local").is_some());

    // GET /list → 200 { jobs: [] }.
    let (code, v) = oneshot(routes(ctx.clone()), "GET", "/list", None).await;
    assert_eq!(code, StatusCode::OK);
    assert!(v["jobs"].is_array());

    // GET /adapters → 200 { adapters: [...] }.
    let (code, v) = oneshot(routes(ctx.clone()), "GET", "/adapters", None).await;
    assert_eq!(code, StatusCode::OK);
    assert!(v["adapters"].is_array());

    // POST /start with no base_model_id → 400 (validation wrapper).
    let (code, v) = oneshot(routes(ctx.clone()), "POST", "/start", Some(json!({}))).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("base_model_id"));

    // GET /:id unknown + worker down → 404.
    let (code, _) = oneshot(routes(ctx.clone()), "GET", "/ghost", None).await;
    assert_eq!(code, StatusCode::NOT_FOUND);

    // DELETE /:id worker down → 502.
    let (code, _) = oneshot(routes(ctx.clone()), "DELETE", "/ghost", None).await;
    assert_eq!(code, StatusCode::BAD_GATEWAY);

    // POST /merge with no adapter → 400.
    let (code, _) = oneshot(routes(ctx.clone()), "POST", "/merge", Some(json!({}))).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);

    // GET /:id/stream worker down → 502.
    let (code, _) = oneshot(routes(ctx), "GET", "/zzz/stream", None).await;
    assert_eq!(code, StatusCode::BAD_GATEWAY);
}

// ── Shared data-dir / catalog-host setup (process-global OnceLocks) ─────────────

fn ensure_data_dir() {
    static DIR: OnceLock<()> = OnceLock::new();
    DIR.get_or_init(|| {
        ryu_finetune::init_data_dir(temp_dir("data"));
    });
}

static CATALOG_DIR: OnceLock<PathBuf> = OnceLock::new();

fn catalog_host_dir() -> PathBuf {
    CATALOG_DIR.get().expect("catalog host installed").clone()
}

/// Install a temp-dir-backed `ModelCatalogHost` so `installed::record` (called by
/// `merge_value` on success) writes to a throwaway `installed-models.json` instead
/// of panicking on a missing host or touching a real `~/.ryu`.
fn install_test_catalog_host() {
    let dir = CATALOG_DIR.get_or_init(|| temp_dir("catalog")).clone();
    static HOST: OnceLock<()> = OnceLock::new();
    HOST.get_or_init(|| {
        struct TestHost {
            dir: PathBuf,
        }
        #[async_trait::async_trait]
        impl ryu_model_catalog::ModelCatalogHost for TestHost {
            fn ryu_dir(&self) -> PathBuf {
                self.dir.clone()
            }
            fn authorize_hf(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
                req
            }
            fn supported_on_node(&self, _engine: &str) -> bool {
                false
            }
            fn default_model_repos(&self) -> ryu_model_catalog::DefaultModelRepos {
                Vec::new()
            }
            async fn active_model_pref(&self) -> Option<String> {
                None
            }
        }
        ryu_model_catalog::set_global_host(std::sync::Arc::new(TestHost { dir }));
    });
}
