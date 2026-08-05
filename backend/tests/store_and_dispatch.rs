//! Integration tests for the finetune durable store and the pre-network branches
//! of the dispatch surface. Public surface only, so no production-source edits.
//!
//! Isolation: every store gets its own temp `finetune.db` via an explicit path
//! (`FinetuneStore::open`), NEVER `open_default()` / `data_dir()` — so the
//! `DATA_DIR` OnceLock and the real `~/.ryu` are never touched.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use ryu_finetune::api::{dispatch, FinetuneCtx, DEFAULT_UNSLOTH_URL};
use ryu_finetune::{FinetuneJob, FinetuneStore};
use serde_json::json;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_db_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut dir = std::env::temp_dir();
    dir.push(format!("ryu-finetune-test-{pid}-{nanos}-{seq}"));
    dir.push("finetune.db");
    dir
}

fn job(id: &str, created_at: &str) -> FinetuneJob {
    FinetuneJob {
        id: id.to_owned(),
        base_model: "unsloth/llama-3-8b".to_owned(),
        output_name: Some("my-adapter".to_owned()),
        state: "queued".to_owned(),
        target: "local".to_owned(),
        remote_url: None,
        remote_token: None,
        output_ref: None,
        error: None,
        created_at: created_at.to_owned(),
        updated_at: created_at.to_owned(),
    }
}

// ── Store CRUD ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn record_and_get_roundtrip() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    store
        .record(&job("j1", "2024-01-01T00:00:00Z"))
        .await
        .unwrap();
    let got = store.get("j1").await.unwrap().expect("job present");
    assert_eq!(got.id, "j1");
    assert_eq!(got.base_model, "unsloth/llama-3-8b");
    assert_eq!(got.state, "queued");
    assert_eq!(got.output_name.as_deref(), Some("my-adapter"));
    assert!(store.get("missing").await.unwrap().is_none());
}

#[tokio::test]
async fn list_orders_by_created_at_desc() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    store
        .record(&job("old", "2020-01-01T00:00:00Z"))
        .await
        .unwrap();
    store
        .record(&job("mid", "2021-01-01T00:00:00Z"))
        .await
        .unwrap();
    store
        .record(&job("new", "2022-01-01T00:00:00Z"))
        .await
        .unwrap();
    let ids: Vec<String> = store
        .list()
        .await
        .unwrap()
        .into_iter()
        .map(|j| j.id)
        .collect();
    assert_eq!(ids, vec!["new", "mid", "old"]);
}

#[tokio::test]
async fn record_is_idempotent_on_id_and_replaces_fields() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    store
        .record(&job("j", "2024-01-01T00:00:00Z"))
        .await
        .unwrap();
    let mut updated = job("j", "2024-01-01T00:00:00Z");
    updated.state = "running".to_owned();
    updated.base_model = "changed".to_owned();
    store.record(&updated).await.unwrap();

    let all = store.list().await.unwrap();
    assert_eq!(all.len(), 1, "INSERT OR REPLACE keeps a single row per id");
    assert_eq!(all[0].state, "running");
    assert_eq!(all[0].base_model, "changed");
}

#[tokio::test]
async fn update_state_returns_false_for_unknown_id() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let changed = store
        .update_state(
            "ghost",
            "failed",
            None,
            Some("boom"),
            "2024-01-02T00:00:00Z",
        )
        .await
        .unwrap();
    assert!(!changed);
}

#[tokio::test]
async fn update_state_coalesce_keeps_old_values_when_none() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let mut j = job("j", "2024-01-01T00:00:00Z");
    j.output_ref = Some("/models/a".to_owned());
    j.error = Some("earlier-error".to_owned());
    store.record(&j).await.unwrap();

    // Passing None for output_ref/error must COALESCE to the existing values.
    let changed = store
        .update_state("j", "running", None, None, "2024-01-02T00:00:00Z")
        .await
        .unwrap();
    assert!(changed);
    let got = store.get("j").await.unwrap().unwrap();
    assert_eq!(got.state, "running");
    assert_eq!(got.output_ref.as_deref(), Some("/models/a"));
    assert_eq!(got.error.as_deref(), Some("earlier-error"));
    assert_eq!(got.updated_at, "2024-01-02T00:00:00Z");
}

#[tokio::test]
async fn update_state_overwrites_when_some() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    store
        .record(&job("j", "2024-01-01T00:00:00Z"))
        .await
        .unwrap();
    store
        .update_state(
            "j",
            "succeeded",
            Some("/models/final"),
            None,
            "2024-01-03T00:00:00Z",
        )
        .await
        .unwrap();
    let got = store.get("j").await.unwrap().unwrap();
    assert_eq!(got.state, "succeeded");
    assert_eq!(got.output_ref.as_deref(), Some("/models/final"));
}

// ── remote_token is a persisted secret that must NOT serialize out ──────────────

#[tokio::test]
async fn remote_token_is_persisted_but_never_serialized() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let mut j = job("remote-job", "2024-01-01T00:00:00Z");
    j.target = "remote".to_owned();
    j.remote_url = Some("https://node.example".to_owned());
    j.remote_token = Some("super-secret-bearer".to_owned());
    store.record(&j).await.unwrap();

    // Persisted: it round-trips through the DB so Core can proxy on the job's behalf.
    let got = store.get("remote-job").await.unwrap().unwrap();
    assert_eq!(got.remote_token.as_deref(), Some("super-secret-bearer"));
    assert_eq!(got.remote_url.as_deref(), Some("https://node.example"));

    // But it is `#[serde(skip_serializing)]`: an API response must never leak it.
    let value = serde_json::to_value(&got).unwrap();
    assert!(
        value.get("remote_token").is_none(),
        "remote_token leaked into serialized job: {value}"
    );
    // Non-secret fields still serialize.
    assert_eq!(
        value.get("remote_url").and_then(|v| v.as_str()),
        Some("https://node.example")
    );
    assert_eq!(
        value.get("base_model").and_then(|v| v.as_str()),
        Some("unsloth/llama-3-8b")
    );
}

#[test]
fn finetune_job_deserializes_without_remote_token() {
    // The field has `default`, so a payload lacking it (the normal case, since it
    // is never serialized) deserializes to None rather than erroring.
    let j: FinetuneJob = serde_json::from_str(
        r#"{
            "id":"x","base_model":"m","output_name":null,"state":"queued",
            "target":"local","output_ref":null,"error":null,
            "created_at":"t","updated_at":"t"
        }"#,
    )
    .unwrap();
    assert!(j.remote_token.is_none());
    assert!(j.remote_url.is_none());
}

// ── FinetuneCtx::new URL normalization (pure) ───────────────────────────────────

#[tokio::test]
async fn ctx_new_normalizes_the_worker_url() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let client = reqwest::Client::new();

    let empty = FinetuneCtx::new(store.clone(), client.clone(), "   ");
    assert_eq!(empty.unsloth_url, DEFAULT_UNSLOTH_URL);

    let trimmed = FinetuneCtx::new(store.clone(), client.clone(), "  http://host:9000/  ");
    assert_eq!(trimmed.unsloth_url, "http://host:9000");

    let no_trailing = FinetuneCtx::new(store, client, "http://host:9000");
    assert_eq!(no_trailing.unsloth_url, "http://host:9000");
}

// ── dispatch: pre-network validation branches only ──────────────────────────────

#[tokio::test]
async fn dispatch_rejects_missing_base_model_id() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let ctx = FinetuneCtx::new(store, reqwest::Client::new(), DEFAULT_UNSLOTH_URL);
    let err = dispatch(&ctx, json!({})).await.unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err
        .1
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap()
        .contains("base_model_id"));

    // Whitespace-only is also empty after trim.
    let err = dispatch(&ctx, json!({ "base_model_id": "   " }))
        .await
        .unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn dispatch_remote_without_url_is_rejected_before_network() {
    let store = FinetuneStore::open(temp_db_path()).unwrap();
    let ctx = FinetuneCtx::new(store, reqwest::Client::new(), DEFAULT_UNSLOTH_URL);
    let body = json!({ "base_model_id": "m", "target": "remote" });
    let err = dispatch(&ctx, body).await.unwrap_err();
    assert_eq!(err.0, StatusCode::BAD_REQUEST);
    assert!(err
        .1
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap()
        .contains("remote.url"));
}
