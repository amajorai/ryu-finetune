//! Its own test binary: the adapter catalog's data dir is a process-wide
//! `OnceLock`, so sharing a binary with another catalog test would share a file.

use axum::http::StatusCode;
use ryu_finetune::adapters::{self, InstalledAdapter};
use ryu_finetune::api::{merge_value_for, new_artifact_id, FinetuneCtx};
use ryu_finetune::{FinetuneJob, FinetuneStore, FinetuneTenantContext};
use serde_json::json;

/// The audit case end to end through the on-disk catalog: a second job (another
/// tenant) that reuses the first job's output NAME must not displace the first
/// job's record, and a catalog written by the old stem-keyed format still loads.
#[tokio::test]
async fn a_same_name_adapter_from_another_job_does_not_displace_the_first() {
    let dir = std::env::temp_dir().join(format!(
        "ryu-finetune-adapters-collide-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    ryu_finetune::init_data_dir(dir.clone());
    let legacy_dir = dir.join("legacy-stem");
    std::fs::create_dir_all(&legacy_dir).unwrap();
    std::fs::write(
        dir.join("installed-adapters.json"),
        serde_json::json!({ "adapters": { "legacy-stem": {
            "stem": "legacy-stem",
            "base_model": "unsloth/base",
            "job_id": "job-legacy",
            "path": legacy_dir.to_string_lossy(),
            "created_at": "2026-01-01T00:00:00Z"
        } } })
        .to_string(),
    )
    .unwrap();

    for (job, artifact) in [("job-alice", "ft-alice"), ("job-bob", "ft-bob")] {
        let path = dir.join(artifact);
        std::fs::create_dir_all(&path).unwrap();
        adapters::record(InstalledAdapter {
            stem: artifact.to_string(),
            name: Some("shared".to_string()),
            base_model: "unsloth/base".to_string(),
            job_id: job.to_string(),
            path: path.to_string_lossy().to_string(),
            created_at: "2026-07-17T00:00:00Z".to_string(),
        })
        .unwrap();
    }

    let mut present = adapters::load_present();
    present.sort_by(|a, b| a.job_id.cmp(&b.job_id));
    let jobs: Vec<&str> = present.iter().map(|a| a.job_id.as_str()).collect();
    assert_eq!(jobs, ["job-alice", "job-bob", "job-legacy"]);
    assert_ne!(present[0].path, present[1].path);
    assert_eq!(present[2].name, None);

    // Merge resolves only the caller's OWN adapters. Bob holds job-bob; Alice's
    // adapter is not mergeable for him by stem, by path, or by pairing his own
    // stem with her path. A refused merge never reaches the worker (an
    // unreachable URL here, which would surface as 502).
    let store = FinetuneStore::open(dir.join("finetune.db")).unwrap();
    let bob = FinetuneTenantContext {
        owner_user_id: Some("bob".to_owned()),
        org_id: Some("org-1".to_owned()),
        node_id: store.node_id().to_owned(),
    };
    store
        .record_for(
            &bob,
            &FinetuneJob {
                id: "job-bob".to_owned(),
                tenant: Default::default(),
                base_model: "unsloth/base".to_owned(),
                method: "sft".to_owned(),
                output_name: Some("shared".to_owned()),
                state: "succeeded".to_owned(),
                target: "local".to_owned(),
                remote_url: None,
                remote_token: None,
                output_ref: None,
                artifact_id: None,
                error: None,
                created_at: "2026-07-17T00:00:00Z".to_owned(),
                updated_at: "2026-07-17T00:00:00Z".to_owned(),
            },
        )
        .await
        .unwrap();
    let ctx = FinetuneCtx::new(store, reqwest::Client::new(), "http://127.0.0.1:9");
    let alice_path = dir.join("ft-alice").to_string_lossy().to_string();
    let refused = merge_value_for(&ctx, &bob, json!({ "adapter_name": "ft-alice" }))
        .await
        .unwrap_err();
    assert_eq!(refused.0, StatusCode::NOT_FOUND);
    let refused = merge_value_for(&ctx, &bob, json!({ "adapter_path": alice_path }))
        .await
        .unwrap_err();
    assert_eq!(refused.0, StatusCode::NOT_FOUND);
    let smuggled = merge_value_for(
        &ctx,
        &bob,
        json!({ "adapter_name": "ft-bob", "adapter_path": alice_path }),
    )
    .await
    .unwrap_err();
    assert_eq!(smuggled.0, StatusCode::BAD_REQUEST);
    let own = merge_value_for(&ctx, &bob, json!({ "adapter_name": "ft-bob" }))
        .await
        .unwrap_err();
    assert_eq!(
        own.0,
        StatusCode::BAD_GATEWAY,
        "his own adapter passes authorization and reaches the worker"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn artifact_ids_are_unique_and_match_the_worker_contract() {
    let first = new_artifact_id();
    let second = new_artifact_id();
    assert_ne!(first, second);
    for id in [first, second] {
        let hex = id.strip_prefix("ft-").expect("ft- prefix");
        assert_eq!(hex.len(), 32);
        assert!(hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
    }
}
