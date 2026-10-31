//! Durable retry and tenant lifecycle regressions against the production SQLite store.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use ryu_finetune::{FinetuneJob, FinetuneStore, FinetuneTenantContext, StartIdempotencyClaim};
use serde_json::json;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Database(PathBuf);

impl Database {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ryu-finetune-lifecycle-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        )))
    }

    fn open(&self) -> FinetuneStore {
        FinetuneStore::open(self.0.join("finetune.db")).unwrap()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn job() -> FinetuneJob {
    FinetuneJob {
        id: "run-1".into(),
        tenant: Default::default(),
        base_model: "unsloth/gemma-3-4b".into(),
        method: "dpo".into(),
        output_name: Some("preference-adapter".into()),
        state: "running".into(),
        target: "local".into(),
        remote_url: None,
        remote_token: None,
        output_ref: None,
        artifact_id: None,
        error: None,
        created_at: "2026-10-05T00:00:00Z".into(),
        updated_at: "2026-10-05T00:00:00Z".into(),
    }
}

#[tokio::test]
async fn simultaneous_start_retries_have_one_claim_and_restart_replays_first_response() {
    let database = Database::new();
    let store = database.open();
    let (first, second) = tokio::join!(
        store.claim_start("request-1", "payload-1"),
        store.claim_start("request-1", "payload-1")
    );
    let claims = [first.unwrap(), second.unwrap()];
    assert_eq!(
        claims
            .iter()
            .filter(|claim| **claim == StartIdempotencyClaim::New)
            .count(),
        1
    );
    assert_eq!(
        claims
            .iter()
            .filter(|claim| **claim == StartIdempotencyClaim::InProgress)
            .count(),
        1
    );

    assert!(!store
        .complete_start(
            "request-1",
            "different-payload",
            &json!({"job_id": "wrong"})
        )
        .await
        .unwrap());
    assert!(!store
        .release_start("request-1", "different-payload")
        .await
        .unwrap());
    let response = json!({"job_id": "run-1", "state": "running"});
    assert!(store
        .complete_start("request-1", "payload-1", &response)
        .await
        .unwrap());
    assert!(!store
        .complete_start("request-1", "payload-1", &json!({"job_id": "replacement"}))
        .await
        .unwrap());
    assert!(!store.release_start("request-1", "payload-1").await.unwrap());
    drop(store);

    let reopened = database.open();
    assert_eq!(
        reopened
            .claim_start("request-1", "payload-1")
            .await
            .unwrap(),
        StartIdempotencyClaim::Replay(response)
    );
    assert_eq!(
        reopened
            .claim_start("request-1", "different-payload")
            .await
            .unwrap(),
        StartIdempotencyClaim::Conflict
    );
}

#[tokio::test]
async fn failed_start_can_be_retried_without_releasing_another_request() {
    let database = Database::new();
    let store = database.open();
    assert_eq!(
        store.claim_start("failed-start", "payload").await.unwrap(),
        StartIdempotencyClaim::New
    );
    assert!(!store
        .release_start("unknown-start", "payload")
        .await
        .unwrap());
    assert!(store
        .release_start("failed-start", "payload")
        .await
        .unwrap());
    assert!(!store
        .release_start("failed-start", "payload")
        .await
        .unwrap());
    assert_eq!(
        store
            .claim_start("failed-start", "corrected-payload")
            .await
            .unwrap(),
        StartIdempotencyClaim::New
    );
}

#[tokio::test]
async fn tenant_refusals_preserve_the_owner_job_and_terminal_result_after_restart() {
    let database = Database::new();
    let store = database.open();
    let owner = FinetuneTenantContext {
        owner_user_id: Some("owner".into()),
        org_id: Some("org-1".into()),
        node_id: store.node_id().into(),
    };
    store.record_if_absent_for(&owner, &job()).await.unwrap();
    for stranger in [
        FinetuneTenantContext {
            owner_user_id: Some("other".into()),
            ..owner.clone()
        },
        FinetuneTenantContext {
            org_id: Some("org-2".into()),
            ..owner.clone()
        },
        FinetuneTenantContext::local(store.node_id()),
    ] {
        assert!(store.list_for(&stranger).await.unwrap().is_empty());
        assert!(store.get_for(&stranger, "run-1").await.unwrap().is_none());
        assert!(!store
            .update_state_for(&stranger, "run-1", "cancelled", None, None, "later")
            .await
            .unwrap());
        assert!(store
            .sync_from_snapshot_for(
                &stranger,
                "run-1",
                "failed",
                None,
                Some("foreign error"),
                "later"
            )
            .await
            .unwrap()
            .is_none());
        assert!(!store.record_if_absent_for(&stranger, &job()).await.unwrap());
    }
    let wrong_node = FinetuneTenantContext {
        node_id: "different-node".into(),
        ..owner.clone()
    };
    assert!(store
        .update_state_for(&wrong_node, "run-1", "cancelled", None, None, "later")
        .await
        .is_err());

    let (success, failure) = tokio::join!(
        store.sync_from_snapshot_for(
            &owner,
            "run-1",
            "succeeded",
            Some("/models/preference"),
            None,
            "2026-10-05T01:00:00Z"
        ),
        store.sync_from_snapshot_for(
            &owner,
            "run-1",
            "failed",
            None,
            Some("worker failed"),
            "2026-10-05T01:00:00Z"
        )
    );
    assert_eq!(
        [success.unwrap(), failure.unwrap()]
            .iter()
            .filter(|prior| prior.is_some())
            .count(),
        1
    );
    let terminal = store.get_for(&owner, "run-1").await.unwrap().unwrap();
    assert!(matches!(terminal.state.as_str(), "succeeded" | "failed"));
    assert!(!store
        .update_state_for(&owner, "run-1", "running", None, None, "later")
        .await
        .unwrap());
    drop(store);

    let reopened = database.open();
    let persisted = reopened.get_for(&owner, "run-1").await.unwrap().unwrap();
    assert_eq!(persisted.state, terminal.state);
    assert_eq!(persisted.output_ref, terminal.output_ref);
    assert_eq!(persisted.error, terminal.error);
    assert_eq!(persisted.method, "dpo");
    assert_eq!(persisted.tenant.owner_user_id.as_deref(), Some("owner"));
}
