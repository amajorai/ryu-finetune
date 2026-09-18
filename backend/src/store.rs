//! Persisted fine-tune job store (`~/.ryu/finetune.db`).
//!
//! Core's durable record of every fine-tune job it has started. The job itself
//! runs *in the sidecar* (a separate process), so live progress is streamed from
//! there; this store is the system-of-record for the job list and survives a Core
//! restart (the sidecar's in-process registry does not). Mirrors the rusqlite
//! pattern in the `ryu-monitors` store.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{params, types::Type, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

struct FinetuneCryptoHost {
    ryu_dir: PathBuf,
}

impl ryu_crypto::CryptoHost for FinetuneCryptoHost {
    fn keyring_account_suffix(&self) -> String {
        let profile = std::env::var("RYU_PROFILE")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "release".to_owned());
        if profile == "release" {
            String::new()
        } else {
            format!("-{profile}")
        }
    }

    fn ryu_dir(&self) -> PathBuf {
        self.ryu_dir.clone()
    }
}

fn install_crypto_host(dir: &Path) {
    // Core installs its own host before opening in-process stores. Standalone
    // sidecars install this narrow equivalent; the crypto crate's OnceLock keeps
    // the first, process-wide custody decision authoritative.
    ryu_crypto::set_global_host(Arc::new(FinetuneCryptoHost {
        ryu_dir: dir.to_path_buf(),
    }));
}

fn seal_remote_token(token: Option<&str>) -> Result<Option<String>> {
    token
        .filter(|value| !value.is_empty())
        .map(|value| {
            ryu_crypto::global_cipher()?
                .seal(value)
                .context("sealing remote fine-tune token")
        })
        .transpose()
}

fn open_remote_token(stored: Option<String>) -> rusqlite::Result<Option<String>> {
    stored
        .map(|value| {
            ryu_crypto::global_cipher()
                .and_then(|cipher| cipher.open(&value))
                .map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        6,
                        Type::Text,
                        Box::new(std::io::Error::other(error.to_string())),
                    )
                })
        })
        .transpose()
}

const MAX_IDEMPOTENCY_KEY_BYTES: usize = 128;
const MAX_IDEMPOTENCY_FINGERPRINT_BYTES: usize = 128;
const MAX_IDEMPOTENCY_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_IDEMPOTENCY_RECORDS: i64 = 4096;
const MAX_IDEMPOTENCY_PENDING: i64 = 32;

/// Result of claiming a client retry key for a fine-tune start request.
#[derive(Debug, Clone, PartialEq)]
pub enum StartIdempotencyClaim {
    /// This caller owns the first execution for the key.
    New,
    /// A completed response can be replayed without starting another job.
    Replay(Value),
    /// Another request with the same key is still executing.
    InProgress,
    /// The key was already used for a different request body.
    Conflict,
}

fn valid_idempotency_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDEMPOTENCY_KEY_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

/// Verified caller context attached by Core plus the server-owned node identity
/// for this sidecar's store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinetuneTenantContext {
    pub owner_user_id: Option<String>,
    pub org_id: Option<String>,
    pub node_id: String,
}

impl FinetuneTenantContext {
    pub fn local(node_id: impl Into<String>) -> Self {
        Self {
            owner_user_id: None,
            org_id: None,
            node_id: node_id.into(),
        }
    }
}

/// Persisted tenancy metadata. Missing metadata represents legacy ownerless
/// state and deliberately does not match any request tenant.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinetuneTenantScope {
    #[serde(default)]
    pub owner_user_id: Option<String>,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub node_id: Option<String>,
}

impl FinetuneTenantScope {
    pub fn from_context(context: &FinetuneTenantContext) -> Self {
        Self {
            owner_user_id: context.owner_user_id.clone(),
            org_id: context.org_id.clone(),
            node_id: Some(context.node_id.clone()),
        }
    }

    pub fn matches(&self, context: &FinetuneTenantContext) -> bool {
        self.node_id.as_deref() == Some(context.node_id.as_str())
            && self.owner_user_id == context.owner_user_id
            && self.org_id == context.org_id
    }
}

/// One fine-tune job as Core records it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuneJob {
    /// Sidecar-assigned job id (also the row primary key).
    pub id: String,
    #[serde(default)]
    pub tenant: FinetuneTenantScope,
    /// HF repo id of the base model being tuned.
    pub base_model: String,
    /// Stem the trained adapter is saved under (`None` until known).
    pub output_name: Option<String>,
    /// Coarse lifecycle state, mirrored from the sidecar:
    /// `queued | running | succeeded | failed | cancelled`.
    pub state: String,
    /// Where the job runs: `local` | `remote` (Unit 5).
    pub target: String,
    /// For a remote job: the remote node's base URL (e.g. `https://node.example`).
    /// Safe to expose; Core proxies status/stream/cancel here. `None` for local.
    pub remote_url: Option<String>,
    /// For a remote job: the bearer token for that node. Kept in memory only
    /// after the encrypted-at-rest SQLite envelope is opened; NEVER serialized
    /// back to API clients.
    #[serde(skip_serializing, default)]
    pub remote_token: Option<String>,
    /// On-disk path/ref of the produced adapter once finished (`None` otherwise).
    pub output_ref: Option<String>,
    /// Terminal error message when `state == "failed"`.
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// SQLite-backed job store, safe to clone (shares one connection behind a mutex).
#[derive(Clone)]
pub struct FinetuneStore {
    conn: Arc<Mutex<Connection>>,
    node_id: String,
}

fn node_id_for_state_path(path: &Path) -> String {
    if let Some(value) = std::env::var("RYU_NODE_ID")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && !value.chars().any(char::is_control))
    {
        return value;
    }
    let data_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .unwrap_or_else(|_| {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    format!("local:{}", data_dir.display())
}

/// Single-row read, shared by [`FinetuneStore::get`] and the read half of
/// [`FinetuneStore::sync_from_snapshot`] so the two can never select a different
/// column order than [`FinetuneStore::map_row`] expects.
const SELECT_ONE_SQL: &str = "SELECT id, owner_user_id, org_id, node_id, base_model, output_name, state, target, remote_url, remote_token, output_ref, error, created_at, updated_at
     FROM finetune_jobs WHERE id = ?1";

/// Post-poll mutable-field write, shared by [`FinetuneStore::update_state`] and
/// [`FinetuneStore::sync_from_snapshot`]. `COALESCE` keeps a previously-recorded
/// `output_ref`/`error` when a later snapshot omits it.
const UPDATE_STATE_SQL: &str = "UPDATE finetune_jobs
     SET state = ?2,
         output_ref = COALESCE(?3, output_ref),
         error = COALESCE(?4, error),
         updated_at = ?5
     WHERE id = ?1";

fn state_rank(state: &str) -> Option<u8> {
    match state {
        "queued" => Some(0),
        "running" => Some(1),
        "succeeded" | "failed" | "cancelled" => Some(2),
        _ => None,
    }
}

/// Worker polls can complete out of order. State updates may advance a job, but
/// must never regress it or replace one terminal outcome with another.
fn state_transition_allowed(previous: &str, next: &str) -> bool {
    let Some(next_rank) = state_rank(next) else {
        return false;
    };
    let Some(previous_rank) = state_rank(previous) else {
        return true;
    };
    next_rank > previous_rank
        || (next_rank == previous_rank && (next_rank < 2 || previous.eq_ignore_ascii_case(next)))
}

impl FinetuneStore {
    /// Open (creating if needed) the store at the default `~/.ryu/finetune.db`.
    pub fn open_default() -> Result<Self> {
        Self::open(crate::data_dir().join("finetune.db"))
    }

    pub fn open(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating db dir {}", parent.display()))?;
            install_crypto_host(parent);
        }
        let conn = Connection::open(&path)
            .with_context(|| format!("opening finetune db {}", path.display()))?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            node_id: node_id_for_state_path(&path),
        })
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    fn tenant_scope(&self, context: &FinetuneTenantContext) -> Result<FinetuneTenantScope> {
        if context.node_id != self.node_id {
            anyhow::bail!("tenant node does not match the Fine-tune store node");
        }
        Ok(FinetuneTenantScope::from_context(context))
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS finetune_jobs (
                 id           TEXT PRIMARY KEY,
                 owner_user_id TEXT,
                 org_id       TEXT,
                 node_id      TEXT,
                 base_model   TEXT NOT NULL,
                 output_name  TEXT,
                 state        TEXT NOT NULL,
                 target       TEXT NOT NULL,
                 remote_url   TEXT,
                 remote_token TEXT,
                 output_ref   TEXT,
                 error        TEXT,
                 created_at   TEXT NOT NULL,
                 updated_at   TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_finetune_created
                 ON finetune_jobs(created_at DESC);
             CREATE TABLE IF NOT EXISTS finetune_start_idempotency (
                 idempotency_key TEXT PRIMARY KEY,
                 request_fingerprint TEXT NOT NULL,
                 response_json TEXT,
                 created_at INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_finetune_idempotency_updated
                 ON finetune_start_idempotency(updated_at);",
        )
        .context("initializing finetune schema")?;
        // Idempotent migration for stores created before the remote columns
        // existed (the feature is new, but a dev db may predate them). Ignore the
        // "duplicate column" error that fires when they're already present.
        for col in ["remote_url", "remote_token"] {
            let _ = conn.execute(
                &format!("ALTER TABLE finetune_jobs ADD COLUMN {col} TEXT"),
                [],
            );
        }
        for col in ["owner_user_id", "org_id", "node_id"] {
            let _ = conn.execute(
                &format!("ALTER TABLE finetune_jobs ADD COLUMN {col} TEXT"),
                [],
            );
        }
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_finetune_tenant
             ON finetune_jobs(node_id, org_id, owner_user_id, created_at DESC)",
            [],
        )?;
        Self::migrate_remote_tokens(conn)?;
        Ok(())
    }

    fn migrate_remote_tokens(conn: &Connection) -> Result<()> {
        let cipher = ryu_crypto::global_cipher().context("loading finetune storage cipher")?;
        let mut stmt = conn.prepare(
            "SELECT id, remote_token FROM finetune_jobs
             WHERE remote_token IS NOT NULL AND remote_token != ''",
        )?;
        let rows: Vec<(String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        drop(stmt);
        for (id, token) in rows {
            if ryu_crypto::FieldCipher::is_sealed(&token) {
                continue;
            }
            let sealed = cipher
                .seal(&token)
                .context("migrating remote fine-tune token")?;
            conn.execute(
                "UPDATE finetune_jobs SET remote_token = ?2 WHERE id = ?1",
                params![id, sealed],
            )?;
        }
        Ok(())
    }

    /// Insert a freshly-started job. Idempotent on `id` (re-insert replaces).
    pub async fn record(&self, job: &FinetuneJob) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO finetune_jobs
               (id, owner_user_id, org_id, node_id, base_model, output_name, state, target,
                remote_url, remote_token, output_ref, error, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                job.id,
                job.tenant.owner_user_id,
                job.tenant.org_id,
                job.tenant.node_id,
                job.base_model,
                job.output_name,
                job.state,
                job.target,
                job.remote_url,
                seal_remote_token(job.remote_token.as_deref())?,
                job.output_ref,
                job.error,
                job.created_at,
                job.updated_at,
            ],
        )
        .context("inserting finetune job")?;
        Ok(())
    }

    /// Insert a newly accepted job without replacing an existing row. Worker
    /// and remote-node job ids are outside this store's trust boundary; treating
    /// a collision as a no-op keeps a compromised source from overwriting a
    /// different user's or run's durable record.
    pub async fn record_if_absent(&self, job: &FinetuneJob) -> Result<bool> {
        let conn = self.conn.lock().await;
        let changed = conn
            .execute(
                "INSERT OR IGNORE INTO finetune_jobs
                   (id, owner_user_id, org_id, node_id, base_model, output_name, state, target,
                    remote_url, remote_token, output_ref, error, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                params![
                    job.id,
                    job.tenant.owner_user_id,
                    job.tenant.org_id,
                    job.tenant.node_id,
                    job.base_model,
                    job.output_name,
                    job.state,
                    job.target,
                    job.remote_url,
                    seal_remote_token(job.remote_token.as_deref())?,
                    job.output_ref,
                    job.error,
                    job.created_at,
                    job.updated_at,
                ],
            )
            .context("inserting finetune job if absent")?;
        Ok(changed > 0)
    }

    pub async fn record_for(
        &self,
        context: &FinetuneTenantContext,
        job: &FinetuneJob,
    ) -> Result<()> {
        let mut job = job.clone();
        job.tenant = self.tenant_scope(context)?;
        self.record(&job).await
    }

    pub async fn record_if_absent_for(
        &self,
        context: &FinetuneTenantContext,
        job: &FinetuneJob,
    ) -> Result<bool> {
        let mut job = job.clone();
        job.tenant = self.tenant_scope(context)?;
        self.record_if_absent(&job).await
    }

    /// Atomically claim a start retry key before any worker/remote request is
    /// made. The response is persisted after the first execution completes, so
    /// a lost HTTP response can be replayed without launching a second GPU job.
    pub async fn claim_start(
        &self,
        idempotency_key: &str,
        request_fingerprint: &str,
    ) -> Result<StartIdempotencyClaim> {
        if !valid_idempotency_key(idempotency_key) {
            anyhow::bail!("invalid finetune idempotency key");
        }
        if request_fingerprint.is_empty()
            || request_fingerprint.len() > MAX_IDEMPOTENCY_FINGERPRINT_BYTES
        {
            anyhow::bail!("invalid finetune request fingerprint");
        }
        let conn = self.conn.lock().await;
        let transaction = conn.unchecked_transaction()?;
        let existing: Option<(String, Option<String>)> = transaction
            .query_row(
                "SELECT request_fingerprint, response_json
                 FROM finetune_start_idempotency WHERE idempotency_key = ?1",
                params![idempotency_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((previous_fingerprint, response_json)) = existing {
            let claim = if previous_fingerprint != request_fingerprint {
                StartIdempotencyClaim::Conflict
            } else if let Some(response_json) = response_json {
                StartIdempotencyClaim::Replay(serde_json::from_str(&response_json)?)
            } else {
                StartIdempotencyClaim::InProgress
            };
            transaction.commit()?;
            return Ok(claim);
        }

        let now = now_millis();
        let retention_cutoff = now.saturating_sub(7 * 24 * 60 * 60 * 1000);
        transaction.execute(
            "DELETE FROM finetune_start_idempotency
             WHERE response_json IS NOT NULL AND updated_at < ?1",
            params![retention_cutoff],
        )?;
        let record_count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM finetune_start_idempotency",
            [],
            |row| row.get(0),
        )?;
        if record_count >= MAX_IDEMPOTENCY_RECORDS {
            anyhow::bail!("finetune idempotency history is full");
        }
        let pending_count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM finetune_start_idempotency WHERE response_json IS NULL",
            [],
            |row| row.get(0),
        )?;
        if pending_count >= MAX_IDEMPOTENCY_PENDING {
            anyhow::bail!("too many fine-tune starts are already in progress");
        }
        transaction.execute(
            "INSERT INTO finetune_start_idempotency
             (idempotency_key, request_fingerprint, response_json, created_at, updated_at)
             VALUES (?1, ?2, NULL, ?3, ?3)",
            params![idempotency_key, request_fingerprint, now],
        )?;
        transaction.commit()?;
        Ok(StartIdempotencyClaim::New)
    }

    /// Store the first successful response for a claimed start key.
    pub async fn complete_start(
        &self,
        idempotency_key: &str,
        request_fingerprint: &str,
        response: &Value,
    ) -> Result<bool> {
        let response_json = serde_json::to_string(response)?;
        if response_json.len() > MAX_IDEMPOTENCY_RESPONSE_BYTES {
            anyhow::bail!("finetune idempotency response is too large");
        }
        let conn = self.conn.lock().await;
        let changed = conn.execute(
            "UPDATE finetune_start_idempotency
             SET response_json = ?1, updated_at = ?2
             WHERE idempotency_key = ?3 AND request_fingerprint = ?4
               AND response_json IS NULL",
            params![
                response_json,
                now_millis(),
                idempotency_key,
                request_fingerprint
            ],
        )?;
        Ok(changed > 0)
    }

    /// Release a failed claim so a client can safely retry the same request.
    pub async fn release_start(
        &self,
        idempotency_key: &str,
        request_fingerprint: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        let changed = conn.execute(
            "DELETE FROM finetune_start_idempotency
             WHERE idempotency_key = ?1 AND request_fingerprint = ?2 AND response_json IS NULL",
            params![idempotency_key, request_fingerprint],
        )?;
        Ok(changed > 0)
    }

    /// Update the mutable fields of a job after a status poll or terminal event.
    pub async fn update_state(
        &self,
        id: &str,
        state: &str,
        output_ref: Option<&str>,
        error: Option<&str>,
        updated_at: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().await;
        if let Some(previous) = Self::read_one(&conn, id)? {
            if !state_transition_allowed(&previous.state, state) {
                return Ok(false);
            }
        }
        let n = conn
            .execute(
                UPDATE_STATE_SQL,
                params![id, state, output_ref, error, updated_at],
            )
            .context("updating finetune job")?;
        Ok(n > 0)
    }

    pub async fn update_state_for(
        &self,
        context: &FinetuneTenantContext,
        id: &str,
        state: &str,
        output_ref: Option<&str>,
        error: Option<&str>,
        updated_at: &str,
    ) -> Result<bool> {
        self.tenant_scope(context)?;
        if self.get_for(context, id).await?.is_none() {
            return Ok(false);
        }
        self.update_state(id, state, output_ref, error, updated_at)
            .await
    }

    /// [`Self::update_state`], but returning the record **as it stood before the
    /// write** (`None` for an id this node never recorded or a stale/invalid
    /// out-of-order snapshot).
    ///
    /// The read and the write share ONE lock acquisition, and that is the whole
    /// point: the stored state is the only memory of what the worker was last seen
    /// doing, so the write is what *claims* a transition. Split into a `get` then an
    /// `update_state`, the `/list` and `/:id` polls the desktop runs concurrently
    /// could both read `running`, both conclude the job had just finished, and both
    /// announce it — firing a subscribing workflow twice for one training run.
    /// Comparing `prior.state` to `state` under this guarantee makes exactly one
    /// caller the observer of any given transition.
    ///
    /// Returning the prior row also spares the caller a second read for the fields
    /// a poll never changes (`base_model`, `output_name`, `target`).
    pub async fn sync_from_snapshot(
        &self,
        id: &str,
        state: &str,
        output_ref: Option<&str>,
        error: Option<&str>,
        updated_at: &str,
    ) -> Result<Option<FinetuneJob>> {
        let conn = self.conn.lock().await;
        let prior = Self::read_one(&conn, id)?;
        if let Some(previous) = &prior {
            if !state_transition_allowed(&previous.state, state) {
                return Ok(None);
            }
            conn.execute(
                UPDATE_STATE_SQL,
                params![id, state, output_ref, error, updated_at],
            )
            .context("updating finetune job")?;
        }
        Ok(prior)
    }

    pub async fn sync_from_snapshot_for(
        &self,
        context: &FinetuneTenantContext,
        id: &str,
        state: &str,
        output_ref: Option<&str>,
        error: Option<&str>,
        updated_at: &str,
    ) -> Result<Option<FinetuneJob>> {
        self.tenant_scope(context)?;
        if self.get_for(context, id).await?.is_none() {
            return Ok(None);
        }
        self.sync_from_snapshot(id, state, output_ref, error, updated_at)
            .await
    }

    pub async fn get(&self, id: &str) -> Result<Option<FinetuneJob>> {
        let conn = self.conn.lock().await;
        Self::read_one(&conn, id)
    }

    pub async fn get_for(
        &self,
        context: &FinetuneTenantContext,
        id: &str,
    ) -> Result<Option<FinetuneJob>> {
        self.tenant_scope(context)?;
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, owner_user_id, org_id, node_id, base_model, output_name, state, target,
                    remote_url, remote_token, output_ref, error, created_at, updated_at
             FROM finetune_jobs
             WHERE id = ?1 AND (
                 (node_id = ?2 AND owner_user_id IS ?3 AND org_id IS ?4)
                 OR (node_id IS NULL AND ?3 IS NULL AND ?4 IS NULL)
             )",
        )?;
        let mut rows = stmt.query_map(
            params![
                id,
                context.node_id,
                context.owner_user_id,
                context.org_id
            ],
            Self::map_row,
        )?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Read one row on an already-held connection guard, so a caller that must read
    /// and write atomically does not have to re-lock between the two.
    fn read_one(conn: &Connection, id: &str) -> Result<Option<FinetuneJob>> {
        let mut stmt = conn.prepare(SELECT_ONE_SQL)?;
        let mut rows = stmt.query_map(params![id], Self::map_row)?;
        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub async fn list(&self) -> Result<Vec<FinetuneJob>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, owner_user_id, org_id, node_id, base_model, output_name, state, target,
                    remote_url, remote_token, output_ref, error, created_at, updated_at
             FROM finetune_jobs ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], Self::map_row)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub async fn list_for(&self, context: &FinetuneTenantContext) -> Result<Vec<FinetuneJob>> {
        self.tenant_scope(context)?;
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, owner_user_id, org_id, node_id, base_model, output_name, state, target,
                    remote_url, remote_token, output_ref, error, created_at, updated_at
             FROM finetune_jobs
             WHERE (node_id = ?1 AND owner_user_id IS ?2 AND org_id IS ?3)
                OR (node_id IS NULL AND ?2 IS NULL AND ?3 IS NULL)
             ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map(
            params![context.node_id, context.owner_user_id, context.org_id],
            Self::map_row,
        )?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<FinetuneJob> {
        Ok(FinetuneJob {
            id: row.get(0)?,
            tenant: FinetuneTenantScope {
                owner_user_id: row.get(1)?,
                org_id: row.get(2)?,
                node_id: row.get(3)?,
            },
            base_model: row.get(4)?,
            output_name: row.get(5)?,
            state: row.get(6)?,
            target: row.get(7)?,
            remote_url: row.get(8)?,
            remote_token: open_remote_token(row.get(9)?)?,
            output_ref: row.get(10)?,
            error: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
        })
    }
}
