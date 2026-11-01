//! Fine-tuning (Unsloth) durable state — extracted Core capability crate.
//!
//! Core owns *what runs* (a fine-tune job on this node's GPU, or a remote Ryu
//! Cloud GPU node) and the durable job record; the actual training happens in the
//! out-of-process Python worker (`apps-store/finetune/sidecar`). This crate owns
//! the durable records it persists — the [`FinetuneStore`] job DB and the
//! [`adapters`] output catalog — AND the [`api`] `/api/finetune/*` HTTP surface
//! (relocated out of `apps/core/src/server/finetune.rs`), a thin proxy that drives
//! the Python worker over HTTP at [`api::FinetuneCtx::unsloth_url`] (the
//! worker's profile-shifted manifest port, [`api::DEFAULT_UNSLOTH_URL`] on release).
//! The surface runs BOTH in-process (Core merges [`routes`]) and out-of-process
//! (the `ryu-finetune` control-plane sidecar in `main.rs`).
//!
//! The lone kernel coupling — the data directory — is inverted through
//! [`init_data_dir`] (this app's `plugin-data/@ryu+finetune`) and
//! [`init_node_dir`] (the node's data-dir root, for at-rest key custody), so this
//! crate has ZERO dependency on `apps/core` (the `ryu-monitors` precedent).

use std::path::PathBuf;

pub mod adapters;
pub mod api;
pub mod store;

pub use api::{openapi, routes, FinetuneCtx};
pub use store::{
    FinetuneJob, FinetuneStore, FinetuneTenantContext, FinetuneTenantScope, StartIdempotencyClaim,
};

/// The crate's data directory (`finetune.db` and `installed-adapters.json` live
/// under it): the app's plugin data dir, `<data dir>/plugin-data/@ryu+finetune`.
/// Set once at startup; [`data_dir`] falls back to the system temp dir so unit
/// tests and any pre-init handler never panic.
static DATA_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// The node's data-dir root, which the at-rest cipher resolves its master key
/// against. Unset in tests, where the store's own directory stands in.
static NODE_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Publish the fine-tune data directory. Idempotent: a second call is ignored.
pub fn init_data_dir(dir: PathBuf) {
    let _ = DATA_DIR.set(dir);
}

/// Publish the node's data-dir root. Idempotent: a second call is ignored.
pub fn init_node_dir(dir: PathBuf) {
    let _ = NODE_DIR.set(dir);
}

/// The fine-tune data directory, or the system temp dir when uninitialized.
pub(crate) fn data_dir() -> PathBuf {
    DATA_DIR.get().cloned().unwrap_or_else(std::env::temp_dir)
}

/// The node's data-dir root, when the process published one.
pub(crate) fn node_dir() -> Option<PathBuf> {
    NODE_DIR.get().cloned()
}
