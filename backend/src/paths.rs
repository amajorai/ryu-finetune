//! Inlined data-dir resolution (tracer copy of `apps/core/src/paths.rs`, matching
//! `apps-store/mail/backend/src/paths.rs`).
//!
//! The sidecar MUST resolve the SAME data dir Core uses so it opens the SAME
//! `finetune.db` (and shares `installed-adapters.json` / `installed-models.json`).
//! The load-bearing rule is `RYU_DIR`-env-first: Core/Kernel passes
//! `RYU_DIR` to the sidecar at spawn, guaranteeing co-location. The pointer-file
//! read + `RYU_PROFILE` suffix are replicated for faithfulness in the headless
//! case, but env-first + default is what actually guarantees the shared path.

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

const RYU_DIR_ENV: &str = "RYU_DIR";
const RYU_PROFILE_ENV: &str = "RYU_PROFILE";
const RELEASE_PROFILE: &str = "release";

/// Data-dir / config-dir suffix for the active profile: `""` for release,
/// `-<profile>` otherwise (e.g. `-dev`). Mirrors `crate::profile::suffix`.
fn suffix() -> String {
    let profile = std::env::var(RYU_PROFILE_ENV)
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| RELEASE_PROFILE.to_string());
    if profile == RELEASE_PROFILE {
        String::new()
    } else {
        format!("-{}", profile.trim())
    }
}

/// The default data dir: `~/.ryu{suffix}` (falling back to `./.ryu` if home is
/// unknown).
fn default_ryu_dir() -> PathBuf {
    let name = format!(".ryu{}", suffix());
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(name)
}

/// Config dir holding the bootstrap pointer file (`ryu{suffix}` under the OS
/// config dir), NOT inside the data dir.
fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(default_ryu_dir)
        .join(format!("ryu{}", suffix()))
}

fn pointer_path() -> PathBuf {
    config_dir().join("data-path.json")
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct DataPathPointer {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data_dir: Option<String>,
}

fn read_pointer() -> DataPathPointer {
    let Ok(bytes) = std::fs::read(pointer_path()) else {
        return DataPathPointer::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn resolve() -> PathBuf {
    if let Some(v) = std::env::var_os(RYU_DIR_ENV) {
        let p = PathBuf::from(v);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Some(dir) = read_pointer().data_dir {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    default_ryu_dir()
}

static RYU_DIR: OnceLock<PathBuf> = OnceLock::new();

/// The active data dir, resolved once and cached for the process lifetime.
pub fn ryu_dir() -> PathBuf {
    RYU_DIR.get_or_init(resolve).clone()
}

#[cfg(test)]
mod tests {
    use super::{
        config_dir, default_ryu_dir, pointer_path, read_pointer, resolve, ryu_dir, suffix,
        RYU_DIR_ENV, RYU_PROFILE_ENV,
    };
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    // `resolve`/`suffix`/`default_ryu_dir` read process-wide env; serialize the
    // env-mutating tests so parallel threads don't clobber each other's setup.
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Snapshot the two env vars, run `f`, then restore them exactly.
    fn with_env<R>(f: impl FnOnce() -> R) -> R {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev_dir = std::env::var_os(RYU_DIR_ENV);
        let prev_profile = std::env::var_os(RYU_PROFILE_ENV);
        let out = f();
        match prev_dir {
            Some(v) => std::env::set_var(RYU_DIR_ENV, v),
            None => std::env::remove_var(RYU_DIR_ENV),
        }
        match prev_profile {
            Some(v) => std::env::set_var(RYU_PROFILE_ENV, v),
            None => std::env::remove_var(RYU_PROFILE_ENV),
        }
        out
    }

    #[test]
    fn suffix_is_empty_for_release_and_unset_but_dashed_otherwise() {
        with_env(|| {
            std::env::remove_var(RYU_PROFILE_ENV);
            assert_eq!(suffix(), "");
            std::env::set_var(RYU_PROFILE_ENV, "release");
            assert_eq!(suffix(), "");
            std::env::set_var(RYU_PROFILE_ENV, "  "); // blank → treated as unset
            assert_eq!(suffix(), "");
            std::env::set_var(RYU_PROFILE_ENV, "dev");
            assert_eq!(suffix(), "-dev");
        });
    }

    #[test]
    fn resolve_prefers_ryu_dir_env() {
        with_env(|| {
            std::env::set_var(RYU_DIR_ENV, "/tmp/explicit-ryu-dir");
            assert_eq!(resolve(), std::path::PathBuf::from("/tmp/explicit-ryu-dir"));
            // An empty RYU_DIR is ignored (falls through to the default).
            std::env::set_var(RYU_DIR_ENV, "");
            std::env::set_var(RYU_PROFILE_ENV, "release");
            assert!(resolve().to_string_lossy().ends_with(".ryu"));
        });
    }

    #[test]
    fn resolve_falls_back_to_profile_suffixed_default() {
        // A unique profile guarantees an empty config dir (no pointer file), so
        // resolve lands on `~/.ryu-<profile>` deterministically without polluting
        // any real config location.
        let profile = format!("test{}", SEQ.fetch_add(1, Ordering::Relaxed));
        with_env(|| {
            std::env::remove_var(RYU_DIR_ENV);
            std::env::set_var(RYU_PROFILE_ENV, &profile);
            let got = resolve();
            assert!(
                got.to_string_lossy().ends_with(&format!(".ryu-{profile}")),
                "unexpected resolved dir: {}",
                got.display()
            );
            // No pointer file at the unique config dir → default pointer (no data_dir).
            assert!(read_pointer().data_dir.is_none());
            // The default helper agrees with resolve here.
            assert_eq!(default_ryu_dir(), got);
        });
    }

    #[test]
    fn config_and_pointer_paths_compose() {
        with_env(|| {
            std::env::set_var(RYU_PROFILE_ENV, "dev");
            let cfg = config_dir();
            assert!(cfg.ends_with("ryu-dev"));
            assert_eq!(pointer_path(), cfg.join("data-path.json"));
        });
    }

    #[test]
    fn ryu_dir_is_stable_across_calls() {
        // Exercises the cached accessor; value is env-dependent so only stability
        // (idempotence of the OnceLock) is asserted.
        assert_eq!(ryu_dir(), ryu_dir());
    }
}
