//! Trained-output (adapter) catalog
//! (`<data dir>/plugin-data/@ryu+finetune/installed-adapters.json`).
//!
//! Mirrors Core's `model_catalog::installed` index but for LoRA adapters produced
//! by fine-tune jobs. An adapter is a directory under `<data dir>/models/adapters/<stem>/`
//! (`adapter_config.json` + `adapter_model.safetensors`); this index records its
//! provenance (base model, the job that produced it) so the desktop can list and
//! merge it. The loader drops records whose directory was deleted out-of-band so
//! the catalog never shows a phantom.
//!
//! The index is keyed by the PRODUCING JOB's id, and `stem` is the adapter's
//! server-generated directory name. A caller-chosen output name is only the
//! display `name`, so two jobs (or two tenants) that pick the same name each keep
//! their own record and directory. Records written before this keying (keyed by
//! `stem`) are still read; re-recording the same job replaces its legacy entry.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// One installed adapter as Core records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledAdapter {
    /// The directory name under `<data dir>/models/adapters/<stem>/`.
    pub stem: String,
    /// The job's human-readable `output_name`, when it had one. Display only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// HF repo id of the base model this adapter was tuned from.
    pub base_model: String,
    /// The fine-tune job that produced it.
    pub job_id: String,
    /// Absolute on-disk path of the adapter directory.
    pub path: String,
    pub created_at: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AdaptersFile {
    /// Keyed by the producing job's id (legacy files: by `stem`).
    #[serde(default)]
    adapters: HashMap<String, InstalledAdapter>,
}

// Serializes concurrent read-modify-write of the JSON store.
static LOCK: Mutex<()> = Mutex::new(());

fn store_path() -> PathBuf {
    crate::data_dir().join("installed-adapters.json")
}

/// Record a freshly-produced adapter. Idempotent on the producing job; never
/// replaces another job's record. Atomic (tmp+rename).
pub fn record(adapter: InstalledAdapter) -> Result<()> {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = store_path();
    let mut file: AdaptersFile = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    upsert(&mut file.adapters, adapter);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&file)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Load every recorded adapter whose directory still exists on disk.
pub fn load_present() -> Vec<InstalledAdapter> {
    let raw = match std::fs::read_to_string(store_path()) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let parsed: AdaptersFile = serde_json::from_str(&raw).unwrap_or_default();
    parsed
        .adapters
        .into_values()
        .filter(|a| PathBuf::from(&a.path).exists())
        .collect()
}

/// Insert under the job id, dropping only a legacy (stem-keyed) entry for the
/// SAME job. Another job's record is never touched, whatever its stem or name.
fn upsert(adapters: &mut HashMap<String, InstalledAdapter>, adapter: InstalledAdapter) {
    adapters.retain(|key, existing| key == &adapter.job_id || existing.job_id != adapter.job_id);
    adapters.insert(adapter.job_id.clone(), adapter);
}

/// Why [`resolve_owned`] refused to pick an adapter.
#[derive(Debug, PartialEq, Eq)]
pub enum ResolveError {
    /// `adapter_name` and `adapter_path` matched two different adapters.
    Ambiguous,
}

/// Whether two adapter paths name the same directory. Paths are compared after
/// canonicalization so `x/../victim` or a trailing slash cannot dodge the match.
fn same_dir(requested: &str, recorded: &str) -> bool {
    match (
        std::fs::canonicalize(requested),
        std::fs::canonicalize(recorded),
    ) {
        (Ok(requested), Ok(recorded)) => requested == recorded,
        _ => requested == recorded,
    }
}

/// The single adapter named by `adapter_name` (its `stem`) and/or `adapter_path`,
/// chosen ONLY among records produced by `owned_jobs`. `Ok(None)` when no owned
/// record matches; [`ResolveError::Ambiguous`] when the two selectors disagree or
/// a selector matches more than one owned record.
pub fn resolve_owned(
    adapters: Vec<InstalledAdapter>,
    owned_jobs: &HashSet<String>,
    adapter_name: Option<&str>,
    adapter_path: Option<&str>,
) -> std::result::Result<Option<InstalledAdapter>, ResolveError> {
    let owned: Vec<InstalledAdapter> = adapters
        .into_iter()
        .filter(|adapter| owned_jobs.contains(&adapter.job_id))
        .collect();
    let pick = |matches: &dyn Fn(&InstalledAdapter) -> bool| {
        let found: Vec<&InstalledAdapter> =
            owned.iter().filter(|adapter| matches(adapter)).collect();
        match found.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some((*one).clone())),
            _ => Err(ResolveError::Ambiguous),
        }
    };
    let by_name = match adapter_name {
        Some(name) => Some(pick(&|adapter| adapter.stem == name)?),
        None => None,
    };
    let by_path = match adapter_path {
        Some(path) => Some(pick(&|adapter| same_dir(path, &adapter.path))?),
        None => None,
    };
    match (by_name, by_path) {
        (Some(Some(a)), Some(Some(b))) if a.job_id != b.job_id => Err(ResolveError::Ambiguous),
        // One selector named an owned adapter and the other named nothing this
        // tenant owns (possibly another tenant's): refuse rather than guess.
        (Some(Some(_)), Some(None)) | (Some(None), Some(Some(_))) => Err(ResolveError::Ambiguous),
        (Some(found), _) | (None, Some(found)) => Ok(found),
        (None, None) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(job_id: &str, stem: &str, name: Option<&str>) -> InstalledAdapter {
        InstalledAdapter {
            stem: stem.to_owned(),
            name: name.map(str::to_owned),
            base_model: "unsloth/base".to_owned(),
            job_id: job_id.to_owned(),
            path: format!("/models/adapters/{stem}"),
            created_at: "now".to_owned(),
        }
    }

    /// The audit case: tenant B's job reused tenant A's output name and the
    /// stem-keyed catalog replaced A's record. Keyed by job, both survive.
    #[test]
    fn a_same_name_job_never_displaces_another_jobs_record() {
        let mut catalog = HashMap::new();
        upsert(&mut catalog, adapter("job-a", "ft-a", Some("shared")));
        upsert(&mut catalog, adapter("job-b", "ft-b", Some("shared")));
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog["job-a"].path, "/models/adapters/ft-a");
        assert_eq!(catalog["job-b"].path, "/models/adapters/ft-b");
    }

    #[test]
    fn re_recording_a_job_replaces_only_its_own_legacy_entry() {
        let mut catalog = HashMap::new();
        // Legacy files keyed records by stem.
        catalog.insert("legacy".to_owned(), adapter("job-a", "legacy", None));
        catalog.insert("other".to_owned(), adapter("job-b", "other", None));
        upsert(&mut catalog, adapter("job-a", "legacy", Some("legacy")));
        assert_eq!(catalog.len(), 2);
        assert!(catalog.contains_key("job-a"));
        assert!(catalog.contains_key("other"));
    }

    #[test]
    fn resolution_only_considers_the_callers_own_adapters() {
        let catalog = vec![
            adapter("job-a", "ft-a", Some("shared")),
            adapter("job-b", "ft-b", Some("shared")),
        ];
        let tenant_b: HashSet<String> = ["job-b".to_owned()].into();
        // B names A's adapter by stem or by path: not resolvable for B.
        assert_eq!(
            resolve_owned(catalog.clone(), &tenant_b, Some("ft-a"), None),
            Ok(None)
        );
        assert_eq!(
            resolve_owned(
                catalog.clone(),
                &tenant_b,
                None,
                Some("/models/adapters/ft-a")
            ),
            Ok(None)
        );
        // B's own adapter by name, and a name/path pair smuggling A's path.
        let own = resolve_owned(catalog.clone(), &tenant_b, Some("ft-b"), None)
            .unwrap()
            .unwrap();
        assert_eq!(own.job_id, "job-b");
        assert_eq!(
            resolve_owned(
                catalog,
                &tenant_b,
                Some("ft-b"),
                Some("/models/adapters/ft-a")
            ),
            Err(ResolveError::Ambiguous)
        );
    }

    #[test]
    fn selectors_must_all_name_the_same_owned_record_however_spelled() {
        let root =
            std::env::temp_dir().join(format!("ryu-finetune-merge-resolve-{}", std::process::id()));
        let mine_dir = root.join("mine");
        let victim_dir = root.join("victim");
        std::fs::create_dir_all(&mine_dir).unwrap();
        std::fs::create_dir_all(&victim_dir).unwrap();
        let record = |stem: &str, dir: &std::path::Path| InstalledAdapter {
            path: dir.to_string_lossy().to_string(),
            ..adapter(&format!("job-{stem}"), stem, None)
        };
        let records = vec![record("mine", &mine_dir), record("victim", &victim_dir)];
        let owned: HashSet<String> = ["job-mine".to_owned(), "job-victim".to_owned()].into();
        let resolved = |name: Option<&str>, path: Option<&str>| {
            resolve_owned(records.clone(), &owned, name, path)
                .ok()
                .flatten()
                .map(|a| a.stem)
        };

        assert_eq!(resolved(Some("mine"), None).as_deref(), Some("mine"));
        let mine_alias = mine_dir.join("..").join("mine");
        assert_eq!(
            resolved(None, Some(&mine_alias.to_string_lossy())).as_deref(),
            Some("mine")
        );
        // A name the caller owns must not vouch for a path to another record,
        // however the path is spelled, nor for a directory nobody recorded.
        let victim_alias = mine_dir.join("..").join("victim");
        for path in [
            victim_dir.to_string_lossy().to_string(),
            victim_alias.to_string_lossy().to_string(),
            root.join("unrecorded").to_string_lossy().to_string(),
        ] {
            assert_eq!(
                resolved(Some("mine"), Some(&path)),
                None,
                "mismatched selectors resolved for {path}"
            );
        }
        assert_eq!(resolved(Some("nobody"), None), None);
        let _ = std::fs::remove_dir_all(&root);
    }
}
