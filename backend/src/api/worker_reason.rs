//! The sentence the Companion shows when the Unsloth worker cannot train.
//!
//! The worker's `/health` reports a stable `reason_code` (see
//! `sidecar/ryu_unsloth/device.py`). Each code maps here to one whole sentence,
//! so the Companion's catalog can translate it; the worker's English `reason`
//! is diagnostic detail and is never spliced into user copy.

use serde_json::Value;

/// No reachable worker on this node.
pub(super) const LOCAL_WORKER_DOWN_MESSAGE: &str =
    "The Unsloth worker is unavailable — start it or use a remote GPU node.";

/// A worker that cannot train but reports no known `reason_code` (an older
/// worker, or a code this backend does not know yet).
pub(super) const WORKER_CANNOT_TRAIN_MESSAGE: &str =
    "The Unsloth worker cannot train on this node. Use a remote GPU node instead.";

/// The user-facing sentence for a not-ready worker `/health` payload.
pub(super) fn worker_reason(sidecar: &Value) -> String {
    let code = sidecar
        .get("reason_code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let cc = sidecar
        .get("compute_capability")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    match (code, cc) {
        ("torch_missing", _) => {
            "PyTorch is not installed in the Unsloth worker. Use a remote GPU node instead."
                .to_string()
        }
        ("unsloth_missing", _) => {
            "Unsloth is not installed in the worker. Use a remote GPU node instead.".to_string()
        }
        ("compute_capability_low", Some(cc)) => format!(
            "This GPU's compute capability is {cc}, below the 7.0 that Unsloth needs. \
             Use a remote GPU node instead."
        ),
        ("compute_capability_low", None) => {
            "This GPU's compute capability is below the 7.0 that Unsloth needs. \
             Use a remote GPU node instead."
                .to_string()
        }
        ("mps_unsupported", _) => {
            "Apple Silicon / unified memory detected — Unsloth training needs an NVIDIA CUDA GPU. \
             Use a remote GPU node instead."
                .to_string()
        }
        ("no_cuda_gpu", _) => {
            "No NVIDIA GPU detected — Unsloth training needs a CUDA GPU. Use a remote GPU node instead."
                .to_string()
        }
        ("probe_failed", _) => {
            "The Unsloth worker could not check this GPU. Use a remote GPU node instead."
                .to_string()
        }
        _ => WORKER_CANNOT_TRAIN_MESSAGE.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const SUFFIX: &str = "Use a remote GPU node instead.";

    #[test]
    fn every_code_maps_to_a_whole_sentence() {
        for code in [
            "torch_missing",
            "unsloth_missing",
            "compute_capability_low",
            "mps_unsupported",
            "no_cuda_gpu",
            "probe_failed",
        ] {
            let sentence = worker_reason(&json!({ "reason_code": code, "reason": "raw detail" }));
            assert!(sentence.ends_with(SUFFIX), "{code}: {sentence}");
            assert_ne!(sentence, WORKER_CANNOT_TRAIN_MESSAGE, "{code} fell through");
            assert!(
                !sentence.contains("raw detail"),
                "{code} spliced the worker reason"
            );
        }
    }

    #[test]
    fn compute_capability_is_named_when_known() {
        assert_eq!(
            worker_reason(
                &json!({ "reason_code": "compute_capability_low", "compute_capability": "6.1" })
            ),
            "This GPU's compute capability is 6.1, below the 7.0 that Unsloth needs. \
             Use a remote GPU node instead."
        );
    }

    #[test]
    fn a_probe_failure_never_shows_the_exception() {
        let sentence = worker_reason(&json!({
            "reason_code": "probe_failed",
            "reason": "GPU probe failed: CUDA driver /dev/nvidia0 exploded",
        }));
        assert!(!sentence.contains("nvidia0"));
    }

    #[test]
    fn an_older_worker_gets_the_generic_sentence() {
        assert_eq!(
            worker_reason(&json!({ "reason": "unsloth is not installed" })),
            WORKER_CANNOT_TRAIN_MESSAGE
        );
        assert_eq!(
            worker_reason(&json!({ "reason_code": "something_new" })),
            WORKER_CANNOT_TRAIN_MESSAGE
        );
    }
}
