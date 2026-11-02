//! Validation of job ids, model ids, output names and remote tokens.

use super::*;

/// Job ids are interpolated into worker and remote URL paths. Keep the accepted
/// alphabet deliberately narrower than a generic URL segment so a compromised
/// worker or remote node cannot turn its response into a path/query injection.
pub(super) fn valid_job_id(id: &str) -> bool {
    !matches!(id, "." | "..")
        && !id.is_empty()
        && id.chars().count() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

/// The worker passes this directly to the model loader. Restrict it to the
/// Hugging Face `namespace/model` form so an authenticated caller cannot turn
/// the training sidecar into an arbitrary local-path or URL loader.
pub(super) fn valid_base_model_id(id: &str) -> bool {
    let mut parts = id.split('/');
    let Some(namespace) = parts.next() else {
        return false;
    };
    let Some(model) = parts.next() else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    [namespace, model].into_iter().all(|part| {
        !part.is_empty()
            && part.chars().count() <= 128
            && part
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    })
}

/// A fresh adapter directory id: `ft-` + 32 lowercase hex characters, the exact
/// shape the worker accepts as `artifact_id`.
pub fn new_artifact_id() -> String {
    format!("ft-{}", uuid::Uuid::new_v4().simple())
}

/// The stem a merge writes `models/chat/<stem>.gguf` under: `base` (kept short
/// enough that the result is a valid output name) plus `-` and 12 random hex
/// characters. `base` must already be a valid output name.
pub fn merged_model_stem(base: &str) -> String {
    let base: String = base.chars().take(115).collect();
    let base = if valid_output_name(&base) {
        base
    } else {
        "merged".to_owned()
    };
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    format!("{base}-{}", &suffix[..12])
}

pub(super) fn valid_output_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= 128
        && name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

pub(super) fn valid_remote_token(token: &str) -> bool {
    !token.is_empty() && token.len() <= 4096 && !token.chars().any(char::is_control)
}

pub(super) fn response_job_state(response: &Value) -> String {
    response
        .get("state")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|state| {
            matches!(
                *state,
                "queued" | "running" | "succeeded" | "failed" | "cancelled"
            )
        })
        .unwrap_or("running")
        .to_owned()
}
