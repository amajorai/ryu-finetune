//! Inbox, toast and push copy as localizable messages.
//!
//! Core renders a notify hint's `i18n` text per recipient (the inbox row and
//! toast in the node's `ui-language`, each push in the locale its device
//! registered with); the English rendering still goes out as the plain
//! `title`/`body`, which an older Core delivers as is.
//!
//! Each `*_MESSAGE` is `format!` with one named argument per placeholder
//! (`name = arg("name")`) so the i18n extractor catalogs the ICU source;
//! `tools/generate-core-i18n.mjs` copies the translations into Core's catalog.
//! Every message is a whole sentence. Core's renderer has no plural selector,
//! so counts are written as `label: {count}`.
#![allow(clippy::useless_format)]

use ryu_app_events::{NotifyHint, NotifyMessage, NotifyTemplate as Template};

pub(crate) const FINETUNE_FINISHED_MESSAGE: Template =
    |arg| format!("Fine-tune “{name}” finished", name = arg("name"));
pub(crate) const FINETUNE_JOB_FINISHED_MESSAGE: Template = |_| format!("Fine-tune job finished");
pub(crate) const ADAPTER_READY_MESSAGE: Template =
    |arg| format!("Adapter ready for {model}", model = arg("model"));
pub(crate) const FINETUNE_FAILED_MESSAGE: Template =
    |arg| format!("Fine-tune “{name}” failed", name = arg("name"));
pub(crate) const FINETUNE_JOB_FAILED_MESSAGE: Template = |_| format!("Fine-tune job failed");

fn titled(named: Template, unnamed: Template, name: Option<&str>) -> NotifyMessage {
    match name.filter(|name| !name.trim().is_empty()) {
        Some(name) => NotifyMessage::new(named).text("name", name),
        None => NotifyMessage::new(unnamed),
    }
}

/// A training job finished; its adapter is ready.
pub(crate) fn job_finished(output_name: Option<&str>, base_model: &str) -> NotifyHint {
    NotifyHint::localized(
        titled(
            FINETUNE_FINISHED_MESSAGE,
            FINETUNE_JOB_FINISHED_MESSAGE,
            output_name,
        ),
        vec![NotifyMessage::new(ADAPTER_READY_MESSAGE).text("model", base_model)],
    )
    .with_level("success")
}

/// A training job failed; the body relays the worker's error, which Core
/// translates when it is one of the worker's catalogued sentences.
pub(crate) fn job_failed(output_name: Option<&str>, error: Option<&str>) -> NotifyHint {
    NotifyHint::localized(
        titled(
            FINETUNE_FAILED_MESSAGE,
            FINETUNE_JOB_FAILED_MESSAGE,
            output_name,
        ),
        error.map(NotifyMessage::relayed).into_iter().collect(),
    )
    .with_level("error")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_job_relays_the_worker_error_for_core_to_translate() {
        let hint = job_failed(Some("support-bot"), Some("dataset has no samples"));
        assert_eq!(hint.title, "Fine-tune “support-bot” failed");
        assert_eq!(hint.body.as_deref(), Some("dataset has no samples"));
        let body = &serde_json::to_value(&hint).unwrap()["i18n"]["body"][0];
        assert_eq!(body["args"]["text"], "dataset has no samples");
        assert_eq!(body["relay"], true);
        assert!(job_failed(None, None).body.is_none());
    }
}
