//! Request body schemas for the fine-tuning surface (documentation only).

use super::*;

// ── Request bodies ──────────────────────────────────────────────────────────
//
// These types describe the wire shape; they are deliberately NOT used as axum
// extractors. `start` and `merge` are proxies: the body is forwarded verbatim to
// the Python worker (`apps-store/finetune/sidecar`, whose pydantic models are the
// contract of record) and, for a remote job, on to another node's Core. Putting a
// Rust struct in the extract path would make this crate a gatekeeper for a schema
// it does not own — every worker-side field addition would then need a Rust
// release to stop being silently dropped. So the handlers keep `Json<Value>` and
// the annotation carries the type, which is the half Core reads.
//
// They mirror `FinetuneRequest`/`MergeRequest` in `ryu_unsloth/server.py` field for
// field; change them together.

/// Request body for `POST /api/finetune/start`.
// Everything below is `//`, not `///`, ON PURPOSE: utoipa lifts a struct's doc
// comment into the schema's own `description`, so internal rationale written as
// `///` ships to the model alongside the arguments.
//
// The FIELD docs below, by contrast, are not decoration — utoipa lifts them
// verbatim into each property's `description`, and they are the only prose the
// model reads when it decides how to call the derived `start` tool.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct StartJobBody {
    /// Hugging Face repo id of the base model to fine-tune — ideally an
    /// `unsloth/*-bnb-4bit` build, which is what fits on a consumer GPU.
    pub base_model_id: String,
    /// Optional stable retry key. Reusing it returns the original accepted job;
    /// using it with a different request body is rejected.
    #[serde(default, alias = "idempotencyKey")]
    pub idempotency_key: Option<String>,
    /// The training data. Supply either inline `samples` or a `path` to a
    /// `.json`/`.jsonl` file with the same row shapes.
    pub dataset: DatasetSpec,
    /// Display name for the trained adapter, e.g. `my-tone-v1`. The adapter's
    /// directory is a server-generated id unique to the job, so two jobs (or two
    /// people) may use the same name without sharing or replacing an adapter.
    #[serde(default)]
    pub output_name: Option<String>,
    /// LoRA adapter shape. Every field is optional; omit the whole object to train
    /// with the worker's defaults.
    // `#[schema(inline)]` — NOT a doc comment: everything above IS lifted into the
    // schema and read by the model, and this rationale is not for it. An
    // `Option<Struct>` renders as `oneOf: [null, <schema>]`, and Core follows only a
    // `$ref` at the TOP of a node — a ref buried in that wrapper reaches the model
    // as an opaque pointer. Inlined, it sees the real sub-fields.
    #[serde(default)]
    #[schema(inline)]
    pub lora: Option<LoraSpec>,
    /// Training hyper-parameters. Every field is optional; omit the whole object to
    /// train with the worker's defaults.
    #[serde(default)]
    #[schema(inline)]
    pub training: Option<TrainingSpec>,
    /// Where to train: `local` (this node's GPU, the default) or `remote` (a Ryu
    /// Cloud GPU node, which then also needs `remote`).
    #[serde(default)]
    pub target: Option<String>,
    /// The GPU node to train on. Required when `target` is `remote`, ignored
    /// otherwise.
    #[serde(default)]
    #[schema(inline)]
    pub remote: Option<RemoteTarget>,
}

/// The training data for a fine-tune job. One of `samples` or `path` must be
/// present — the worker rejects a dataset with neither.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct DatasetSpec {
    /// Row shape: `chat` (default) — `{"messages":[{role,content}]}`; `alpaca` —
    /// `{instruction,input?,output}`; `text` — `{"text":"..."}`; or `preference` —
    /// `{prompt,chosen,rejected}` for DPO.
    #[serde(default)]
    pub format: Option<String>,
    /// The rows themselves, whose shape follows `format`. Left untyped because the
    /// three accepted row shapes are genuinely different objects.
    #[serde(default)]
    pub samples: Option<Vec<Value>>,
    /// Absolute path to a `.json` or `.jsonl` file of rows, as an alternative to
    /// sending them inline.
    #[serde(default)]
    pub path: Option<String>,
}

/// LoRA adapter shape (all optional — the worker fills its own defaults).
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct LoraSpec {
    /// Rank of the adapter. Higher trains more capacity and costs more VRAM.
    #[serde(default)]
    pub r: Option<u32>,
    /// LoRA alpha (scaling). Conventionally equal to, or twice, `r`.
    #[serde(default)]
    pub alpha: Option<u32>,
    /// Dropout applied to the adapter during training, 0.0–1.0.
    #[serde(default)]
    pub dropout: Option<f32>,
    /// Which projection modules to adapt, e.g. `["q_proj","v_proj"]`.
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,
}

/// Training hyper-parameters (all optional — the worker fills its own defaults).
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct TrainingSpec {
    /// Training objective: `sft` (default) or `dpo` preference optimization.
    #[serde(default)]
    pub method: Option<String>,
    /// DPO preference strength, greater than 0 and at most 1. Defaults to `0.1`.
    #[serde(default)]
    pub beta: Option<f32>,
    /// Passes over the dataset. Fractional values are allowed.
    #[serde(default)]
    pub epochs: Option<f32>,
    /// Hard cap on optimizer steps, which overrides `epochs` when set.
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// Learning rate, e.g. `0.0002`.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    /// Per-device batch size.
    #[serde(default)]
    pub batch_size: Option<u32>,
    /// Gradient-accumulation steps — raises the effective batch size without
    /// costing more VRAM.
    #[serde(default)]
    pub grad_accum: Option<u32>,
    /// Token context length each row is truncated to.
    #[serde(default)]
    pub max_seq_length: Option<u32>,
    /// Load the base model in 4-bit. On by default in the worker; this is the knob
    /// that decides whether a large model fits at all.
    #[serde(default)]
    pub load_in_4bit: Option<bool>,
    /// Seed, for a reproducible run.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// The remote Ryu node a `target: "remote"` job is dispatched to.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct RemoteTarget {
    /// Base URL of the remote node's Core, e.g. `https://gpu.example.com`.
    pub url: String,
    /// Bearer token for that node, when it requires one.
    #[serde(default)]
    pub token: Option<String>,
}

/// Request body for `POST /api/finetune/merge`.
#[derive(Debug, serde::Deserialize, utoipa::ToSchema)]
pub struct MergeBody {
    /// The adapter's `stem` as listed by `GET /api/finetune/adapters`. Supply this
    /// or `adapter_path`. On a shared node only adapters produced by the caller's
    /// own jobs resolve.
    #[serde(default)]
    pub adapter_name: Option<String>,
    /// The adapter's `path` as listed by `GET /api/finetune/adapters`. Supply this
    /// or `adapter_name`; when both are given they must name the same adapter.
    #[serde(default)]
    pub adapter_path: Option<String>,
    /// Preferred name for the merged `.gguf`. The registered stem is this name
    /// (or the adapter's display name) plus a server-generated suffix, returned
    /// as `stem`; a merge never replaces an existing model.
    #[serde(default)]
    pub output_name: Option<String>,
    /// The base model this adapter was trained on, recorded as provenance on the
    /// resulting installed model.
    #[serde(default)]
    pub base_model_id: Option<String>,
    /// GGUF quantization, e.g. `q4_k_m` (default), `q8_0`, or `f16`.
    #[serde(default)]
    pub quantization_method: Option<String>,
    /// Context length baked into the merged GGUF.
    #[serde(default)]
    pub max_seq_length: Option<u32>,
}
