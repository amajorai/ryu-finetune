"""The training loop — Unsloth + TRL ``SFTTrainer``, run on a worker thread.

Everything heavy (unsloth, torch, trl, datasets) is imported *inside* ``run_job``
so the server boots and serves ``/health`` even when training deps are absent —
the job then fails with a clear, actionable error instead of crashing the process.

Import order matters: ``unsloth`` must be imported before transformers/trl so its
kernel patches apply.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import stat
import threading
import time
from typing import Any

from .dataset import render_preference_rows, render_texts
from .jobs import Job
from .paths import adapters_root, checkpoints_root, make_private_dir, restrict_tree

_DEFAULTS: dict[str, Any] = {
    "method": "sft",
    "max_seq_length": 2048,
    "max_prompt_length": 512,
    "beta": 0.1,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "epochs": 1.0,
    "max_steps": 0,  # 0 => use epochs
    "learning_rate": 2e-4,
    "batch_size": 2,
    "grad_accum": 4,
    "warmup_steps": 5,
    "logging_steps": 1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
}


def _cfg(config: dict[str, Any], key: str) -> Any:
    val = config.get(key)
    return _DEFAULTS[key] if val is None else val


def resolve_training_method(config: dict[str, Any]) -> str:
    """Validate the trainer/dataset pairing before importing GPU dependencies."""
    method = str(config.get("method") or "sft").strip().lower()
    if method not in {"sft", "dpo"}:
        raise ValueError("training method must be 'sft' or 'dpo'")
    dataset = config.get("dataset")
    dataset_format = (
        str(dataset.get("format", "chat")).strip().lower()
        if isinstance(dataset, dict)
        else ""
    )
    if method == "dpo" and dataset_format != "preference":
        raise ValueError("DPO training requires preference rows")
    if method == "sft" and dataset_format == "preference":
        raise ValueError("preference rows require DPO training")
    return method


_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_output_name(value: Any, *, fallback: str | None = None) -> str:
    """Return a filename-safe output stem, or reject a path-like value."""
    name = str(value).strip() if value is not None else ""
    if not name:
        name = fallback or ""
    if not _OUTPUT_NAME_RE.fullmatch(name):
        raise ValueError(
            "output_name must be a simple filename stem using letters, numbers, "
            "dots, hyphens, or underscores"
        )
    return name


def output_root() -> pathlib.Path:
    """Where adapters are written: ``<data dir>/models/adapters`` (see ``paths``)."""
    return adapters_root()


def checkpoint_dir(job_id: str) -> pathlib.Path:
    """The trainer's scratch checkpoints for one job, under the app's data dir."""
    return checkpoints_root().expanduser().resolve() / safe_output_name(job_id)


def has_reparse_point(path: pathlib.Path) -> bool:
    """Return whether a path is a symlink/junction-like filesystem reparse point."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & reparse_flag
    )


def has_reparse_component(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether an existing component between root and path is reparse-backed."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if has_reparse_point(current):
            return True
    return False


_ARTIFACT_ID_RE = re.compile(r"^ft-[0-9a-f]{32}$")


def artifact_dir_name(config: dict[str, Any], job_id: str) -> str:
    """The adapter directory name for one job: never the caller's ``output_name``.

    Core allocates a unique ``artifact_id`` per job; a worker driven without one
    falls back to the job's own unique id. ``output_name`` is display metadata
    only, so two jobs that share a human-readable name never share a directory.
    """
    artifact_id = str(config.get("artifact_id") or "").strip()
    if artifact_id:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ValueError("artifact_id must be a server-generated id")
        return artifact_id
    return safe_output_name(f"adapter-{job_id}")


def safe_output_dir(output_name: str) -> pathlib.Path:
    """Create a NEW real output directory under the canonical output root.

    Creation is exclusive: an existing directory (another job's adapter) is never
    reused, so a later job cannot write into or replace an earlier job's files.
    """
    root = make_private_dir(output_root().expanduser().resolve())
    target = root / safe_output_name(output_name)
    if has_reparse_point(target):
        raise ValueError("output directory must not be a symlink or junction")
    try:
        target.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("output directory already exists; refusing to overwrite it") from exc
    make_private_dir(target)
    if has_reparse_point(target) or target.resolve() != target:
        raise ValueError("output directory escaped the configured output root")
    return target


# One GPU, one training run: concurrent submissions queue here (state stays
# `queued`) instead of loading several base models at once and OOMing the device.
_TRAIN_LOCK = threading.Lock()


def run_job(job: Job) -> None:
    """Thread target: execute one fine-tune job end to end. The job's checkpoint
    folder is removed however it ends — only the saved adapter is kept."""
    with _TRAIN_LOCK:
        # A job cancelled while it waited must not load a model just to stop.
        if job.cancel_requested():
            job.mark_cancelled()
            return
        try:
            _run(job)
        except Exception as exc:  # noqa: BLE001 — any failure becomes a job error
            job.mark_failed(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                shutil.rmtree(checkpoint_dir(job.id), ignore_errors=True)
            except (RuntimeError, ValueError):
                pass


def _run(job: Job) -> None:
    config = job.config
    job.mark_running()

    method = resolve_training_method(config)

    base_model_id = config["base_model_id"]
    max_seq_length = _cfg(config, "max_seq_length")
    job.emit("log", message=f"loading base model for {method.upper()}")

    # --- heavy imports (deferred) -----------------------------------------
    from unsloth import FastLanguageModel  # noqa: PLC0415 — order-sensitive
    from datasets import Dataset  # noqa: PLC0415
    from transformers import TrainerCallback  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: bf16 where supported, else fp16
        load_in_4bit=bool(_cfg(config, "load_in_4bit")),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(_cfg(config, "lora_r")),
        target_modules=_cfg(config, "target_modules"),
        lora_alpha=int(_cfg(config, "lora_alpha")),
        lora_dropout=float(_cfg(config, "lora_dropout")),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(_cfg(config, "seed")),
    )

    job.emit("log", message="preparing dataset")
    if method == "dpo":
        rows = render_preference_rows(config["dataset"])
        dataset = Dataset.from_list(rows)
    else:
        texts = render_texts(config["dataset"], tokenizer)
        dataset = Dataset.from_dict({"text": texts})
        rows = texts
    example_count = len(rows)

    # --- progress + cancellation callback ---------------------------------
    class _Progress(TrainerCallback):
        def __init__(self) -> None:
            self.t0 = time.time()

        def on_train_begin(self, args, state, control, **kw):  # noqa: ANN001
            job.emit(
                "log",
                message=f"{method.upper()} training started · {example_count} rows",
            )
            return control

        def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
            if job.cancel_requested():
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
            logs = logs or {}
            step = int(state.global_step)
            total = int(state.max_steps) or 0
            elapsed = time.time() - self.t0
            eta = (elapsed / step) * (total - step) if step and total else None
            job.update_progress(
                step=step,
                max_steps=total,
                loss=logs.get("loss"),
                eta_secs=eta,
            )
            return control

    # Checkpoints hold full optimizer/adapter state: an owner-only folder (the
    # root too), deleted when the job ends and swept at the next start.
    make_private_dir(checkpoints_root().expanduser().resolve())
    make_private_dir(checkpoint_dir(job.id))
    shared_training_args = {
        "per_device_train_batch_size": int(_cfg(config, "batch_size")),
        "gradient_accumulation_steps": int(_cfg(config, "grad_accum")),
        "warmup_steps": int(_cfg(config, "warmup_steps")),
        "num_train_epochs": float(_cfg(config, "epochs")),
        "max_steps": int(_cfg(config, "max_steps")) or -1,
        "learning_rate": float(_cfg(config, "learning_rate")),
        "logging_steps": int(_cfg(config, "logging_steps")),
        "optim": "adamw_8bit",
        "weight_decay": float(_cfg(config, "weight_decay")),
        "lr_scheduler_type": str(_cfg(config, "lr_scheduler_type")),
        "seed": int(_cfg(config, "seed")),
        "output_dir": str(checkpoint_dir(job.id)),
        "report_to": "none",
    }

    if method == "dpo":
        # DPO needs a padding token to collate chosen/rejected completions. Use the
        # model's EOS token when its tokenizer has no dedicated pad token.
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        training_args = DPOConfig(
            **shared_training_args,
            beta=float(_cfg(config, "beta")),
            max_length=int(max_seq_length),
            max_prompt_length=min(
                int(_cfg(config, "max_prompt_length")), int(max_seq_length)
            ),
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
            callbacks=[_Progress()],
        )
    else:
        training_args = SFTConfig(
            **shared_training_args,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[_Progress()],
        )

    trainer.train()

    if job.cancel_requested():
        job.mark_cancelled()
        return

    # --- save the LoRA adapter --------------------------------------------
    out_dir = safe_output_dir(artifact_dir_name(config, job.id))
    job.emit("log", message=f"saving adapter to {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    restrict_tree(out_dir)  # adapter weights stay owner-only (not encrypted)
    job.mark_done(output_dir=str(out_dir))
