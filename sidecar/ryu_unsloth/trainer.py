"""The training loop — Unsloth + TRL ``SFTTrainer``, run on a worker thread.

Everything heavy (unsloth, torch, trl, datasets) is imported *inside* ``run_job``
so the server boots and serves ``/health`` even when training deps are absent —
the job then fails with a clear, actionable error instead of crashing the process.

Import order matters: ``unsloth`` must be imported before transformers/trl so its
kernel patches apply.
"""

from __future__ import annotations

import os
import pathlib
import re
import stat
import time
from typing import Any

from .dataset import render_texts
from .jobs import Job

_DEFAULTS: dict[str, Any] = {
    "max_seq_length": 2048,
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
    """Where adapters are written. Core overrides via RYU_UNSLOTH_OUTPUT_DIR so the
    artifacts land under ~/.ryu/models; default is a local ./outputs for dev."""
    root = os.environ.get("RYU_UNSLOTH_OUTPUT_DIR") or str(
        pathlib.Path.cwd() / "outputs"
    )
    return pathlib.Path(root)


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


def safe_output_dir(output_name: str) -> pathlib.Path:
    """Create or reuse a real output directory under the canonical output root."""
    root = output_root().expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / safe_output_name(output_name)
    if has_reparse_point(target):
        raise ValueError("output directory must not be a symlink or junction")
    target.mkdir(parents=True, exist_ok=True)
    if has_reparse_point(target) or target.resolve() != target:
        raise ValueError("output directory escaped the configured output root")
    return target


def run_job(job: Job) -> None:
    """Thread target: execute one fine-tune job end to end."""
    try:
        _run(job)
    except Exception as exc:  # noqa: BLE001 — any failure becomes a job error
        job.mark_failed(f"{type(exc).__name__}: {exc}")


def _run(job: Job) -> None:
    config = job.config
    job.mark_running()
    job.emit("log", message="loading base model")

    # --- heavy imports (deferred) -----------------------------------------
    from unsloth import FastLanguageModel  # noqa: PLC0415 — order-sensitive
    from datasets import Dataset  # noqa: PLC0415
    from transformers import TrainerCallback  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    base_model_id = config["base_model_id"]
    max_seq_length = _cfg(config, "max_seq_length")

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
    texts = render_texts(config["dataset"], tokenizer)
    dataset = Dataset.from_dict({"text": texts})

    # --- progress + cancellation callback ---------------------------------
    class _Progress(TrainerCallback):
        def __init__(self) -> None:
            self.t0 = time.time()

        def on_train_begin(self, args, state, control, **kw):  # noqa: ANN001
            job.emit("log", message=f"training started · {len(texts)} rows")
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

    sft_args = SFTConfig(
        per_device_train_batch_size=int(_cfg(config, "batch_size")),
        gradient_accumulation_steps=int(_cfg(config, "grad_accum")),
        warmup_steps=int(_cfg(config, "warmup_steps")),
        num_train_epochs=float(_cfg(config, "epochs")),
        max_steps=int(_cfg(config, "max_steps")) or -1,
        learning_rate=float(_cfg(config, "learning_rate")),
        logging_steps=int(_cfg(config, "logging_steps")),
        optim="adamw_8bit",
        weight_decay=float(_cfg(config, "weight_decay")),
        lr_scheduler_type=str(_cfg(config, "lr_scheduler_type")),
        seed=int(_cfg(config, "seed")),
        output_dir=str(output_root() / f".checkpoints-{job.id}"),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
        callbacks=[_Progress()],
    )

    trainer.train()

    if job.cancel_requested():
        job.mark_cancelled()
        return

    # --- save the LoRA adapter --------------------------------------------
    output_name = safe_output_name(
        config.get("output_name"), fallback=f"adapter-{job.id[:8]}"
    )
    out_dir = safe_output_dir(output_name)
    job.emit("log", message=f"saving adapter to {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    job.mark_done(output_dir=str(out_dir))
