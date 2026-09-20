"""Dataset normalization — turn the request's dataset into rendered text rows.

We accept three shapes so callers/UI can stay simple, and render each to a single
``text`` field that ``SFTTrainer`` trains on:

  - ``chat``    : {"format":"chat", "samples":[{"messages":[{role,content}, ...]}]}
                  rendered via the tokenizer's chat template (preserves EOS).
  - ``alpaca``  : {"format":"alpaca", "samples":[{instruction,input?,output}]}
  - ``text``    : {"format":"text", "samples":[{"text":"..."}]}  (passthrough)

A ``path`` to a .json/.jsonl file with the same row shapes is also accepted.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Optional

MAX_DATASET_FILE_BYTES = 64 * 1024 * 1024
MAX_DATASET_ROWS = 100_000
MAX_RENDERED_TEXT_CHARS = 1_000_000
MAX_RENDERED_TOTAL_CHARS = 64 * 1024 * 1024

_ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
_ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def dataset_root() -> pathlib.Path:
    """Return the canonical directory allowed for file-backed datasets."""
    configured = os.environ.get("RYU_UNSLOTH_OUTPUT_DIR")
    root = pathlib.Path(configured) if configured else pathlib.Path.cwd() / "outputs"
    return root.expanduser().resolve()


def resolve_dataset_path(path: str) -> pathlib.Path:
    """Resolve a dataset path while keeping it under the configured root."""
    root = dataset_root()
    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset path must be inside the configured dataset root") from exc
    return candidate


def _load_rows(dataset: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    fmt = str(dataset.get("format", "chat")).lower()
    path = dataset.get("path")
    if path:
        rows = _read_file(str(path))
    else:
        rows = list(dataset.get("samples") or [])
    if not rows:
        raise ValueError("dataset has no samples")
    if len(rows) > MAX_DATASET_ROWS:
        raise ValueError(f"dataset contains more than {MAX_DATASET_ROWS} rows")
    return fmt, rows


def _read_file(path: str) -> list[dict[str, Any]]:
    p = resolve_dataset_path(path)
    if not p.exists():
        raise ValueError(f"dataset path not found: {path}")
    if p.stat().st_size > MAX_DATASET_FILE_BYTES:
        raise ValueError(
            f"dataset file exceeds the {MAX_DATASET_FILE_BYTES} byte limit"
        )
    raw = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) > MAX_DATASET_ROWS:
            raise ValueError(f"dataset contains more than {MAX_DATASET_ROWS} rows")
        return rows
    data = json.loads(raw)
    if isinstance(data, dict) and "samples" in data:
        rows = list(data["samples"])
        if len(rows) > MAX_DATASET_ROWS:
            raise ValueError(f"dataset contains more than {MAX_DATASET_ROWS} rows")
        return rows
    if isinstance(data, list):
        if len(data) > MAX_DATASET_ROWS:
            raise ValueError(f"dataset contains more than {MAX_DATASET_ROWS} rows")
        return data
    raise ValueError("json dataset must be a list or {samples:[...]}")


def render_texts(dataset: dict[str, Any], tokenizer: Optional[Any]) -> list[str]:
    """Render every row to a training string, appending EOS where we control it."""
    fmt, rows = _load_rows(dataset)
    eos = getattr(tokenizer, "eos_token", "") or "" if tokenizer else ""
    texts: list[str] = []
    rendered_chars = 0

    def append_rendered(value: Any) -> None:
        nonlocal rendered_chars
        text = value if isinstance(value, str) else str(value)
        if len(text) > MAX_RENDERED_TEXT_CHARS:
            raise ValueError(
                f"a rendered dataset row exceeds the {MAX_RENDERED_TEXT_CHARS} character limit"
            )
        rendered_chars += len(text)
        if rendered_chars > MAX_RENDERED_TOTAL_CHARS:
            raise ValueError(
                f"rendered dataset exceeds the {MAX_RENDERED_TOTAL_CHARS} character limit"
            )
        texts.append(text)

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every dataset row must be an object")
        if fmt == "text":
            append_rendered(row["text"])
        elif fmt == "alpaca":
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            inp = str(row.get("input", "")).strip()
            tmpl = _ALPACA_WITH_INPUT if inp else _ALPACA_NO_INPUT
            append_rendered(
                tmpl.format(instruction=instruction, input=inp, output=output) + eos
            )
        elif fmt == "chat":
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("chat rows must have a `messages` array")
            if len(messages) > 256:
                raise ValueError("chat rows may contain at most 256 messages")
            for message in messages:
                if not isinstance(message, dict) or len(str(message.get("content", ""))) > MAX_RENDERED_TEXT_CHARS:
                    raise ValueError("chat message content is too large")
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                append_rendered(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
            else:
                # Fallback rendering when no tokenizer template is available.
                joined = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )
                append_rendered(joined + eos)
        else:
            raise ValueError(f"unknown dataset format '{fmt}'")

    if not texts:
        raise ValueError("dataset rendered to zero training rows")
    return texts
