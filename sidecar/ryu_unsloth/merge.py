"""Merge a trained LoRA adapter into a single GGUF for serving.

llama.cpp serves one merged GGUF — it cannot load a LoRA adapter at serve time —
so this is how a fine-tune becomes a runnable model. Unsloth's
``save_pretrained_gguf`` merges the adapter into the base and converts to GGUF in
one step, preserving the training chat template + EOS (critical: a mismatched
template yields gibberish/runaway generation).

Heavy imports are deferred so the server boots without the training stack.
"""

from __future__ import annotations

import pathlib
import tempfile
import shutil
from typing import Any

from .trainer import (
    has_reparse_component,
    has_reparse_point,
    output_root,
    safe_output_name,
)


def resolve_adapter_dir(req: dict[str, Any]) -> pathlib.Path:
    """Resolve an adapter path and keep it inside the configured output root."""
    root = output_root().resolve()
    raw_path = req.get("adapter_path")
    if raw_path is not None and str(raw_path).strip():
        raw_candidate = pathlib.Path(str(raw_path)).expanduser()
    else:
        adapter_name = safe_output_name(req.get("adapter_name"))
        raw_candidate = root / adapter_name

    if has_reparse_point(raw_candidate) or has_reparse_component(raw_candidate, root):
        raise ValueError("adapter path must not be a symlink or junction")
    candidate = raw_candidate.resolve()

    if candidate == root:
        raise ValueError("adapter path must name a directory inside the output root")
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("adapter path must be inside the configured output root") from exc
    return candidate


def run_merge(req: dict[str, Any]) -> dict[str, Any]:
    """Load an adapter dir, merge + export to ``<output_dir>/<output_name>.gguf``.

    Returns ``{ gguf_path, stem, size_bytes, base_model }`` for Core to register
    as an installed model. The adapter dir carries its base model in
    ``adapter_config.json`` so Unsloth resolves it automatically.
    """
    adapter_dir = resolve_adapter_dir(req)
    if not adapter_dir.is_dir():
        raise ValueError(f"adapter not found: {adapter_dir}")

    from unsloth import FastLanguageModel  # noqa: PLC0415 — order-sensitive, heavy

    quant = str(req.get("quantization_method") or "q4_k_m")
    output_name = safe_output_name(
        req.get("output_name"), fallback=f"{adapter_dir.name}-merged"
    )
    max_seq_length = int(req.get("max_seq_length") or 2048)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    # save_pretrained_gguf writes into a directory; export there then flatten the
    # produced .gguf into Core's flat models layout (~/.ryu/models/<stem>.gguf).
    root = output_root().expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(
        tempfile.mkdtemp(prefix=f".gguf-{output_name}-", dir=str(root))
    )
    try:
        model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=quant)
        ggufs = sorted(tmp.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        if not ggufs:
            raise ValueError("merge produced no .gguf file")
        dest = root / f"{output_name}.gguf"
        if has_reparse_point(dest):
            raise ValueError("merge output must not be a symlink or junction")
        shutil.move(str(ggufs[0]), str(dest))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "gguf_path": str(dest),
        "stem": output_name,
        "size_bytes": dest.stat().st_size,
        "base_model": req.get("base_model_id") or "",
        "quantization_method": quant,
    }
