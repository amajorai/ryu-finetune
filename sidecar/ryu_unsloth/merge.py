"""Merge a trained LoRA adapter into a single GGUF for serving.

llama.cpp serves one merged GGUF — it cannot load a LoRA adapter at serve time —
so this is how a fine-tune becomes a runnable model. Unsloth's
``save_pretrained_gguf`` merges the adapter into the base and converts to GGUF in
one step, preserving the training chat template + EOS (critical: a mismatched
template yields gibberish/runaway generation).

Heavy imports are deferred so the server boots without the training stack.
"""

from __future__ import annotations

import os
import pathlib
import secrets
import tempfile
import shutil
from typing import Any

from .paths import (
    GGUF_SCRATCH_PREFIX,
    make_private_dir,
    make_private_file,
    merged_models_root,
    work_root,
)
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


_EXISTS_MESSAGE = "a model with this output name already exists; refusing to overwrite it"


def merged_dest(output_name: str) -> pathlib.Path:
    """``models/chat/<output_name>.gguf``, refused when anything already exists there.

    This is the early (fail-fast) check; :func:`install_merged` repeats it
    atomically when the GGUF is put in place.
    """
    root = merged_models_root().expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{safe_output_name(output_name)}.gguf"
    if has_reparse_point(dest):
        raise ValueError("merge output must not be a symlink or junction")
    if os.path.lexists(dest):
        raise ValueError(_EXISTS_MESSAGE)
    return dest


def install_merged(source: pathlib.Path, dest: pathlib.Path) -> None:
    """Put an exported GGUF at ``dest`` without ever replacing an existing file.

    The GGUF is first staged beside ``dest`` under a name the catalog never lists,
    then published with a hard link, which fails atomically when ``dest`` exists
    (another tenant's merge, or any installed chat GGUF with the same stem). On a
    filesystem without hard links it falls back to an exclusive ``O_EXCL`` create.
    """
    staged = dest.with_name(f".{dest.name}.partial-{secrets.token_hex(8)}")
    try:
        try:
            os.replace(source, staged)
        except OSError:
            # Scratch and the models folder can sit on different filesystems.
            shutil.copyfile(source, staged)
        try:
            os.link(staged, dest, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError(_EXISTS_MESSAGE) from exc
        except OSError:
            _install_exclusive_without_links(staged, dest)
    finally:
        staged.unlink(missing_ok=True)


def _install_exclusive_without_links(staged: pathlib.Path, dest: pathlib.Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(dest, flags, 0o644)
    except FileExistsError as exc:
        raise ValueError(_EXISTS_MESSAGE) from exc
    os.close(fd)
    os.replace(staged, dest)


def run_merge(req: dict[str, Any]) -> dict[str, Any]:
    """Load an adapter dir, merge + export to ``models/chat/<output_name>.gguf``.

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
    # Fail before the heavy load when the name is already taken.
    dest = merged_dest(output_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    # save_pretrained_gguf writes into a directory; export into the app's scratch
    # dir, then move the produced .gguf into the chat-model folder the catalog
    # scans (<data dir>/models/chat/<stem>.gguf). The install is exclusive, so a
    # merge never replaces an existing model.
    # Scratch is owner-only (mkdtemp is 0700) and swept at startup if a crash
    # leaves it behind.
    scratch = make_private_dir(work_root().expanduser().resolve())
    tmp = pathlib.Path(
        tempfile.mkdtemp(prefix=f"{GGUF_SCRATCH_PREFIX}{output_name}-", dir=str(scratch))
    )
    try:
        model.save_pretrained_gguf(str(tmp), tokenizer, quantization_method=quant)
        ggufs = sorted(tmp.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
        if not ggufs:
            raise ValueError("merge produced no .gguf file")
        # 0600 before it becomes visible in the shared chat-model folder.
        make_private_file(ggufs[0])
        install_merged(ggufs[0], dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "gguf_path": str(dest),
        "stem": output_name,
        "size_bytes": dest.stat().st_size,
        "base_model": req.get("base_model_id") or "",
        "quantization_method": quant,
    }
