"""Where the worker reads and writes on disk.

Core spawns this worker with the data-dir layout already resolved:

  - ``RYU_MODELS_DIR``       the shared ``<data dir>/models`` root
  - ``RYU_PLUGIN_DATA_DIR``  this app's ``<data dir>/plugin-data/@ryu+finetune``

and every artifact kind lives at a fixed place below them:

  - LoRA adapters        ``models/adapters/<artifact_id>/``
  - merged GGUFs         ``models/chat/<name>.gguf``
  - file-backed datasets ``plugin-data/@ryu+finetune/datasets``
  - per-job checkpoints  ``plugin-data/@ryu+finetune/checkpoints/<job>``

Both roots fall back to ``RYU_DIR``, so a worker started by hand with only
``RYU_DIR`` set still writes into the Ryu data-dir layout.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import stat
import sys

PLUGIN_DIR_NAME = "@ryu+finetune"


def _env_path(name: str) -> pathlib.Path | None:
    value = (os.environ.get(name) or "").strip()
    return pathlib.Path(value).expanduser() if value else None


def _ryu_dir() -> pathlib.Path:
    root = _env_path("RYU_DIR")
    if root is None:
        raise RuntimeError(
            "RYU_DIR / RYU_MODELS_DIR / RYU_PLUGIN_DATA_DIR are not set; the worker "
            "is spawned by Ryu Core, which injects them"
        )
    return root


def models_root() -> pathlib.Path:
    """The shared ``<data dir>/models`` root."""
    return _env_path("RYU_MODELS_DIR") or _ryu_dir() / "models"


def plugin_data_dir() -> pathlib.Path:
    """This app's own ``<data dir>/plugin-data/@ryu+finetune``."""
    return _env_path("RYU_PLUGIN_DATA_DIR") or _ryu_dir() / "plugin-data" / PLUGIN_DIR_NAME


def adapters_root() -> pathlib.Path:
    """Where trained LoRA adapters are saved, one new directory per job."""
    return models_root() / "adapters"


def merged_models_root() -> pathlib.Path:
    """Where merged GGUFs land — the chat-model folder the catalog scans."""
    return models_root() / "chat"


def datasets_root() -> pathlib.Path:
    """The only directory file-backed datasets may be read from."""
    return plugin_data_dir() / "datasets"


def checkpoints_root() -> pathlib.Path:
    """Per-job trainer checkpoints; each job's folder is deleted when it ends."""
    return plugin_data_dir() / "checkpoints"


def work_root() -> pathlib.Path:
    """Scratch space for the GGUF export before it is moved into place."""
    return plugin_data_dir() / "tmp"


# ── owner-only storage ───────────────────────────────────────────────────────
#
# Adapters, merged GGUFs, trainer checkpoints and the GGUF scratch are model
# weights that the training stack and llama.cpp read by path (and mmap), so they
# are not encrypted. They are kept owner-only instead: directories 0700, files
# 0600. On Windows the per-user data dir's ACL governs access and these calls
# are skipped.

GGUF_SCRATCH_PREFIX = "gguf-"
_PRIVATE = os.name != "nt"


def make_private_dir(path: pathlib.Path) -> pathlib.Path:
    """Create ``path`` (and parents) and make it owner-only (0700)."""
    path.mkdir(parents=True, exist_ok=True)
    if _PRIVATE and not path.is_symlink():
        os.chmod(path, 0o700)
    return path


def make_private_file(path: pathlib.Path) -> None:
    """Make one regular file owner-only (0600); symlinks are left alone."""
    if _PRIVATE and path.is_file() and not path.is_symlink():
        os.chmod(path, 0o600)


def restrict_tree(root: pathlib.Path) -> None:
    """Make ``root`` and everything under it owner-only, without following links."""
    if not _PRIVATE or root.is_symlink() or not root.is_dir():
        return
    os.chmod(root, 0o700)
    for current, dirs, files in os.walk(root, followlinks=False):
        base = pathlib.Path(current)
        for name in dirs:
            child = base / name
            if not child.is_symlink():
                os.chmod(child, 0o700)
        for name in files:
            make_private_file(base / name)


def _remove_tree(path: pathlib.Path) -> bool:
    try:
        if path.is_symlink() or not path.is_dir():
            return False
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode):
            return False
        shutil.rmtree(path)
        return True
    except OSError as error:
        print(f"[ryu-finetune] could not remove {path.name}: {error}", file=sys.stderr)
        return False


def sweep_orphans() -> dict[str, int]:
    """Delete trainer checkpoints and GGUF scratch left behind by a crash.

    Jobs live only in memory, so at startup no job owns a checkpoint folder and
    no merge owns a ``gguf-*`` scratch folder: each one found is an orphan of a
    previous run (killed process, power loss). Symlinks are never followed.
    """
    removed = {"checkpoints": 0, "gguf_scratch": 0}
    checkpoints = checkpoints_root().expanduser()
    if checkpoints.is_dir() and not checkpoints.is_symlink():
        for child in checkpoints.iterdir():
            if _remove_tree(child):
                removed["checkpoints"] += 1
    scratch = work_root().expanduser()
    if scratch.is_dir() and not scratch.is_symlink():
        for child in scratch.glob(f"{GGUF_SCRATCH_PREFIX}*"):
            if _remove_tree(child):
                removed["gguf_scratch"] += 1
    return removed


def prepare_private_storage() -> dict[str, int]:
    """Startup: sweep orphans, then make the app's folders owner-only.

    The shared ``models/chat`` folder is not narrowed (other models live there);
    each merged GGUF is made 0600 when it is written. User-placed datasets are
    read as-is (plaintext); their folder is made owner-only.
    """
    removed = sweep_orphans()
    for root in (plugin_data_dir(), checkpoints_root(), work_root(), datasets_root()):
        make_private_dir(root.expanduser())
    adapters = adapters_root().expanduser()
    make_private_dir(adapters)
    restrict_tree(adapters)
    return removed
