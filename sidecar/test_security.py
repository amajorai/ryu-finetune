"""Regression checks for request-controlled fine-tune filesystem inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ryu_unsloth import paths
from ryu_unsloth.merge import resolve_adapter_dir
from ryu_unsloth.trainer import safe_output_dir, safe_output_name

_STORAGE_ENV = ("RYU_PLUGIN_DATA_DIR", "RYU_MODELS_DIR")


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def private_storage_sweeps_orphans_and_is_owner_only() -> None:
    """Startup drops crash leftovers and narrows the weight folders (#59)."""
    root = Path(tempfile.mkdtemp(prefix="ryu-finetune-private-"))
    plugin = root / "plugin-data" / "@ryu+finetune"
    models = root / "models"
    saved = {name: os.environ.get(name) for name in _STORAGE_ENV}
    for name in _STORAGE_ENV:
        os.environ.pop(name, None)
    os.environ["RYU_PLUGIN_DATA_DIR"] = str(plugin)
    os.environ["RYU_MODELS_DIR"] = str(models)
    try:
        orphan = plugin / "checkpoints" / "job-1" / "checkpoint-10"
        orphan.mkdir(parents=True)
        (orphan / "optimizer.pt").write_bytes(b"state")
        scratch = plugin / "tmp" / "gguf-model-abc"
        scratch.mkdir(parents=True)
        (scratch / "model.gguf").write_bytes(b"GGUF")
        keep = plugin / "tmp" / "other"
        keep.mkdir()
        outside = root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("x")
        (plugin / "checkpoints" / "link").symlink_to(outside, target_is_directory=True)
        adapter = models / "adapters" / "mine"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"w")
        os.chmod(adapter / "adapter_model.safetensors", 0o644)
        os.chmod(adapter, 0o755)

        removed = paths.prepare_private_storage()

        assert removed == {"checkpoints": 1, "gguf_scratch": 1}, removed
        assert not orphan.parent.exists()
        assert not scratch.exists()
        assert keep.is_dir()
        assert (outside / "keep.txt").exists(), "a symlinked folder is never followed"
        if os.name != "nt":
            for folder in (plugin, plugin / "checkpoints", plugin / "tmp", plugin / "datasets",
                           models / "adapters", adapter):
                assert _mode(folder) == 0o700, (folder, oct(_mode(folder)))
            assert _mode(adapter / "adapter_model.safetensors") == 0o600
        # A second start finds nothing to sweep.
        assert paths.prepare_private_storage() == {"checkpoints": 0, "gguf_scratch": 0}

        target = safe_output_dir("fresh")
        if os.name != "nt":
            assert _mode(target) == 0o700
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_private_storage_sweeps_orphans_and_is_owner_only() -> None:
    private_storage_sweeps_orphans_and_is_owner_only()


def test_request_controlled_paths_stay_inside_their_roots() -> None:
    main()


def main() -> None:
    for bad in ("../escape", "nested/name", "nested\\name", ".", "..", ""):
        try:
            safe_output_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe output name: {bad!r}")

    root = Path(tempfile.mkdtemp(prefix="ryu-finetune-security-"))
    adapter = root / "adapters" / "adapter"
    adapter.mkdir(parents=True)
    previous = os.environ.get("RYU_MODELS_DIR")
    os.environ["RYU_MODELS_DIR"] = str(root)
    try:
        assert resolve_adapter_dir({"adapter_name": "adapter"}) == adapter.resolve()
        for bad in (root.parent, root / ".." / "escape", Path(tempfile.gettempdir())):
            try:
                resolve_adapter_dir({"adapter_path": str(bad)})
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted adapter outside output root: {bad}")
    finally:
        if previous is None:
            os.environ.pop("RYU_MODELS_DIR", None)
        else:
            os.environ["RYU_MODELS_DIR"] = previous

    private_storage_sweeps_orphans_and_is_owner_only()
    print("SECURITY_OK")


if __name__ == "__main__":
    main()
