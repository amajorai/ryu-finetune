"""Regression checks for request-controlled fine-tune filesystem inputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ryu_unsloth.merge import resolve_adapter_dir
from ryu_unsloth.trainer import safe_output_name


def main() -> None:
    for bad in ("../escape", "nested/name", "nested\\name", ".", "..", ""):
        try:
            safe_output_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe output name: {bad!r}")

    root = Path(tempfile.mkdtemp(prefix="ryu-finetune-security-"))
    adapter = root / "adapter"
    adapter.mkdir()
    previous = os.environ.get("RYU_UNSLOTH_OUTPUT_DIR")
    os.environ["RYU_UNSLOTH_OUTPUT_DIR"] = str(root)
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
            os.environ.pop("RYU_UNSLOTH_OUTPUT_DIR", None)
        else:
            os.environ["RYU_UNSLOTH_OUTPUT_DIR"] = previous

    print("SECURITY_OK")


if __name__ == "__main__":
    main()
