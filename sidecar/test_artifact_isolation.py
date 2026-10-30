"""Regression checks: two jobs that share an output_name never share an adapter dir.

The audit case: on a shared node two tenants submit jobs with the same
``output_name``. The worker used to resolve both to ``adapters/<output_name>`` and
reuse the existing directory, so the later job overwrote the earlier tenant's
adapter. The directory is now the server-generated ``artifact_id`` (or the job's
own id), and creation is exclusive.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ryu_unsloth.merge import install_merged, merged_dest
from ryu_unsloth.trainer import artifact_dir_name, safe_output_dir


def same_output_name_resolves_to_distinct_dirs() -> None:
    alice = {"output_name": "shared", "artifact_id": "ft-" + "a" * 32}
    bob = {"output_name": "shared", "artifact_id": "ft-" + "b" * 32}
    first = safe_output_dir(artifact_dir_name(alice, "job-alice"))
    second = safe_output_dir(artifact_dir_name(bob, "job-bob"))
    assert first != second, (first, second)
    assert first.name == alice["artifact_id"], first
    assert "shared" not in (first.name, second.name)


def an_existing_artifact_dir_is_never_reused() -> None:
    config = {"artifact_id": "ft-" + "c" * 32}
    safe_output_dir(artifact_dir_name(config, "job-one"))
    try:
        safe_output_dir(artifact_dir_name(config, "job-two"))
    except ValueError:
        pass
    else:
        raise AssertionError("a second job reused an existing adapter directory")


def without_an_artifact_id_the_job_id_names_the_dir() -> None:
    name = artifact_dir_name({"output_name": "shared"}, "0123abcd" * 4)
    assert name == "adapter-" + "0123abcd" * 4, name


def a_caller_chosen_artifact_id_is_rejected() -> None:
    for bad in ("shared", "../escape", "ft-XYZ", "ft-" + "a" * 31):
        try:
            artifact_dir_name({"artifact_id": bad}, "job")
        except ValueError:
            continue
        raise AssertionError(f"accepted a non-server artifact id: {bad!r}")


def _export(scratch: Path, body: bytes) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    source = scratch / f"export-{len(list(scratch.iterdir()))}.gguf"
    source.write_bytes(body)
    return source


def a_merge_never_overwrites_an_existing_model(scratch: Path) -> None:
    """Two merges with the same output name: the second is refused, the first kept."""
    dest = merged_dest("shared-gguf")
    install_merged(_export(scratch, b"alice"), dest)
    assert dest.read_bytes() == b"alice"
    # The early check refuses a taken name before any heavy work.
    try:
        merged_dest("shared-gguf")
    except ValueError:
        pass
    else:
        raise AssertionError("merged_dest accepted a name that already exists")
    # Even when the early check raced (both merges passed it), the install is
    # exclusive and leaves the first model untouched.
    try:
        install_merged(_export(scratch, b"bob"), dest)
    except ValueError:
        pass
    else:
        raise AssertionError("a second merge replaced an existing model")
    assert dest.read_bytes() == b"alice"
    leftovers = [p.name for p in dest.parent.iterdir() if p.name != dest.name]
    assert leftovers == [], leftovers


def a_merge_refuses_a_symlink_at_the_destination(scratch: Path) -> None:
    root = merged_dest("probe").parent
    victim = root / "victim.bin"
    victim.write_bytes(b"victim")
    link = root / "linked.gguf"
    link.symlink_to(victim)
    attempts = (
        lambda: merged_dest("linked"),
        lambda: install_merged(_export(scratch, b"x"), link),
    )
    for attempt in attempts:
        try:
            attempt()
        except ValueError:
            continue
        raise AssertionError("merge wrote through a symlinked destination")
    assert victim.read_bytes() == b"victim"


def main() -> None:
    previous = os.environ.get("RYU_MODELS_DIR")
    with tempfile.TemporaryDirectory(prefix="ryu-unsloth-artifacts-") as temporary:
        # Adapters land in <models>/adapters and merged GGUFs in <models>/chat.
        os.environ["RYU_MODELS_DIR"] = str(Path(temporary).resolve())
        try:
            same_output_name_resolves_to_distinct_dirs()
            an_existing_artifact_dir_is_never_reused()
            without_an_artifact_id_the_job_id_names_the_dir()
            a_caller_chosen_artifact_id_is_rejected()
            a_merge_never_overwrites_an_existing_model(Path(temporary).resolve() / "scratch")
            a_merge_refuses_a_symlink_at_the_destination(Path(temporary).resolve() / "scratch")
        finally:
            if previous is None:
                os.environ.pop("RYU_MODELS_DIR", None)
            else:
                os.environ["RYU_MODELS_DIR"] = previous
    print("ARTIFACT_ISOLATION_OK")


if __name__ == "__main__":
    main()
