import os
import tempfile
from pathlib import Path

from ryu_unsloth.merge import resolve_adapter_dir
from ryu_unsloth.trainer import safe_output_dir


def main() -> None:
    previous = os.environ.get("RYU_MODELS_DIR")
    with tempfile.TemporaryDirectory(prefix="ryu-unsloth-security-") as temporary:
        os.environ["RYU_MODELS_DIR"] = str(Path(temporary).resolve())
        root = Path(temporary).resolve() / "adapters"
        safe_output_dir("adapter")

        real = root / "real"
        real.mkdir()
        link = root / "linked"
        try:
            link.symlink_to(real, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows developer mode may disable symlink creation; lexical path
            # checks are still exercised by test_security.py.
            pass
        else:
            try:
                safe_output_dir("linked")
            except ValueError:
                pass
            else:
                raise AssertionError("safe output creation followed a symlink")

            try:
                resolve_adapter_dir({"adapter_path": str(link)})
            except ValueError:
                pass
            else:
                raise AssertionError("adapter resolution accepted a reparse point")

    if previous is None:
        os.environ.pop("RYU_MODELS_DIR", None)
    else:
        os.environ["RYU_MODELS_DIR"] = previous
    print("REPARSE_SECURITY_OK")


if __name__ == "__main__":
    main()
