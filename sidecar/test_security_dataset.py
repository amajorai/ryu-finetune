import os
import tempfile
from pathlib import Path

from ryu_unsloth.dataset import _read_file, resolve_dataset_path


def main() -> None:
    previous = os.environ.get("RYU_UNSLOTH_OUTPUT_DIR")
    with tempfile.TemporaryDirectory(prefix="ryu-unsloth-dataset-") as temporary:
        root = (Path(temporary) / "root").resolve()
        root.mkdir()
        os.environ["RYU_UNSLOTH_OUTPUT_DIR"] = str(root)
        inside = root / "inside.json"
        inside.write_text('[{"text": "hello"}]', encoding="utf-8")
        assert resolve_dataset_path("inside.json") == inside
        assert _read_file("inside.json") == [{"text": "hello"}]

        outside = root.parent / "outside.json"
        outside.write_text('[{"text": "secret"}]', encoding="utf-8")
        try:
            resolve_dataset_path(str(outside))
        except ValueError:
            pass
        else:
            raise AssertionError("dataset resolver accepted an outside absolute path")

        try:
            resolve_dataset_path("../outside.json")
        except ValueError:
            pass
        else:
            raise AssertionError("dataset resolver accepted parent traversal")

        link = root / "linked.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pass
        else:
            try:
                resolve_dataset_path("linked.json")
            except ValueError:
                pass
            else:
                raise AssertionError("dataset resolver followed an outside symlink")

    if previous is None:
        os.environ.pop("RYU_UNSLOTH_OUTPUT_DIR", None)
    else:
        os.environ["RYU_UNSLOTH_OUTPUT_DIR"] = previous
    print("DATASET_SECURITY_OK")


if __name__ == "__main__":
    main()
