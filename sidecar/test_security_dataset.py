import os
import tempfile
from pathlib import Path

from ryu_unsloth.dataset import (
    _read_file,
    render_preference_rows,
    resolve_dataset_path,
)
from ryu_unsloth.trainer import resolve_training_method


def main() -> None:
    previous = os.environ.get("RYU_PLUGIN_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="ryu-unsloth-dataset-") as temporary:
        plugin_data = (Path(temporary) / "plugin-data").resolve()
        root = plugin_data / "datasets"
        root.mkdir(parents=True)
        os.environ["RYU_PLUGIN_DATA_DIR"] = str(plugin_data)
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
        os.environ.pop("RYU_PLUGIN_DATA_DIR", None)
    else:
        os.environ["RYU_PLUGIN_DATA_DIR"] = previous

    pairs = render_preference_rows(
        {
            "format": "preference",
            "samples": [
                {
                    "prompt": "Explain the model catalog.",
                    "chosen": "It finds and installs local models.",
                    "rejected": "It is a calendar.",
                }
            ],
        }
    )
    assert pairs == [
        {
            "prompt": "Explain the model catalog.",
            "chosen": "It finds and installs local models.",
            "rejected": "It is a calendar.",
        }
    ]
    try:
        render_preference_rows(
            {
                "format": "preference",
                "samples": [{"prompt": "Question", "chosen": "Preferred"}],
            }
        )
    except ValueError as error:
        assert "rejected" in str(error)
    else:
        raise AssertionError("preference dataset without a rejected answer was accepted")

    assert resolve_training_method({"dataset": {"format": "chat"}}) == "sft"
    assert (
        resolve_training_method(
            {"method": "dpo", "dataset": {"format": "preference"}}
        )
        == "dpo"
    )
    try:
        resolve_training_method(
            {"method": "dpo", "dataset": {"format": "chat"}}
        )
    except ValueError as error:
        assert "preference" in str(error)
    else:
        raise AssertionError("DPO accepted an SFT dataset")

    print("DATASET_SECURITY_OK")


if __name__ == "__main__":
    main()
