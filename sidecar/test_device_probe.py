"""Regression checks for the hardware probe's stable ``reason_code`` values.

The backend turns each code into a whole translatable sentence, so every
not-ready branch must report a code, and a failed probe must keep the raw
exception text out of ``reason`` (it belongs in the log).
"""

from __future__ import annotations

import sys
import types
from typing import Any

from ryu_unsloth import device


class _Props:
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor
        self.name = "Test GPU"
        self.total_memory = 8 * 1024**3


def _fake_torch(
    *,
    cuda: bool = False,
    props: _Props | None = None,
    mps: bool = False,
    explode: bool = False,
) -> types.ModuleType:
    torch = types.ModuleType("torch")

    def is_available() -> bool:
        if explode:
            raise RuntimeError("driver secret /dev/nvidia0 exploded")
        return cuda

    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=is_available,
        get_device_properties=lambda _index: props,
    )
    torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return torch


def _probe(torch: types.ModuleType | None, *, unsloth: bool = True) -> dict[str, Any]:
    saved_torch = sys.modules.get("torch", None)
    had_torch = "torch" in sys.modules
    saved_present = device._module_present
    device._module_present = lambda _name: unsloth  # type: ignore[assignment]
    # ``None`` in sys.modules makes ``import torch`` raise ImportError.
    sys.modules["torch"] = torch  # type: ignore[assignment]
    try:
        return device.probe()
    finally:
        device._module_present = saved_present  # type: ignore[assignment]
        if had_torch:
            sys.modules["torch"] = saved_torch  # type: ignore[assignment]
        else:
            del sys.modules["torch"]


def test_torch_missing() -> None:
    info = _probe(None)
    assert info["can_finetune"] is False
    assert info["reason_code"] == "torch_missing"


def test_unsloth_missing_on_a_supported_gpu() -> None:
    info = _probe(_fake_torch(cuda=True, props=_Props(8, 6)), unsloth=False)
    assert info["can_finetune"] is False
    assert info["reason_code"] == "unsloth_missing"
    assert info["compute_capability"] == "8.6"


def test_compute_capability_low_reports_the_capability() -> None:
    info = _probe(_fake_torch(cuda=True, props=_Props(6, 1)))
    assert info["can_finetune"] is False
    assert info["reason_code"] == "compute_capability_low"
    assert info["compute_capability"] == "6.1"


def test_ready_gpu_has_no_code() -> None:
    info = _probe(_fake_torch(cuda=True, props=_Props(8, 0)))
    assert info["can_finetune"] is True
    assert info["reason_code"] == ""
    assert info["reason"] == ""


def test_mps_and_cpu_codes() -> None:
    assert _probe(_fake_torch(mps=True))["reason_code"] == "mps_unsupported"
    assert _probe(_fake_torch())["reason_code"] == "no_cuda_gpu"


def test_probe_failure_keeps_the_exception_out_of_the_reason() -> None:
    info = _probe(_fake_torch(explode=True))
    assert info["can_finetune"] is False
    assert info["reason_code"] == "probe_failed"
    assert "exploded" not in info["reason"]
    assert "/dev/nvidia0" not in info["reason"]


def main() -> None:
    test_torch_missing()
    test_unsloth_missing_on_a_supported_gpu()
    test_compute_capability_low_reports_the_capability()
    test_ready_gpu_has_no_code()
    test_mps_and_cpu_codes()
    test_probe_failure_keeps_the_exception_out_of_the_reason()


if __name__ == "__main__":
    main()
