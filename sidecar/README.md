# <img src="https://raw.githubusercontent.com/amajorai/ryu/main/.github/logo.png" width="50" align="middle" alt="" />&nbsp; Ryu Unsloth Sidecar

> A Core-managed runtime for SFT and DPO fine-tuning with LoRA/QLoRA. Part of [Ryu](../../../README.md).

[![License](https://shieldcn.dev/badge/License-Apache--2.0-73DC8C.svg?logo=apache&logoColor=white)](../../../README.md#repository-layout--licensing)
[![Stack](https://shieldcn.dev/badge/Python-FastAPI-3776AB.svg?logo=python&logoColor=white)](../../../README.md)

A Core-managed Python runtime that wraps the Apache-2.0 [`unsloth`](https://github.com/unslothai/unsloth) library and TRL to run supervised fine-tuning and Direct Preference Optimization (DPO) with LoRA/QLoRA adapters behind one small HTTP contract. Ryu Core owns lifecycle, persistence, routing, and the desktop UI, so this process only trains. It uses the library, not Unsloth's AGPL-3.0 Studio UI. Same shape as `apps-store/voice/sidecar`: a pure runtime Core spawns and health-checks, fenced out of the bun/turbo workspace, on its own Python toolchain.

**Tier:** OSS, Apache-2.0

## Run

```bash
bun run dev:unsloth          # from repo root → python -m ryu_unsloth on 127.0.0.1:8086
```

Install deps into a venv first:

```bash
cd apps-store/finetune/sidecar
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/ on *nix
pip install -e .            # server only (boots anywhere)
pip install -e ".[test]"   # + the in-process smoke-test client
pip install -e ".[train]"   # + the training stack (needs a CUDA GPU to train)
```

Run the dependency-light lifecycle smoke test with `python smoke_test.py` after installing the
`test` extra. It proves auth, validation, job failure handling, and SSE without downloading a
model; a real training/merge run still needs the `train` extra and an NVIDIA CUDA GPU.

## What it provides

- **Fine-tune jobs over HTTP:** `POST /finetune` starts a job; `GET /finetune[/{id}]` lists/inspects snapshots; `GET /finetune/{id}/stream` is SSE progress; `DELETE /finetune/{id}` cancels cooperatively.
- **Dataset formats:** SFT accepts `chat`, `alpaca`, or `text` samples. DPO accepts `preference` rows shaped as `{"prompt":"…","chosen":"…","rejected":"…"}`. Both methods accept configurable LoRA and training parameters.
- **Hardware honesty:** `GET /health` reports `can_finetune`/`backend`/`gpu`/`vram_bytes`; Core gates local training on it and falls back to a remote node. Training needs an NVIDIA CUDA GPU (compute ≥ 7.0); CPU/Apple-Silicon machines can run the server but not train.
- **Outputs:** LoRA adapters under `$RYU_MODELS_DIR/adapters` (`<data dir>/models/adapters/<artifact_id>/`, a new directory per job named by the server-generated `artifact_id`, never by the caller's `output_name`); merged GGUFs under `$RYU_MODELS_DIR/chat` (`<data dir>/models/chat/<name>.gguf`). File-backed datasets are read only from `$RYU_PLUGIN_DATA_DIR/datasets`. Per-job trainer checkpoints go to `$RYU_PLUGIN_DATA_DIR/checkpoints/<job>` and are deleted when the job ends.

## Env

| Var | Default | Meaning |
|---|---|---|
| `RYU_UNSLOTH_PORT` | `8086` | bind port |
| `RYU_MODELS_DIR` | `$RYU_DIR/models` | shared models root (Core injects it) |
| `RYU_PLUGIN_DATA_DIR` | `$RYU_DIR/plugin-data/@ryu+finetune` | this app's data dir (Core injects it) |
| `HF_HOME` | `$RYU_DIR/models/hf` | Hugging Face cache (Core injects the shared one) |

## License

Apache-2.0. See [LICENSE](../../../README.md#repository-layout--licensing). © 2026 A Major Pte. Ltd.
