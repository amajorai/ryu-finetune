# ryu-finetune

Fine-tuning (Unsloth) for Ryu — a LoRA/QLoRA training studio: durable job state, the trained-adapter catalog, and the Python training sidecar.

> **Read-only mirror.** Developed in https://github.com/amajorai/ryu —
> please open issues and pull requests there, not on this repository.

## Install

- Binary: `ryu-finetune` from the [Ryu releases](https://github.com/amajorai/ryu/releases).
- Crate: `cargo install ryu-finetune`.

## License

Apache-2.0 — see [LICENSE](./LICENSE).

---

# com.ryu.finetune — Fine-tuning

A LoRA/QLoRA training studio: launch fine-tune jobs on this node's GPU (or a remote
Ryu Cloud GPU node), track them durably, merge the trained adapter to GGUF, and
register the result as a swappable local model. The app-ified successor to the
built-in desktop fine-tuning page.

## Parts

- **`backend/` — `ryu-finetune` (out-of-process control-plane sidecar).** An extracted
  Core capability crate now run as a standalone `[[bin]]` (`kind:local`, `public_mount`,
  `RYU_FINETUNE_BIN`/`RYU_FINETUNE_PORT`, default `:7990`); Core links **zero finetune
  code** (no path-dep) and proxies `/api/finetune/*` to it. It owns the *durable* records
  that must survive a restart: the `FinetuneStore` job DB (`finetune.db`) and the
  trained-adapter catalog (`installed-adapters.json`), plus the Python `unsloth` worker
  (over `RYU_UNSLOTH_URL`). Core's one reverse-coupling — the `host.finetune_*` plugin-host
  bridge — reaches the sidecar over loopback via `apps/core/src/finetune_client.rs`.
- **`sidecar/` — `ryu-unsloth` (out-of-process Python sidecar).** A FastAPI runtime
  wrapping the Apache-2.0 `unsloth` library + TRL `SFTTrainer` behind one small HTTP
  contract; the actual training runs here. Base server deps boot on any machine; the
  heavy training stack (`unsloth`/`torch`/CUDA/`trl`) is the optional `[train]`
  extra so `/health` serves even without a GPU. Manifest-managed: provisioned from a
  hosted tarball and started on enable + boot-reconcile.
- **`ui/` — companion (`@ryu/finetune-app`).** A sandboxed full-page Companion
  (Path B, `ui_format: "html"`), one self-contained `dist/index.html`. Drives the
  orchestration + job store through `window.ryu.finetune.*`.

## Manifest (`ui/plugin.json`)

- **Capability grant:** `finetune:runs`.
- **Sidecar:** `unsloth` on `:8086`, `process.kind: python`, entry `ryu_unsloth`,
  Python 3.11, `pyproject_extra: train`; source is a hosted `tar.gz` release
  (`unsloth-sidecar-v1`) auto-published from the public repo mirror.
- **Runnable:** one `companion` (`Fine-tuning`, icon `sparkles`).

## Split of concerns / swap seam

Core owns *what runs* (GPU gate, job store, adapter→GGUF merge, model registration);
the Python sidecar only trains; the durable crate only persists. Training backend and
the record store are independently swappable behind the `/api/finetune/*` contract.
