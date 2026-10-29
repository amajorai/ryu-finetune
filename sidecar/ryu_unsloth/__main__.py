"""Entry point: `python -m ryu_unsloth` starts the uvicorn server.

The server always binds loopback (127.0.0.1). The port is overridable via env
so Core can pin it at spawn time: RYU_UNSLOTH_PORT (default 8086)
"""

from __future__ import annotations

import os

import uvicorn

from . import DEFAULT_PORT
from .paths import prepare_private_storage


def main() -> None:
    # Loopback only: Core proxies this process from the same machine.
    host = "127.0.0.1"
    port = int(os.environ.get("RYU_UNSLOTH_PORT", str(DEFAULT_PORT)))
    # Before serving: drop checkpoints and GGUF scratch a crashed run left
    # behind, and keep the app's weight folders owner-only.
    prepare_private_storage()
    # Single worker on purpose: training jobs hold GPU/model state in-process.
    uvicorn.run("ryu_unsloth.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
