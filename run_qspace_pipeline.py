#!/usr/bin/env python
"""Single-command entry point for the Qspace knowledge-compilation pipeline.

Usage:
    python run_qspace_pipeline.py [PROJECT_ROOT]

If PROJECT_ROOT is omitted the current working directory is used. The pipeline
auto-discovers a corpus; if none is found it seeds a synthetic sample corpus.
All artifacts are written under ``outputs/qspace/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from qspace_pipeline.run import run


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    try:
        run(root)
        return 0
    except Exception as exc:  # surface a clean error but never leave a traceback-only exit
        import traceback
        print("\n[FATAL] Pipeline aborted with an unhandled error:")
        traceback.print_exc()
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
