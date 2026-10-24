#!/usr/bin/env python3
"""Thin explicit-context entrypoint for the formal online GRPO package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root(argv: list[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root")
    known, _ = parser.parse_known_args(argv)
    return Path(known.repo_root).expanduser().resolve() if known.repo_root else None


def main() -> int:
    root = _repo_root(sys.argv[1:])
    if root is not None:
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    from enterprise_agent_rl.grpo.cli import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
