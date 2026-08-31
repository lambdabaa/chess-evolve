"""CLI entry point for chess-evolve."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess prompt evolution via remote-factory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run the evolution loop")

    serve_parser = sub.add_parser("serve", help="Start the live web UI")
    serve_parser.add_argument("--port", type=int, default=8422)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    if args.command == "run":
        from chess_evolve.evolve import main as evolve_main
        asyncio.run(evolve_main())
    elif args.command == "serve":
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "chess_evolve.serve:app",
             "--host", args.host, "--port", str(args.port), "--log-level", "warning"],
        )
    else:
        parser.print_help()
