"""Terminal display helpers."""

from __future__ import annotations

import functools

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[97m"

print = functools.partial(print, flush=True)  # type: ignore[assignment]


def header(text: str) -> None:
    print(f"\n{'━' * 70}")
    print(f"  {BOLD}{CYAN}{text}{RESET}")
    print(f"{'━' * 70}")


def bar(value: float, width: int = 25) -> str:
    filled = int(value * width)
    return f"{GREEN}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"
