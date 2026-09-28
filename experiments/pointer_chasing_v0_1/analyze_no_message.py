"""Exhaustively measure small-n no-message prior and local MAP baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_tasks import exact_no_message_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        exact_no_message_diagnostic(size, depth)
        for size in args.sizes
        for depth in args.depths
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "conditions": len(rows), "episodes": sum(row["episode_count"] for row in rows)}))


if __name__ == "__main__":
    main()
