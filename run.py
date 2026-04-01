"""A command-line script to run a single, non-interactive legal debate experiment.

This script serves as a simple entry point for executing a full debate
simulation from start to finish without a graphical user interface. It
initializes the `DebateEngine`, runs the debate loop until completion, and prints
turn summaries plus final claim statuses to the console.
This is useful for batch processing, automated testing, or environments where a
GUI is not available.
"""

import argparse
import asyncio
import os
from pathlib import Path

from mas.core.engine import DebateEngine
from mas.infrastructure.settings_provider import build_system_config

_DEFAULT_CASE_FILE = (
    Path(__file__).resolve().parent / "data" / "sampling" / "cleaned_samples.jsonl"
)

DATA_FILE = str(os.getenv("MAS_CASE_DATA_FILE", str(_DEFAULT_CASE_FILE)))


async def run_experiment(*, memory_dir: str | None = None):
    """Orchestrate the entire experiment.

    This coroutine handles the lifecycle of a single debate simulation:
    1.  Initializes the `DebateEngine` with a system configuration.
    2.  Sets up the engine with a specific data file.
    3.  Enters a loop, calling the engine's `step()` method until the debate
        is marked as finished.
    4.  Prints a summary of each turn's action to the console.
    5.  After the loop, it prints the final status of the root claims.
    6.  Ensures that all resources (like database connections) are closed properly.

    Args:
        memory_dir: Optional override for the runtime memory directory.
    """
    config = build_system_config(memory_dir)
    engine = DebateEngine(config=config)

    try:
        import json

        with open(DATA_FILE, "r", encoding="utf-8") as f:
            case_data = json.loads(f.readline())

        await engine.setup(case_data=case_data)

        while not engine.is_finished:
            if engine.is_ready_for_adjudication:
                await engine.adjudicate()
                continue

            await engine.step()
            snapshot = engine.get_snapshot()
            log = snapshot.get("last_log", {})
            print(f"\n--- Turn {log.get('turn')} Summary ---")
            print(f"Action: {log.get('action')}")

        final_snapshot = engine.get_snapshot()
        print("\n" + "=" * 30 + " FINAL ADJUDICATION " + "=" * 30)
        root_claims_status = final_snapshot.get("root_claims_status", {})
        print("\n--- Claim Statuses ---")

        if root_claims_status:
            for claim_id, status in root_claims_status.items():
                print(f"  {claim_id}: {status}")

        else:
            print("  No claims adjudicated.")

    finally:
        await engine.close_resources()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one legal debate experiment.")
    parser.add_argument("--memory-dir", default="")
    return parser


def main(argv: list[str] | None = None):
    """Run the CLI experiment coroutine inside an isolated event loop.

    This wrapper creates and owns an asyncio loop so the script can be invoked
    directly from synchronous entry points.

    Args:
        argv: Optional command-line argument override.
    """
    args = _build_parser().parse_args(argv)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_experiment(memory_dir=args.memory_dir or None))

    finally:
        loop.close()


if __name__ == "__main__":
    main()
