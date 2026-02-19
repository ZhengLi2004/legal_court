"""A command-line script to run a single, non-interactive legal debate experiment.

This script serves as a simple entry point for executing a full debate
simulation from start to finish without a graphical user interface. It
initializes the `DebateEngine`, runs the debate loop until completion, and prints
all outputs, including turn summaries and the final judgment, to the console.
This is useful for batch processing, automated testing, or environments where a
GUI is not available.
"""

import asyncio

from mas.config import SystemConfig
from mas.core.engine import DebateEngine

DATA_FILE = "data/sampling/cleaned_samples.jsonl"


async def run_experiment():
    """Orchestrate the entire experiment.

    This coroutine handles the lifecycle of a single debate simulation:
    1.  Initializes the `DebateEngine` with a system configuration.
    2.  Sets up the engine with a specific data file.
    3.  Enters a loop, calling the engine's `step()` method until the debate
        is marked as finished.
    4.  Prints a summary of each turn's action to the console.
    5.  After the loop, it prints the final adjudication document and the status
        of the root claims.
    6.  Ensures that all resources (like database connections) are closed properly.
    """
    engine = DebateEngine(config=SystemConfig(), judge_config={})

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
        judgment_document = final_snapshot.get("judgment_document", "N/A")
        root_claims_status = final_snapshot.get("root_claims_status", {})
        print("\n--- Judge's Document ---")
        print(judgment_document)
        print("\n--- Claim Statuses ---")

        if root_claims_status:
            for claim_id, status in root_claims_status.items():
                print(f"  {claim_id}: {status}")

        else:
            print("  No claims adjudicated.")

    finally:
        await engine.close_resources()


def main():
    """Run the CLI experiment coroutine inside an isolated event loop.

    This wrapper creates and owns an asyncio loop so the script can be invoked
    directly from synchronous entry points.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_experiment())

    finally:
        loop.close()


if __name__ == "__main__":
    main()
