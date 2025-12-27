import asyncio
from metagpt.logs import logger
from mas.engine import DebateEngine
from mas.config import SystemConfig
DATA_FILE = "data/sampling/cleaned_samples.jsonl"

async def run_experiment():
    logger.info(">>> Starting Experiment Run...")

    engine = DebateEngine(
        config=SystemConfig(),
        judge_config={}
    )

    try:
        await engine.setup(DATA_FILE)

        while not engine.is_finished:
            await engine.step()
            snapshot = engine.get_snapshot()
            log = snapshot.get("last_log", {})
            print(f"\n--- Turn {log.get('turn')} Summary ---")
            print(f"Action: {log.get('action')}")

        final_snapshot = engine.get_snapshot()
        print("\n" + "="*30 + " FINAL ADJUDICATION " + "="*30)
        judgment_document = final_snapshot.get("judgment_document", "N/A")
        root_claims_status = final_snapshot.get("root_claims_status", {})
        print("\n--- Judge's Document ---")
        print(judgment_document)
        print("\n--- Claim Statuses ---")
        
        if root_claims_status:
            for claim_id, status in root_claims_status.items(): print(f"  {claim_id}: {status}")
        
        else: print("  No claims adjudicated.")

    finally: await engine.close_resources()

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try: loop.run_until_complete(run_experiment())
    finally: loop.close()

if __name__ == "__main__": main()