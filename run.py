import asyncio
from metagpt.logs import logger
from mas.engine import DebateEngine
from mas.config import SystemConfig
DATA_FILE = "data/sampling/cleaned_samples.jsonl"

JUDGE_CONFIG = {
    "model_name": "法衡",
    "temperature": 0.0,
    "max_tokens": 512
}

async def main():
    logger.info(">>> Starting Non-Interactive (Experiment) Run...")
    engine = DebateEngine(config=SystemConfig(), judge_config=JUDGE_CONFIG)
    
    try:
        await engine.setup(DATA_FILE)
        
        while not engine.is_finished:
            await engine.step()
            snapshot = engine.get_snapshot()
            log = snapshot.get("last_log", {})
            print(f"--- Turn {log.get('turn')} Summary ---")
            print(log.get('action', 'No action logged.'))

        final_snapshot = engine.get_snapshot()
        winner = final_snapshot.get("winner", "Unknown")
        print("\n" + "="*30 + " FINAL RESULT " + "="*30)
        print(f"Winner: {winner}")

    finally:
        if engine: await engine.close_resources()

if __name__ == "__main__": asyncio.run(main())