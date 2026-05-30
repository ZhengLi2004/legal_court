# Legal Court MAS

This repository contains a multi-agent legal reasoning system and the experiment
pipeline used to evaluate claim extraction, claim-status judgment, learning
attribution, and quality-cost tradeoffs.

The project has two main entry points:

- `run.py`: run one legal debate case from the command line.
- `benchmarks/experiments/core/orchestrator.py`: run the frozen experiment
  pipeline.

The repository does not require checked-in data or credentials. Fill local data,
model paths, and API credentials in your own environment before running.

## Repository Layout

- `mas/`: core multi-agent system, graph logic, memory, retrieval, API service.
- `benchmarks/experiments/`: experiment data loaders, method wrappers,
  evaluation metrics, orchestration, and reporting.
- `benchmarks/optim/`: offline threshold and retrieval-parameter tuning tools.
- `data/`: local case data. The default single-case input is
  `data/sampling/cleaned_samples.jsonl`.
- `reports/experiments/`: default output root for experiment artifacts.
- `frontend/`: React/Vite frontend for API-backed inspection.
- `scripts/`: local scratch script location. This directory is intentionally not
  required by the documented workflow.

## Environment

Use Python 3.10 unless you have already verified another version with your local
dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies only if you need the web UI:

```bash
npm --prefix frontend install
```

Create a local `.env` file. Do not commit this file.

```dotenv
ES_HOST=http://127.0.0.1:9200
EMBEDDING_MODEL_PATH=/absolute/path/to/bge-m3
MAS_STORAGE_DIR=/absolute/path/to/runtime_storage
LEGAL_LLM_KEY=your_api_key
LEGAL_LLM_URL=https://your-compatible-chat-endpoint/v1

# Optional: override the default single-case JSONL input.
MAS_CASE_DATA_FILE=/absolute/path/to/cleaned_samples.jsonl

# Optional: frontend/API integration.
MAS_CORS_ORIGINS=http://localhost:5173

# Optional: evaluator profiles may reference additional API variables.
OPENAI_API_KEY=your_optional_openai_key
```

The LLM endpoint is expected to be OpenAI-compatible. The default model name and
sampling parameters are defined in `mas/config.py`.

## Data

The command-line and benchmark entry points consume JSONL files with one case
object per line. The built-in default is:

```text
data/sampling/cleaned_samples.jsonl
```

Two schemas are supported by the local loaders:

- A nested raw case format with `metaInfo` and `content`.
- A flat case format compatible with `data/schema.py`, including fields such as
  `uid`, `title`, `cause`, `plaintiffs`, `defendants`, `plaintiff_claim`,
  `defendant_argument`, `fact_finding`, `court_opinion`, `verdict_result`, and
  `cited_laws`.

For benchmark runs, also prepare or fill the frozen gold artifacts under:

```text
benchmarks/experiments/artifacts/gold/
benchmarks/experiments/artifacts/splits/
```

The default paths are referenced by `benchmarks/experiments/core/step09a.py`.

## Run One Case

Run the first case in the configured JSONL file:

```bash
python run.py
```

Use an isolated memory directory for a specific run:

```bash
python run.py --memory-dir /absolute/path/to/runtime_storage/single_case_run
```

If `MAS_CASE_DATA_FILE` is unset, `run.py` reads
`data/sampling/cleaned_samples.jsonl`.

## Run API And Frontend

Start the backend API:

```bash
python run_api.py
```

In another terminal, start the frontend:

```bash
npm --prefix frontend run dev
```

Useful frontend checks:

```bash
npm --prefix frontend run lint
npm --prefix frontend run test:unit
npm --prefix frontend run build
```

## Experiment Workflow

All benchmark commands below write into `reports/experiments/<run-id>/` unless
`--reports-root` is changed. Use run IDs that encode the date and purpose, for
example `20260513_claim1_full`.

Before a formal run, freeze the protocol and verify local dependencies:

```bash
python -m benchmarks.experiments.core.step09a preflight \
  --run-id <freeze_run_id> \
  --input-path <your_cases.jsonl> \
  --memory-dir <runtime_memory_dir> \
  --sample-size 3
```

Finalize the freeze after reviewing the preflight output:

```bash
python -m benchmarks.experiments.core.step09a finalize \
  --run-id <freeze_run_id>
```

### Claim 1: Main Status Experiment

Pilot run:

```bash
python -m benchmarks.experiments.core.orchestrator claim1-pilot \
  --freeze-run-id <freeze_run_id> \
  --run-id <claim1_pilot_run_id> \
  --input-path <your_cases.jsonl> \
  --memory-dir <runtime_memory_dir> \
  --resume
```

Full run:

```bash
python -m benchmarks.experiments.core.orchestrator claim1-full \
  --freeze-run-id <freeze_run_id> \
  --run-id <claim1_full_run_id> \
  --input-path <your_cases.jsonl> \
  --memory-dir <runtime_memory_dir> \
  --pilot-review-path reports/experiments/<claim1_pilot_run_id>/claim1/pilot_review.json \
  --resume
```

Restrict methods when needed:

```bash
python -m benchmarks.experiments.core.orchestrator claim1-full \
  --freeze-run-id <freeze_run_id> \
  --run-id <claim1_full_run_id> \
  --input-path <your_cases.jsonl> \
  --memory-dir <runtime_memory_dir> \
  --pilot-review-path <pilot_review.json> \
  --methods main_system baseline_b1_structured_rag \
  --resume
```

### Claim 2: Consistency And Faithfulness

Prepare an evaluator profile from one of the templates in
`benchmarks/experiments/artifacts/evaluator_profiles/`, fill the API variables,
then run:

```bash
python -m benchmarks.experiments.core.orchestrator claim2-run \
  --freeze-run-id <freeze_run_id> \
  --source-claim1-run-id <claim1_full_run_id> \
  --run-id <claim2_run_id> \
  --input-path <your_cases.jsonl> \
  --evaluator-profile-path <filled_evaluator_profile.json> \
  --resume
```

### Claim 3: Learning Attribution

Prepare warmup snapshots and fixed evidence packs:

```bash
python -m benchmarks.experiments.core.orchestrator claim3-prepare \
  --freeze-run-id <freeze_run_id> \
  --source-claim1-run-id <claim1_full_run_id> \
  --run-id <claim3_run_id> \
  --input-path <your_cases.jsonl> \
  --warmup-points 0,25,50,75,100 \
  --fixed-pack-points 0,25,50,75,100
```

Run automatic and fixed-evidence branches:

```bash
python -m benchmarks.experiments.core.orchestrator claim3-run \
  --run-id <claim3_run_id> \
  --branch normal \
  --resume

python -m benchmarks.experiments.core.orchestrator claim3-run \
  --run-id <claim3_run_id> \
  --branch fixed-pack \
  --resume
```

Summarize:

```bash
python -m benchmarks.experiments.core.orchestrator claim3-summarize \
  --run-id <claim3_run_id>
```

### Claim 4: Quality-Cost Tradeoff

Prepare the stage:

```bash
python -m benchmarks.experiments.core.orchestrator claim4-prepare \
  --freeze-run-id <freeze_run_id> \
  --source-claim1-run-id <claim1_full_run_id> \
  --run-id <claim4_run_id> \
  --input-path <your_cases.jsonl>
```

Run fixed-budget slices. Repeat for the required stage, budget point, and repeat.

```bash
python -m benchmarks.experiments.core.orchestrator claim4-run \
  --run-id <claim4_run_id> \
  --stage dev \
  --policy fixed \
  --point q50 \
  --repeat 1 \
  --resume
```

For Claim 4, use `--policy fixed`. Valid budget points are `q25`, `q50`,
`q75`, and `full`; valid repeats are `1`, `2`, and `3`.

Audit and summarize:

```bash
python -m benchmarks.experiments.core.orchestrator claim4-audit \
  --run-id <claim4_run_id> \
  --stage dev

python -m benchmarks.experiments.core.orchestrator claim4-summarize \
  --run-id <claim4_run_id>
```

### Final Reporting

Prepare consolidated reporting inputs:

```bash
python -m benchmarks.experiments.core.orchestrator step14-prepare \
  --run-id <reporting_run_id> \
  --claim1-internal-run-id <claim1_internal_run_id> \
  --claim1-external-run-id <claim1_external_run_id> \
  --claim2-internal-run-id <claim2_internal_run_id> \
  --claim2-external-run-id <claim2_external_run_id> \
  --claim3-run-id <claim3_run_id> \
  --claim4-run-id <claim4_run_id>
```

Rebuild reports and figures from existing artifacts:

```bash
python -m benchmarks.experiments.core.orchestrator step14-rebuild \
  --run-id <reporting_run_id>
```

## Resume, Logs, And Long Runs

Most experiment commands accept `--resume`. Keep the same `--run-id` and rerun
the same command to continue from already completed artifacts.

For long runs, redirect output to a local log file:

```bash
nohup python -m benchmarks.experiments.core.orchestrator claim3-run \
  --run-id <claim3_run_id> \
  --branch normal \
  --resume \
  > logs/<claim3_run_id>_normal.log 2>&1 &
```

Create the `logs/` directory if it does not exist:

```bash
mkdir -p logs
```

## Validation

Run backend tests:

```bash
python -m pytest tests -q
```

Run frontend checks:

```bash
npm --prefix frontend run lint
npm --prefix frontend run test:unit
npm --prefix frontend run build
```

## Notes

- Keep credentials only in `.env` or your shell environment.
- Keep generated reports under `reports/experiments/`.
- Use separate `MAS_STORAGE_DIR` or `--memory-dir` values when comparing runs.
- Do not edit frozen artifacts for a completed run unless you intentionally
  invalidate and rebuild downstream reports.
