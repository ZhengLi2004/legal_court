#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-20260328_step13_claim4_internal_current}"
FREEZE_RUN_ID="${FREEZE_RUN_ID:-20260323_step09a_status_small}"
SOURCE_CLAIM1_RUN_ID="${SOURCE_CLAIM1_RUN_ID:-20260323_step10_claim1_status_small}"
REPORTS_ROOT="${REPORTS_ROOT:-reports/experiments}"
INPUT_PATH="${INPUT_PATH:-data/sampling/cleaned_samples.jsonl}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RESUME="${RESUME:-0}"
mkdir -p "$LOG_DIR"
resume_args=()

if [[ "$RESUME" == "1" ]]; then
	resume_args=(--resume)
fi

run_logged() {
	local log_name="$1"
	shift
	local log_path="$LOG_DIR/${RUN_ID}.${log_name}.log"
	echo "Running ${log_name} -> ${log_path}"
	"$PYTHON_BIN" -u -m benchmarks.experiments.core.orchestrator "$@" >"$log_path" 2>&1
}

run_prepare() {
	run_logged \
		"prepare" \
		claim4-prepare \
		--freeze-run-id "$FREEZE_RUN_ID" \
		--source-claim1-run-id "$SOURCE_CLAIM1_RUN_ID" \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--input-path "$INPUT_PATH"
}

run_dev() {
	local point
	local repeat

	for point in q25 q75 full; do
		for repeat in 1 2 3; do
			run_logged \
				"dev.fixed.${point}.r${repeat}" \
				claim4-run \
				--run-id "$RUN_ID" \
				--reports-root "$REPORTS_ROOT" \
				--stage dev \
				--policy fixed \
				--point "$point" \
				--repeat "$repeat" \
				"${resume_args[@]}"
		done
	done

	for repeat in 1 2 3; do
		run_logged \
			"dev.adaptive.full.r${repeat}" \
			claim4-run \
			--run-id "$RUN_ID" \
			--reports-root "$REPORTS_ROOT" \
			--stage dev \
			--policy adaptive \
			--point full \
			--repeat "$repeat" \
			"${resume_args[@]}"
	done

	run_logged \
		"dev.audit" \
		claim4-audit \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage dev
}

run_test() {
	run_logged \
		"dev.audit.precheck" \
		claim4-audit \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage dev

	run_logged \
		"test.fixed.q25.r1" \
		claim4-run \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage test \
		--policy fixed \
		--point q25 \
		--repeat 1 \
		"${resume_args[@]}"

	run_logged \
		"test.fixed.q75.r1" \
		claim4-run \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage test \
		--policy fixed \
		--point q75 \
		--repeat 1 \
		"${resume_args[@]}"

	run_logged \
		"test.fixed.full.r1" \
		claim4-run \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage test \
		--policy fixed \
		--point full \
		--repeat 1 \
		"${resume_args[@]}"

	run_logged \
		"test.adaptive.full.r1" \
		claim4-run \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage test \
		--policy adaptive \
		--point full \
		--repeat 1 \
		"${resume_args[@]}"

	run_logged \
		"test.audit" \
		claim4-audit \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT" \
		--stage test
}

run_summarize() {
	run_logged \
		"summarize" \
		claim4-summarize \
		--run-id "$RUN_ID" \
		--reports-root "$REPORTS_ROOT"
}

case "$MODE" in
prepare)
	run_prepare
	;;

dev)
	run_prepare
	run_dev
	;;

test)
	run_test
	;;

summarize)
	run_summarize
	;;

all)
	run_prepare
	run_dev
	run_summarize
	;;

*)
	echo "Unsupported mode: $MODE"
	echo "Usage: bash scripts/run_step13_claim4.sh [prepare|dev|test|summarize|all]"
	exit 1
	;;
esac

echo "Step 13 mode=${MODE} completed for run_id=${RUN_ID}"
