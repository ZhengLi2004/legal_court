#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/.pids"
BACKEND_PID_FILE="$PID_DIR/dev_backend.pid"
FRONTEND_PID_FILE="$PID_DIR/dev_frontend.pid"

stop_pid() {
	local label="$1"
	local pid_file="$2"

	if [[ ! -f "$pid_file" ]]; then
		echo "$label: no pid file"
		return
	fi

	local pid
	pid="$(cat "$pid_file")"

	if kill -0 "$pid" >/dev/null 2>&1; then
		kill "$pid" >/dev/null 2>&1 || true

		for _ in $(seq 1 20); do
			if ! kill -0 "$pid" >/dev/null 2>&1; then
				break
			fi

			sleep 0.2
		done

		if kill -0 "$pid" >/dev/null 2>&1; then
			kill -9 "$pid" >/dev/null 2>&1 || true
		fi

		echo "$label: stopped ($pid)"
	else
		echo "$label: process already exited ($pid)"
	fi

	rm -f "$pid_file"
}

stop_pid "Backend" "$BACKEND_PID_FILE"
stop_pid "Frontend" "$FRONTEND_PID_FILE"
