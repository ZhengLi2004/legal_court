#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/.pids"
BACKEND_PID_FILE="$PID_DIR/dev_backend.pid"
FRONTEND_PID_FILE="$PID_DIR/dev_frontend.pid"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

wait_for_exit() {
	local pid="$1"
	local timeout_loops="${2:-30}"

	for _ in $(seq 1 "$timeout_loops"); do
		if ! kill -0 "$pid" >/dev/null 2>&1; then
			return 0

		fi
		sleep 0.2

	done
	return 1
}

stop_pid_group() {
	local label="$1"
	local pid_file="$2"

	if [[ ! -f "$pid_file" ]]; then
		echo "$label: no pid file"
		return

	fi
	local pid
	pid="$(cat "$pid_file")"

	if [[ -z "$pid" ]]; then
		echo "$label: empty pid file"
		rm -f "$pid_file"
		return

	fi

	if kill -0 "$pid" >/dev/null 2>&1; then
		kill -- "-$pid" >/dev/null 2>&1 || true
		kill "$pid" >/dev/null 2>&1 || true

		if ! wait_for_exit "$pid" 35; then
			kill -9 -- "-$pid" >/dev/null 2>&1 || true
			kill -9 "$pid" >/dev/null 2>&1 || true

		fi
		echo "$label: stopped group ($pid)"

	else
		echo "$label: process already exited ($pid)"

	fi
	rm -f "$pid_file"
}

pids_on_port() {
	local port="$1"

	if command -v lsof >/dev/null 2>&1; then
		lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
		return

	fi

	if command -v fuser >/dev/null 2>&1; then
		fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u
		return

	fi

	if command -v ss >/dev/null 2>&1; then
		ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u

	fi
}

cleanup_port() {
	local label="$1"
	local port="$2"
	local pids
	pids="$(pids_on_port "$port" || true)"

	if [[ -z "$pids" ]]; then
		echo "$label: port $port is free"
		return

	fi
	echo "$label: cleaning lingering listeners on port $port ($pids)"

	for pid in $pids; do
		if [[ "$pid" != "$$" ]]; then
			kill "$pid" >/dev/null 2>&1 || true

		fi

	done
	sleep 0.6
	pids="$(pids_on_port "$port" || true)"

	if [[ -n "$pids" ]]; then
		for pid in $pids; do
			if [[ "$pid" != "$$" ]]; then
				kill -9 "$pid" >/dev/null 2>&1 || true

			fi

		done

	fi
	local remaining
	remaining="$(pids_on_port "$port" || true)"

	if [[ -n "$remaining" ]]; then
		echo "$label: port $port still occupied by $remaining"
		return 1

	fi
	echo "$label: port $port cleared"
}

stop_pid_group "Backend" "$BACKEND_PID_FILE"
stop_pid_group "Frontend" "$FRONTEND_PID_FILE"
cleanup_port "Backend" "$BACKEND_PORT"
cleanup_port "Frontend" "$FRONTEND_PORT"
