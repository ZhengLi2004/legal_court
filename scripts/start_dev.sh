#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-30}"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/.pids"
BACKEND_LOG="$LOG_DIR/dev_backend.log"
FRONTEND_LOG="$LOG_DIR/dev_frontend.log"
BACKEND_PID_FILE="$PID_DIR/dev_backend.pid"
FRONTEND_PID_FILE="$PID_DIR/dev_frontend.pid"

require_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "Missing command: $1"
		exit 1
	fi
}

kill_group_by_pid_file() {
	local pid_file="$1"

	if [[ ! -f "$pid_file" ]]; then
		return
	fi

	local pid
	pid="$(cat "$pid_file")"

	if [[ -n "$pid" ]]; then
		kill -- "-$pid" >/dev/null 2>&1 || true
		kill "$pid" >/dev/null 2>&1 || true
	fi

	rm -f "$pid_file"
}

port_is_free() {
	local port="$1"

	python - "$port" <<'PY'
import socket
import sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.6)

try:
    code = sock.connect_ex(("0.0.0.0", port))
    sys.exit(1 if code == 0 else 0)

except Exception:
    sys.exit(0)

finally:
    sock.close()
PY
}

wait_http_ok() {
	local url="$1"
	local timeout="$2"

	python - "$url" "$timeout" <<'PY'
import json
import sys
import time
import urllib.request
url = sys.argv[1]
timeout = int(sys.argv[2])
start = time.time()

while time.time() - start < timeout:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))

                if payload.get("status") == "ok":
                    sys.exit(0)

    except Exception:
        pass

    time.sleep(0.5)

sys.exit(1)
PY
}

wait_port_open() {
	local host="$1"
	local port="$2"
	local timeout="$3"

	python - "$host" "$port" "$timeout" <<'PY'
import socket
import sys
import time
host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
start = time.time()

while time.time() - start < timeout:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.5)

    try:
        sock.connect((host, port))
        sock.close()
        sys.exit(0)

    except Exception:
        sock.close()
        time.sleep(0.5)

sys.exit(1)
PY
}

cleanup_on_error() {
	set +e
	kill_group_by_pid_file "$BACKEND_PID_FILE"
	kill_group_by_pid_file "$FRONTEND_PID_FILE"
}

require_cmd python
require_cmd npm
require_cmd setsid
mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ -f "$BACKEND_PID_FILE" ]] || [[ -f "$FRONTEND_PID_FILE" ]]; then
	echo "PID file exists. Run scripts/stop_dev.sh first."
	exit 1
fi

if ! port_is_free "$BACKEND_PORT"; then
	echo "Backend port is busy: $BACKEND_PORT"
	exit 1
fi

if ! port_is_free "$FRONTEND_PORT"; then
	echo "Frontend port is busy: $FRONTEND_PORT"
	exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1; then
import fastapi
import uvicorn
PY
	echo "Installing Python dependencies..."
	python -m pip install -r requirements.txt

fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
	echo "Installing frontend dependencies..."
	npm --prefix "$ROOT_DIR/frontend" install
fi

trap cleanup_on_error ERR
echo "Starting backend..."

nohup setsid python -m uvicorn mas.api.server:app \
	--host "$BACKEND_HOST" \
	--port "$BACKEND_PORT" \
	>"$BACKEND_LOG" 2>&1 </dev/null &

echo $! >"$BACKEND_PID_FILE"

if ! wait_http_ok "http://$BACKEND_HOST:$BACKEND_PORT/api/v1/health" "$HEALTH_TIMEOUT_SEC"; then
	echo "Backend health check failed. See $BACKEND_LOG"
	exit 1
fi

echo "Starting frontend..."

nohup setsid env \
	VITE_COMPAT_MODE=http \
	VITE_API_BASE_URL="http://$BACKEND_HOST:$BACKEND_PORT" \
	npm --prefix "$ROOT_DIR/frontend" run dev -- \
	--host "$FRONTEND_HOST" \
	--port "$FRONTEND_PORT" \
	>"$FRONTEND_LOG" 2>&1 </dev/null &

echo $! >"$FRONTEND_PID_FILE"

if ! wait_port_open "$FRONTEND_HOST" "$FRONTEND_PORT" "$HEALTH_TIMEOUT_SEC"; then
	echo "Frontend port check failed. See $FRONTEND_LOG"
	exit 1
fi

trap - ERR
echo "Dev stack is up."
echo "Frontend: http://$FRONTEND_HOST:$FRONTEND_PORT"
echo "Backend : http://$BACKEND_HOST:$BACKEND_PORT"
echo "Backend log : $BACKEND_LOG"
echo "Frontend log: $FRONTEND_LOG"
echo "Backend group PID : $(cat "$BACKEND_PID_FILE")"
echo "Frontend group PID: $(cat "$FRONTEND_PID_FILE")"
echo "Stop with: bash scripts/stop_dev.sh"
