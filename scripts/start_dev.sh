#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
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

port_is_free() {
	local port="$1"

	python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    print("busy")
    sys.exit(1)
finally:
    sock.close()
print("free")
PY
}

wait_http_ok() {
	local url="$1"
	local timeout="$2"

	python - "$url" "$timeout" <<'PY'
import json
import sys
import time
import urllib.error
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

	if [[ -f "$BACKEND_PID_FILE" ]]; then
		kill "$(cat "$BACKEND_PID_FILE")" >/dev/null 2>&1 || true
		rm -f "$BACKEND_PID_FILE"
	fi

	if [[ -f "$FRONTEND_PID_FILE" ]]; then
		kill "$(cat "$FRONTEND_PID_FILE")" >/dev/null 2>&1 || true
		rm -f "$FRONTEND_PID_FILE"
	fi
}

require_cmd python
require_cmd npm
require_cmd bash
mkdir -p "$LOG_DIR" "$PID_DIR"

if [[ -f "$BACKEND_PID_FILE" ]] || [[ -f "$FRONTEND_PID_FILE" ]]; then
	echo "PID file exists. Run scripts/stop_dev.sh first."
	exit 1
fi

if ! port_is_free "$BACKEND_PORT" >/dev/null; then
	echo "Backend port is busy: $BACKEND_PORT"
	exit 1
fi

if ! port_is_free "$FRONTEND_PORT" >/dev/null; then
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

nohup python -m uvicorn mas.api.server:app \
	--host "$BACKEND_HOST" \
	--port "$BACKEND_PORT" \
	>"$BACKEND_LOG" 2>&1 &

echo $! >"$BACKEND_PID_FILE"

if ! wait_http_ok "http://$BACKEND_HOST:$BACKEND_PORT/api/v1/health" "$HEALTH_TIMEOUT_SEC"; then
	echo "Backend health check failed. See $BACKEND_LOG"
	exit 1
fi

echo "Starting frontend..."

nohup env \
	VITE_COMPAT_MODE=http \
	VITE_API_BASE_URL="http://$BACKEND_HOST:$BACKEND_PORT" \
	npm --prefix "$ROOT_DIR/frontend" run dev -- \
	--host "$FRONTEND_HOST" \
	--port "$FRONTEND_PORT" \
	>"$FRONTEND_LOG" 2>&1 &

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
echo "Backend PID : $(cat "$BACKEND_PID_FILE")"
echo "Frontend PID: $(cat "$FRONTEND_PID_FILE")"
echo "Stop with: bash scripts/stop_dev.sh"
