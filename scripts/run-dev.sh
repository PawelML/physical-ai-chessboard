#!/usr/bin/env bash
# Start the Chess Arena locally (backend + frontend) so the app is reachable on
# the LAN at http://<this-machine-ip>:5173 — e.g. http://192.168.10.73:5173.
#
# Usage:   bash scripts/run-dev.sh
# Stop:    Ctrl-C  (stops both backend and frontend together)
#
# This does NOT auto-start on boot; it only runs while you keep it open.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Backend talks to this DB; the one with your games. Override with: ARENA_DATABASE_URL=... bash scripts/run-dev.sh
export ARENA_DATABASE_URL="${ARENA_DATABASE_URL:-sqlite+aiosqlite:///./arena.db}"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Pick the venv's uvicorn if present, else whatever is on PATH.
if [ -x ".venv/bin/uvicorn" ]; then
  UVICORN=".venv/bin/uvicorn"
else
  UVICORN="uvicorn"
fi

# Free the ports if a stale process is holding them (the bug we hit last time).
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  pids="$(lsof -ti ":$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "→ port $port zajęty przez [$pids] — ubijam stary proces"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -ti ":$port" 2>/dev/null || true)"
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  fi
done

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

cleanup() {
  echo
  echo "→ zatrzymuję backend + frontend…"
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "→ backend  (uvicorn) na 127.0.0.1:$BACKEND_PORT  [DB=$ARENA_DATABASE_URL]"
"$UVICORN" backend.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "→ frontend (vite)    na 0.0.0.0:$FRONTEND_PORT"
( cd frontend && npm run dev -- --port "$FRONTEND_PORT" ) &
FRONTEND_PID=$!

echo
echo "===================================================================="
echo "  Otwórz na Macu:  http://${LAN_IP:-<ip-ubuntu>}:$FRONTEND_PORT"
echo "  (Ctrl-C zatrzymuje oba procesy)"
echo "===================================================================="

# Keep running until either child exits or you Ctrl-C.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
