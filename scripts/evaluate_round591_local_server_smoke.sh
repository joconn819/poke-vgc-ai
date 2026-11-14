#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SHOWDOWN="${POKEMON_SHOWDOWN_PATH:-/home/justin/p/poke_learning/pokemon-showdown}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
FORMAT="${FORMAT:-gen9championsdoubles2026mcclosed}"
BOT_USERNAME="${BOT_USERNAME:-CodexEvalBot}"
CHALLENGER_USERNAME="${CHALLENGER_USERNAME:-VGCTestOpp2}"
TEAM_SEED="${TEAM_SEED:-891001}"
CHALLENGER_SEED="${CHALLENGER_SEED:-891101}"
BOT_LOG="$(mktemp)"
export PYTHONPATH=. POKEMON_SHOWDOWN_PATH="$SHOWDOWN" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
venv/bin/python scripts/run_lan_bot.py \
  --checkpoint training/outputs/phase12_headless/round430_hard_opponent_auxiliary_ppo_20k_20261030/checkpoints/latest.pt \
  --team-selection-checkpoint training/outputs/phase12_headless/round402_selector_currentdata_8ep_20261030/selector_candidate.pt \
  --showdown-path "$SHOWDOWN" --format "$FORMAT" --server-host "$SERVER_HOST" --port "$PORT" \
  --username "$BOT_USERNAME" --seed "$TEAM_SEED" --battles 1 >"$BOT_LOG" 2>&1 &
BOT_PID=$!
cleanup() {
  kill "$BOT_PID" 2>/dev/null || true
  wait "$BOT_PID" 2>/dev/null || true
  cat "$BOT_LOG"
  rm -f "$BOT_LOG"
}
trap cleanup EXIT
for _ in $(seq 1 30); do
  if grep -q "Listening for challenges" "$BOT_LOG"; then break; fi
  if ! kill -0 "$BOT_PID" 2>/dev/null; then cat "$BOT_LOG"; exit 1; fi
  sleep 1
done
grep -q "Listening for challenges" "$BOT_LOG" || { echo "Bot did not start listening" >&2; exit 1; }
venv/bin/python scripts/smoke_local_showdown_bot.py \
  --bot "$BOT_USERNAME" --username "$CHALLENGER_USERNAME" --format "$FORMAT" \
  --showdown-path "$SHOWDOWN" --host "$SERVER_HOST" --port "$PORT" \
  --seed "$CHALLENGER_SEED" --timeout 240
