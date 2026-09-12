#!/usr/bin/env bash
# Build and launch ScorpionApp (the macOS app, started from main.swift via NSApplication).
#
# Builds the app, makes sure MLX's precompiled Metal library sits next to the binary
# (see build-metallib.sh — `swift build` can't produce it), then starts the app.
#
# Usage: scripts/run-app.sh [options] [-- app arguments]
#   -r, --release     build and run the release configuration (faster probes on real models)
#   -d, --detach      launch in the background and return; logs go to .build/ScorpionApp.log
#   -m, --metallib    rebuild mlx.metallib even if one is already in place
#   -b, --build-only  build and prepare everything, but don't launch
#   -h, --help        show this help
#
# Environment is passed through, e.g. SCORPION_FLUX2_DIR (Klein weights), HF_TOKEN,
# SCORPION_SEED_KEY.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=debug
DETACH=0
FORCE_METALLIB=0
BUILD_ONLY=0
APP_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--release) CONFIG=release ;;
    -d|--detach) DETACH=1 ;;
    -m|--metallib) FORCE_METALLIB=1 ;;
    -b|--build-only) BUILD_ONLY=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; APP_ARGS=("$@"); break ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "ScorpionApp needs macOS on Apple Silicon." >&2
  exit 1
fi
if ! xcrun -sdk macosx -f metal >/dev/null 2>&1; then
  echo "The Metal toolchain is missing: install Xcode (xcrun metal must work)." >&2
  exit 1
fi

echo "· building ScorpionApp ($CONFIG)"
swift build -c "$CONFIG" --product ScorpionApp

BIN_DIR="$(swift build -c "$CONFIG" --show-bin-path)"
APP="$BIN_DIR/ScorpionApp"
[[ -x "$APP" ]] || { echo "build produced no $APP" >&2; exit 1; }

# MLX loads mlx.metallib from next to the binary; build it when missing, stale, or asked.
METALLIB="$BIN_DIR/mlx.metallib"
BUILT="$ROOT/.build/metallib/mlx.metallib"
if [[ $FORCE_METALLIB -eq 1 || ! -f "$METALLIB" ]]; then
  echo "· building mlx.metallib"
  "$ROOT/scripts/build-metallib.sh" "$CONFIG"
elif [[ -f "$BUILT" && "$BUILT" -nt "$METALLIB" ]]; then
  cp "$BUILT" "$METALLIB"
fi

if [[ $BUILD_ONLY -eq 1 ]]; then
  echo "ready: $APP"
  exit 0
fi

if [[ $DETACH -eq 1 ]]; then
  LOG="$ROOT/.build/ScorpionApp.log"
  nohup "$APP" "${APP_ARGS[@]+"${APP_ARGS[@]}"}" >"$LOG" 2>&1 &
  echo "· ScorpionApp running (pid $!) — log: $LOG"
else
  echo "· launching ScorpionApp (Ctrl-C here quits it)"
  exec "$APP" "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
fi
