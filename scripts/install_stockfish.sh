#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/stockfish"

mkdir -p "$VENDOR_DIR/download" "$VENDOR_DIR/root"
cd "$VENDOR_DIR/download"

apt-get download stockfish
dpkg-deb -x stockfish_*.deb "$VENDOR_DIR/root"

printf 'uci\nquit\n' | "$VENDOR_DIR/root/usr/games/stockfish" | sed -n '1,5p'
