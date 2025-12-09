#!/usr/bin/env bash
set -euo pipefail

# Run common setup steps in one go:
# 1) Import map.xlsx -> map.sqlite
# 2) Pre-cache road network to cache/ehime_drive.graphml
# 3) (optional) Launch Streamlit app
#
# Usage:
#   ./scripts/run_all.sh               # do all steps and launch app
#   ./scripts/run_all.sh --no-app      # skip launching the app
#   ./scripts/run_all.sh --only-app    # only launch the app (assumes data/cache exist)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

XLSX_PATH="${STATIONS_XLSX_PATH:-map.xlsx}"
SQLITE_PATH="${STATIONS_DB_PATH:-map.sqlite}"
GRAPHML_PATH="${GRAPHML_PATH:-cache/ehime_drive.graphml}"

RUN_IMPORT=true
RUN_PRECACHE=true
RUN_APP=true

usage() {
  echo "Usage: $0 [--no-app] [--only-app]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-app)
      RUN_APP=false
      shift
      ;;
    --only-app)
      RUN_IMPORT=false
      RUN_PRECACHE=false
      RUN_APP=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if $RUN_IMPORT; then
  if [[ ! -f "$XLSX_PATH" ]]; then
    echo "[ERROR] Station Excel not found: $XLSX_PATH" >&2
    exit 1
  fi
  echo "[1/3] Importing $XLSX_PATH -> $SQLITE_PATH ..."
  python3 scripts/import_map_to_db.py --input "$XLSX_PATH" --output "$SQLITE_PATH"
fi

if $RUN_PRECACHE; then
    echo "[2/3] Pre-caching road network -> $GRAPHML_PATH ..."
  python3 scripts/precache_graph.py --source "$SQLITE_PATH" --fallback "$XLSX_PATH" --output "$GRAPHML_PATH"
fi

if $RUN_APP; then
  echo "[3/3] Launching Streamlit app... (Ctrl+C to stop)"
  streamlit run app.py
fi
