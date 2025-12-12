#!/usr/bin/env bash
set -euo pipefail

# Run common setup steps in one go:
# 1) Import map.xlsx -> map.sqlite
# 2) Import R6.xlsx & H27.xls -> incidents.sqlite
# 3) Pre-cache road network to cache/matsuyama_drive.graphml
# 4) Pre-geocode incident addresses
# 5) (optional) Launch Streamlit app
#
# Usage:
#   ./scripts/run_all.sh               # do all steps and launch app
#   ./scripts/run_all.sh --no-app      # skip launching the app
#   ./scripts/run_all.sh --only-app    # only launch the app (assumes data/cache exist)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

XLSX_PATH="${STATIONS_XLSX_PATH:-map.xlsx}"
SQLITE_PATH="${STATIONS_DB_PATH:-map.sqlite}"
INCIDENTS_DB_PATH="${INCIDENTS_DB_PATH:-incidents.sqlite}"
GRAPHML_PATH="${GRAPHML_PATH:-cache/matsuyama_drive.graphml}"
GEOCODE_CACHE="${GEOCODE_CACHE:-cache/incident_geocode.parquet}"

RUN_IMPORT=true
RUN_PRECACHE=true
RUN_GEOCODE=true
RUN_APP=true

usage() {
  echo "Usage: $0 [--no-app] [--only-app] [--skip-geocode]" >&2
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
      RUN_GEOCODE=false
      RUN_APP=true
      shift
      ;;
    --skip-geocode)
      RUN_GEOCODE=false
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

STEP=1
TOTAL_STEPS=5

if $RUN_IMPORT; then
  if [[ ! -f "$XLSX_PATH" ]]; then
    echo "[ERROR] Station Excel not found: $XLSX_PATH" >&2
    exit 1
  fi
  echo "[$STEP/$TOTAL_STEPS] Importing $XLSX_PATH -> $SQLITE_PATH ..."
  python3 scripts/import_map_to_db.py --input "$XLSX_PATH" --output "$SQLITE_PATH"
  STEP=$((STEP + 1))
  
  echo "[$STEP/$TOTAL_STEPS] Importing incident data -> $INCIDENTS_DB_PATH ..."
  python3 scripts/import_incidents_to_db.py --output "$INCIDENTS_DB_PATH"
  STEP=$((STEP + 1))
fi

if $RUN_PRECACHE; then
  echo "[$STEP/$TOTAL_STEPS] Pre-caching road network -> $GRAPHML_PATH ..."
  python3 scripts/precache_graph.py --source "$SQLITE_PATH" --fallback "$XLSX_PATH" --output "$GRAPHML_PATH"
  STEP=$((STEP + 1))
fi

if $RUN_GEOCODE; then
  echo "[$STEP/$TOTAL_STEPS] Pre-geocoding incident addresses -> $GEOCODE_CACHE ..."
  python3 scripts/precompute_incident_geocode.py --output "$GEOCODE_CACHE" --sleep 1.0
  STEP=$((STEP + 1))
fi

if $RUN_APP; then
  echo "[$STEP/$TOTAL_STEPS] Launching Streamlit app... (Ctrl+C to stop)"
  streamlit run app.py
fi
