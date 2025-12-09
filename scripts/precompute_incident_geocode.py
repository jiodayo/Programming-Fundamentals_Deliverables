"""Precompute geocoded incident locations from R6.xlsx and cache them.

Usage examples:
    python scripts/precompute_incident_geocode.py --input R6.xlsx \
        --output cache/incident_geocode.parquet --region 愛媛県 --sleep 1.0

Only addresses not yet present in the cache are geocoded. Results are stored in
Parquet (columns: address, lat, lon) and reused by the Streamlit app.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import osmnx as ox
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geocode R6 incident addresses with caching")
    parser.add_argument("--input", default="R6.xlsx", help="Incident Excel file (default: R6.xlsx)")
    parser.add_argument(
        "--output", default="cache/incident_geocode.parquet", help="Cache output path (parquet)"
    )
    parser.add_argument(
        "--region",
        default="愛媛県",
        help="Region prefix added before each address to improve hit rate (default: 愛媛県)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between requests to respect Nominatim rate limits (default: 1.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of new addresses to geocode (useful for incremental runs)",
    )
    return parser.parse_args()


def load_addresses(path: Path) -> list[str]:
    df = pd.read_excel(path, usecols=["出動場所"])
    return sorted(df["出動場所"].dropna().astype(str).unique())


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["address", "lat", "lon"])
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["address", "lat", "lon"])


def geocode_missing(addresses: Iterable[str], region: str, sleep_sec: float) -> pd.DataFrame:
    records: list[dict] = []
    for idx, addr in enumerate(addresses, start=1):
        query = f"{region} {addr}" if region else addr
        try:
            lat, lon = ox.geocode(query)
            records.append({"address": addr, "lat": lat, "lon": lon})
            print(f"[{idx}] OK {addr} -> ({lat:.6f}, {lon:.6f})")
        except Exception as exc:
            print(f"[{idx}] FAIL {addr}: {exc}")
            records.append({"address": addr, "lat": None, "lon": None})
        time.sleep(max(0.0, sleep_sec))
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    addresses = load_addresses(input_path)
    cache = load_cache(output_path)

    cached_addrs = set(cache["address"].tolist())
    missing = [a for a in addresses if a not in cached_addrs]
    if args.limit is not None:
        missing = missing[: args.limit]

    print(f"Unique addresses: {len(addresses)} (cached: {len(cache)}, to geocode: {len(missing)})")
    if not missing:
        print("Nothing to geocode. Cache is up to date.")
        return

    new_df = geocode_missing(missing, region=args.region, sleep_sec=args.sleep)
    combined = pd.concat([cache, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["address"], keep="last")

    combined.to_parquet(output_path, index=False)
    print(f"Saved cache to {output_path} with {len(combined)} rows")


if __name__ == "__main__":
    main()
