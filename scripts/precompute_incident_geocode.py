"""Precompute geocoded incident locations from R6.xlsx and H27.xls and cache them.

Usage examples:
    python scripts/precompute_incident_geocode.py --region 愛媛県 --sleep 1.0

Both R6.xlsx and H27.xls are processed if present. Only addresses not yet in
the cache are geocoded. Results are stored in Parquet (columns: address, lat, lon)
and reused by the Streamlit app.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import osmnx as ox
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geocode incident addresses with caching")
    parser.add_argument("--r6", default="R6.xlsx", help="R6 incident file (default: R6.xlsx)")
    parser.add_argument("--h27", default="H27.xls", help="H27 incident file (default: H27.xls)")
    parser.add_argument("--db", default="incidents.sqlite", help="SQLite database (preferred source)")
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


def load_addresses_from_db(db_path: Path) -> list[str]:
    """Load addresses from SQLite database (fast)."""
    import sqlite3
    if not db_path.exists():
        return []
    
    addresses = []
    with sqlite3.connect(db_path) as conn:
        # Check available tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        if "incidents_r6" in tables:
            df = pd.read_sql("SELECT DISTINCT 出動場所 FROM incidents_r6 WHERE 出動場所 IS NOT NULL", conn)
            addresses.extend(df["出動場所"].astype(str).tolist())
        
        if "incidents_h27" in tables:
            df = pd.read_sql("SELECT DISTINCT 出動場所 FROM incidents_h27 WHERE 出動場所 IS NOT NULL", conn)
            addresses.extend(df["出動場所"].astype(str).tolist())
    
    return addresses


def load_addresses_r6(path: Path) -> list[str]:
    """Load addresses from R6 format Excel (fallback)."""
    if not path.exists():
        return []
    df = pd.read_excel(path, usecols=["出動場所"])
    return df["出動場所"].dropna().astype(str).unique().tolist()


def load_addresses_h27(path: Path) -> list[str]:
    """Load addresses from H27 format Excel (fallback, different column name)."""
    if not path.exists():
        return []
    df = pd.read_excel(path, usecols=["出場場所-1"])
    return df["出場場所-1"].dropna().astype(str).unique().tolist()


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["address", "lat", "lon"])
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["address", "lat", "lon"])


def geocode_missing(addresses: Iterable[str], region: str, sleep_sec: float) -> pd.DataFrame:
    records: list[dict] = []
    total = len(list(addresses))
    # Re-materialize iterator so we can iterate twice for counting and processing
    addresses = list(addresses)
    if total == 0:
        return pd.DataFrame(columns=["address", "lat", "lon"])

    bar_width = 28
    def _update_bar(done: int) -> None:
        frac = done / total
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\r[{bar}] {done}/{total}", end="")

    for idx, addr in enumerate(addresses, start=1):
        query = f"{region} {addr}" if region else addr
        try:
            lat, lon = ox.geocode(query)
            records.append({"address": addr, "lat": lat, "lon": lon})
        except Exception as exc:
            records.append({"address": addr, "lat": None, "lon": None})
        _update_bar(idx)
        time.sleep(max(0.0, sleep_sec))

    print()  # newline after bar
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    r6_path = Path(args.r6)
    h27_path = Path(args.h27)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer SQLite for speed, fallback to Excel
    if db_path.exists():
        print(f"Loading addresses from SQLite ({db_path})...")
        all_addresses = sorted(set(load_addresses_from_db(db_path)))
        print(f"Found {len(all_addresses)} unique addresses")
    else:
        # Fallback to Excel
        print("SQLite not found, loading from Excel files...")
        addresses_r6 = load_addresses_r6(r6_path)
        addresses_h27 = load_addresses_h27(h27_path)
        all_addresses = sorted(set(addresses_r6 + addresses_h27))
        print(f"Addresses found: R6={len(addresses_r6)}, H27={len(addresses_h27)}, unique={len(all_addresses)}")

    cache = load_cache(output_path)

    cached_addrs = set(cache["address"].tolist())
    missing = [a for a in all_addresses if a not in cached_addrs]
    if args.limit is not None:
        missing = missing[: args.limit]

    print(f"Already cached: {len(cached_addrs)}, to geocode: {len(missing)}")
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
