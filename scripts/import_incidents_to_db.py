"""Import incident data (R6.xlsx, H27.xls) into SQLite for faster reads.

Usage:
    python scripts/import_incidents_to_db.py --output incidents.sqlite

This script:
1. Reads R6.xlsx and H27.xls
2. Normalizes column names to a common schema
3. Stores both in a single SQLite database with tables: incidents_r6, incidents_h27
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


def normalize_r6(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize R6 data to common schema."""
    df = df.copy()
    df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
    df = df[df["覚知"].notna()].copy()
    df["date"] = df["覚知"].dt.date.astype(str)
    df["year"] = df["覚知"].dt.year
    
    # Keep relevant columns and rename for consistency
    keep_cols = ["覚知", "date", "year", "出動場所", "出動隊", "曜日", "搬送区分(事案)"]
    result = df[[c for c in keep_cols if c in df.columns]].copy()
    return result


def normalize_h27(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize H27 data to common schema (different column names)."""
    df = df.copy()
    
    # Build datetime from separate columns
    df["覚知"] = pd.to_datetime(
        df["覚知日付(年)"].astype(str) + "-" +
        df["覚知日付(月)"].astype(str).str.zfill(2) + "-" +
        df["覚知日付(日)"].astype(str).str.zfill(2) + " " +
        df["覚知時刻(時)"].astype(str).str.zfill(2) + ":" +
        df["覚知時刻(分)"].astype(str).str.zfill(2) + ":" +
        df["覚知時刻(秒)"].fillna(0).astype(int).astype(str).str.zfill(2),
        errors="coerce"
    )
    df = df[df["覚知"].notna()].copy()
    df["date"] = df["覚知"].dt.date.astype(str)
    df["year"] = df["覚知"].dt.year
    
    # Normalize column names to match R6
    df["出動場所"] = df["出場場所-1"]
    df["出動隊"] = df["出場隊名"]
    df["曜日"] = df["覚知曜日名"]
    
    keep_cols = ["覚知", "date", "year", "出動場所", "出動隊", "曜日"]
    result = df[[c for c in keep_cols if c in df.columns]].copy()
    return result


def import_to_sqlite(df: pd.DataFrame, sqlite_path: Path, table: str) -> int:
    """Import DataFrame to SQLite with progress bar."""
    total = len(df)
    chunk_size = max(200, min(2000, total // 10 or 200))
    bar_width = 28

    def update_bar(done: int) -> None:
        frac = min(1.0, done / max(1, total))
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(frac * 100)
        sys.stdout.write(f"\r  [{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write("\n")

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as conn:
        for start in range(0, total, chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            chunk.to_sql(table, conn, if_exists="append" if start else "replace", index=False, method="multi")
            update_bar(min(start + len(chunk), total))

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Import incident data into SQLite")
    parser.add_argument("--r6", default="R6.xlsx", help="R6 incident file (default: R6.xlsx)")
    parser.add_argument("--h27", default="H27.xls", help="H27 incident file (default: H27.xls)")
    parser.add_argument("--output", default="incidents.sqlite", help="Output SQLite file")
    args = parser.parse_args()

    sqlite_path = Path(args.output)
    
    # Remove existing database to start fresh
    if sqlite_path.exists():
        sqlite_path.unlink()
        print(f"Removed existing {sqlite_path}")

    # Import R6
    r6_path = Path(args.r6)
    if r6_path.exists():
        print(f"Loading {r6_path}...")
        df_r6 = pd.read_excel(r6_path)
        df_r6 = normalize_r6(df_r6)
        print(f"Importing R6 ({len(df_r6)} rows)...")
        import_to_sqlite(df_r6, sqlite_path, "incidents_r6")
    else:
        print(f"[SKIP] {r6_path} not found")

    # Import H27
    h27_path = Path(args.h27)
    if h27_path.exists():
        print(f"Loading {h27_path}...")
        df_h27 = pd.read_excel(h27_path)
        df_h27 = normalize_h27(df_h27)
        print(f"Importing H27 ({len(df_h27)} rows)...")
        import_to_sqlite(df_h27, sqlite_path, "incidents_h27")
    else:
        print(f"[SKIP] {h27_path} not found")

    # Create indices for faster queries
    with sqlite3.connect(sqlite_path) as conn:
        print("Creating indices...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_r6_date ON incidents_r6(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_r6_place ON incidents_r6(出動場所)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_h27_date ON incidents_h27(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_h27_place ON incidents_h27(出動場所)")

    print(f"Done! Database saved to {sqlite_path}")


if __name__ == "__main__":
    main()
