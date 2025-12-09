"""Convert map.xlsx into a lightweight SQLite database for faster reads.

Usage:
    python scripts/import_map_to_db.py --input map.xlsx --output map.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


def import_excel_to_sqlite(excel_path: Path, sqlite_path: Path, table: str = "stations") -> None:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print("Loading Excel...")
    df = pd.read_excel(excel_path)
    total = len(df)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunked write with simple progress bar
    chunk_size = max(200, min(2000, total // 10 or 200))
    bar_width = 28

    def update_bar(done: int) -> None:
        frac = min(1.0, done / max(1, total))
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(frac * 100)
        sys.stdout.write(f"\r[{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()
        if frac >= 1.0:
            sys.stdout.write("\n")

    with sqlite3.connect(sqlite_path) as conn:
        for start in range(0, total, chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            chunk.to_sql(table, conn, if_exists="append" if start else "replace", index=False, method="multi")
            update_bar(min(start + len(chunk), total))

    print(f"Imported {total} rows into {sqlite_path} (table: {table}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Excel station data into SQLite")
    parser.add_argument("--input", default="map.xlsx", help="Source Excel file (default: map.xlsx)")
    parser.add_argument("--output", default="map.sqlite", help="Destination SQLite file (default: map.sqlite)")
    parser.add_argument("--table", default="stations", help="Table name to create/replace (default: stations)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_excel_to_sqlite(Path(args.input), Path(args.output), table=args.table)


if __name__ == "__main__":
    main()
