"""Convert map.xlsx into a lightweight SQLite database for faster reads.

Usage:
    python scripts/import_map_to_db.py --input map.xlsx --output map.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def import_excel_to_sqlite(excel_path: Path, sqlite_path: Path, table: str = "stations") -> None:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)

    print(f"Imported {len(df)} rows into {sqlite_path} (table: {table}).")


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
