"""Pre-download and cache the road network graph so the app starts fast.

Usage:
    python scripts/precache_graph.py \
        --source map.sqlite \
        --fallback map.xlsx \
        --output cache/ehime_drive.graphml

Notes:
- If --source exists (SQLite), it is used to derive the bounding box; otherwise --fallback Excel is used.
- The script pads the bounding box slightly to ensure coverage and stores travel_time edges for reuse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

DEFAULT_OUTPUT = Path("cache/ehime_drive.graphml")


def load_stations(source: Path, fallback: Path) -> gpd.GeoDataFrame:
    if source.exists() and source.suffix == ".sqlite":
        import sqlite3

        with sqlite3.connect(source) as conn:
            df = pd.read_sql("SELECT * FROM stations", conn)
    elif fallback.exists():
        df = pd.read_excel(fallback)
    else:
        raise FileNotFoundError("No station data found: provide --source or --fallback")

    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def download_and_cache_graph(stations: gpd.GeoDataFrame, output: Path) -> None:
    west, south, east, north = stations.total_bounds
    padding_deg = 0.1
    north += padding_deg
    south -= padding_deg
    east += padding_deg
    west -= padding_deg

    output.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading road network (may take a while on first run)...")
    try:
        graph = ox.graph_from_bbox((north, south, east, west), network_type="drive")
    except ValueError as exc:
        if "no graph nodes" not in str(exc).lower():
            raise
        print("指定範囲に道路ノードが見つからなかったため、愛媛県全域データにフォールバックします。")
        graph = ox.graph_from_place("Ehime, Japan", network_type="drive")

    # Store speed and travel time so runtime doesn’t have to recompute.
    graph = ox.add_edge_speeds(graph, hwy_speeds={
        "residential": 30,
        "secondary": 40,
        "tertiary": 40,
        "primary": 50,
        "motorway": 80,
    })
    graph = ox.add_edge_travel_times(graph)

    ox.save_graphml(graph, output)
    print(f"Cached graph saved to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-cache the road network graph")
    parser.add_argument("--source", default="map.sqlite", help="Station data SQLite (default: map.sqlite)")
    parser.add_argument("--fallback", default="map.xlsx", help="Fallback Excel data (default: map.xlsx)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="GraphML output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_stations(Path(args.source), Path(args.fallback))
    download_and_cache_graph(stations, Path(args.output))


if __name__ == "__main__":
    main()
