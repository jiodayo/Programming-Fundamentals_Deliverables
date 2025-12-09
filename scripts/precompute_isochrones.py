"""Precompute all isochrone polygons and save to Parquet for instant loading.

Usage:
    python scripts/precompute_isochrones.py \
        --times 5 10 15 20 \
        --source map.sqlite \
        --fallback map.xlsx \
        --graph cache/ehime_drive.graphml \
        --output cache/isochrones.parquet

Notes:
- Requires graph to have travel_time on edges. If not, speeds are added.
- Output contains columns: name, time, geometry (EPSG:4326).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPoint

ox.settings.use_cache = True


def load_stations(source: Path, fallback: Path) -> gpd.GeoDataFrame:
    if source.exists() and source.suffix == ".sqlite":
        with sqlite3.connect(source) as conn:
            df = pd.read_sql("SELECT * FROM stations", conn)
    elif fallback.exists():
        df = pd.read_excel(fallback)
    else:
        raise FileNotFoundError("No station data found: provide --source or --fallback")

    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def ensure_graph(graph_path: Path, stations: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        west, south, east, north = stations.total_bounds
        padding = 0.1
        bbox = (north + padding, south - padding, east + padding, west - padding)
        print("Downloading road network (first run may take time)...")
        try:
            graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")
        except ValueError as exc:
            if "no graph nodes" not in str(exc).lower():
                raise
            print("指定範囲に道路ノードが見つからなかったため、愛媛県全域データにフォールバックします。")
            graph = ox.graph_from_place("Ehime, Japan", network_type="drive")
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, graph_path)

    if "travel_time" not in next(iter(graph.edges(data=True)))[2]:
        graph = ox.add_edge_speeds(graph, hwy_speeds={
            "residential": 30,
            "secondary": 40,
            "tertiary": 40,
            "primary": 50,
            "motorway": 80,
        })
        graph = ox.add_edge_travel_times(graph)
        ox.save_graphml(graph, graph_path)

    return graph


def compute_isochrones(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    trip_times: Iterable[int],
) -> gpd.GeoDataFrame:
    xs = stations["経度"].to_list()
    ys = stations["緯度"].to_list()
    trip_times_sorted = sorted(trip_times)
    max_radius = trip_times_sorted[-1] * 60 if trip_times_sorted else 0

    try:
        center_nodes = ox.distance.nearest_nodes(graph, xs, ys)
    except Exception:
        center_nodes = [ox.distance.nearest_nodes(graph, x, y) for x, y in zip(xs, ys)]

    node_xy = {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}
    total = len(center_nodes)

    records: list[dict] = []
    bar_width = 28

    def update_bar(done: int) -> None:
        frac = done / max(1, total)
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(frac * 100)
        sys.stdout.write(f"\r[{bar}] {percent:3d}% ({done}/{total})")
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")

    for idx, (row, center_node) in enumerate(zip(stations.itertuples(index=False), center_nodes), start=1):
        lengths = nx.single_source_dijkstra_path_length(
            graph,
            center_node,
            cutoff=max_radius,
            weight="travel_time",
        )
        for minutes in trip_times_sorted:
            cutoff = minutes * 60
            reachable_nodes = [nid for nid, dist in lengths.items() if dist <= cutoff]
            if not reachable_nodes:
                continue
            pts = [node_xy[nid] for nid in reachable_nodes if nid in node_xy]
            if not pts:
                continue
            hull = MultiPoint(pts).convex_hull
            records.append({"name": row.略称, "time": minutes, "geometry": hull})
        update_bar(idx)

    if not records:
        raise RuntimeError("到達圏ポリゴンを生成できませんでした。入力データを確認してください。")

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute all isochrone polygons")
    parser.add_argument("--times", nargs="+", type=int, default=[5, 10, 15, 20], help="Minutes (e.g., 5 10 15 20)")
    parser.add_argument("--source", default="map.sqlite", help="Station data SQLite")
    parser.add_argument("--fallback", default="map.xlsx", help="Fallback Excel path")
    parser.add_argument("--graph", default="cache/ehime_drive.graphml", help="GraphML cache path")
    parser.add_argument("--output", default="cache/isochrones.parquet", help="Output Parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stations = load_stations(Path(args.source), Path(args.fallback))
    graph = ensure_graph(Path(args.graph), stations)
    print("Computing isochrones for all stations...")
    iso = compute_isochrones(graph, stations, args.times)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iso.to_parquet(out_path)
    print(f"Saved {len(iso)} polygons to {out_path}")


if __name__ == "__main__":
    main()
