"""Generate a folium map with isochrones overlaid on a Google Maps-like basemap."""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, MultiPoint

ox.settings.use_cache = True
GRAPHML_PATH = Path("cache/ehime_drive.graphml")
GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
STATIONS_DB_PATH = Path("map.sqlite")


def load_or_build_graph(north: float, south: float, east: float, west: float) -> nx.MultiDiGraph:
    """Return cached road network when available, otherwise download and cache it."""
    if GRAPHML_PATH.exists():
        print("キャッシュ済みの道路ネットワークを再利用します。")
        return ox.load_graphml(GRAPHML_PATH)

    print("道路データを準備中...（初回取得は時間がかかります）")
    bbox = (north, south, east, west)
    try:
        graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")
    except ValueError as exc:
        if "no graph nodes" not in str(exc).lower():
            raise
        print("指定範囲に道路ノードが見つからなかったため、愛媛県全域データにフォールバックします。")
        graph = ox.graph_from_place("Ehime, Japan", network_type="drive")

    ox.save_graphml(graph, GRAPHML_PATH)
    print("道路ネットワークを保存しました。次回以降は高速に読み込めます。")
    return graph


def compute_isochrones(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    trip_times: list[int],
    progress_cb=None,
) -> gpd.GeoDataFrame:
    """Build convex-hull polygons using single-source Dijkstra per station for speed."""
    iso_records: list[dict] = []

    xs = stations["経度"].to_list()
    ys = stations["緯度"].to_list()
    trip_times_sorted = sorted(trip_times)
    max_radius = trip_times_sorted[-1] * 60 if trip_times_sorted else 0

    # Vectorized nearest node lookup
    try:
        center_nodes = ox.distance.nearest_nodes(graph, xs, ys)
    except Exception:
        center_nodes = [ox.distance.nearest_nodes(graph, x, y) for x, y in zip(xs, ys)]

    node_xy = {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}

    total = len(center_nodes)

    for idx, (row, center_node) in enumerate(zip(stations.itertuples(index=False), center_nodes)):
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
            iso_records.append({"name": row.略称, "time": minutes, "geometry": hull})

        if progress_cb:
            progress_cb((idx + 1) / total)

    if not iso_records:
        raise RuntimeError("到達圏ポリゴンを生成できませんでした。入力データを確認してください。")

    return gpd.GeoDataFrame(iso_records, crs="EPSG:4326")


def load_stations(db_path: Path = STATIONS_DB_PATH, excel_path: str = "map.xlsx") -> gpd.GeoDataFrame:
    """Load station data from SQLite when available, otherwise fallback to Excel."""
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM stations", conn)
    else:
        df = pd.read_excel(excel_path)

    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def render_map(
    isochrones: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    output_html: Path,
    tiles: str = "CartoDB Positron",
) -> None:
    """Render an interactive folium map styled similarly to Google Maps."""
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=tiles)

    color_map = {5: "#ff6b6b", 10: "#4361ee"}
    for minutes, color in color_map.items():
        layer = isochrones[isochrones["time"] == minutes]
        if layer.empty:
            continue

        folium.GeoJson(
            data=layer.__geo_interface__,
            name=f"{minutes}分圏",
            style_function=lambda _feature, c=color, m=minutes: {
                "fillColor": c,
                "color": c,
                "weight": 1.2,
                "opacity": 0.8,
                "fillOpacity": 0.25 if m == 10 else 0.45,
            },
            tooltip=folium.GeoJsonTooltip(fields=["name", "time"], aliases=["拠点", "到達時間(分)"]),
        ).add_to(fmap)

    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row["緯度"], row["経度"]],
            radius=6,
            color="#1f1f1f",
            weight=2,
            fill=True,
            fill_color="#f6bd60",
            fill_opacity=0.9,
            popup=f"{row['略称']}",
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(output_html)
    print(f"インタラクティブマップを {output_html.resolve()} に出力しました。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate folium isochrone map")
    parser.add_argument(
        "--stations",
        nargs="+",
        help="略称で指定した拠点のみを描画（例: --stations 松山中央 新居浜）",
    )
    parser.add_argument(
        "--output",
        default="isochrone_map.html",
        help="出力するHTMLファイル名 (default: isochrone_map.html)",
    )
    return parser.parse_args()


def _make_progress_printer(total_steps: int):
    """Lightweight ASCII progress bar for CLI usage without extra deps."""
    bar_width = 28

    def _update(fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(fraction * 100)
        sys.stdout.write(f"\r[{bar}] {percent:3d}% ({int(fraction*total_steps)}/{total_steps})")
        sys.stdout.flush()
        if fraction >= 1.0:
            sys.stdout.write("\n")

    return _update


def main() -> None:
    args = parse_args()
    stations = load_stations(db_path=STATIONS_DB_PATH, excel_path="map.xlsx")

    if args.stations:
        selected = stations["略称"].isin(args.stations)
        missing = [name for name in args.stations if name not in set(stations["略称"])]
        if missing:
            print(f"存在しない拠点名: {', '.join(missing)}")
        stations = stations[selected].copy()
        if stations.empty:
            raise ValueError("指定した拠点がデータに存在しません。map.xlsxを確認してください。")
        print(f"{len(stations)}件の拠点のみで到達圏を計算します。")

    west, south, east, north = stations.total_bounds
    padding_deg = 0.1
    north += padding_deg
    south -= padding_deg
    east += padding_deg
    west -= padding_deg

    graph = load_or_build_graph(north=north, south=south, east=east, west=west)
    if "travel_time" not in next(iter(graph.edges(data=True)))[2]:
        graph = ox.add_edge_speeds(graph, hwy_speeds={
            "residential": 30,
            "secondary": 40,
            "tertiary": 40,
            "primary": 50,
            "motorway": 80,
        })
        graph = ox.add_edge_travel_times(graph)
        ox.save_graphml(graph, GRAPHML_PATH)

    trip_times = [5, 10]
    print("到達圏を計算中...")
    progress = _make_progress_printer(len(stations))
    isochrones = compute_isochrones(graph, stations, trip_times, progress_cb=progress)
    render_map(isochrones, stations, output_html=Path(args.output))


if __name__ == "__main__":
    main()
