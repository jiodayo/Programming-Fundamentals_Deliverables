"""Streamlit app that visualizes EMS isochrones on a Google-like map.

$ streamlit run app.py で実行
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
from shapely.geometry import Point

ox.settings.use_cache = True
GRAPHML_PATH = Path("cache/ehime_drive.graphml")
GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)


def graph_data_version() -> float:
    """Return a timestamp that reflects the cached graph version."""
    return GRAPHML_PATH.stat().st_mtime if GRAPHML_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_station_data(filepath: str) -> gpd.GeoDataFrame:
    df = pd.read_excel(filepath)
    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


@st.cache_resource(show_spinner=False)
def load_graph_cached(bbox: tuple[float, float, float, float]) -> nx.MultiDiGraph:
    north, south, east, west = bbox
    if GRAPHML_PATH.exists():
        return ox.load_graphml(GRAPHML_PATH)

    print("道路データを準備中...（初回取得は時間がかかります）")
    try:
        graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")
    except ValueError as exc:
        if "no graph nodes" not in str(exc).lower():
            raise
        graph = ox.graph_from_place("Ehime, Japan", network_type="drive")

    ox.save_graphml(graph, GRAPHML_PATH)
    return graph


def compute_isochrones(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    trip_times: Iterable[int],
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    for _, row in stations.iterrows():
        center_point = (row["緯度"], row["経度"])
        try:
            center_node = ox.distance.nearest_nodes(graph, center_point[1], center_point[0])
        except Exception as err:
            st.warning(f"{row['略称']} 付近の道路ノード取得に失敗: {err}")
            continue

        for minutes in trip_times:
            subgraph = nx.ego_graph(graph, center_node, radius=minutes * 60, distance="travel_time")
            node_points = [Point((data["x"], data["y"])) for _, data in subgraph.nodes(data=True)]
            if not node_points:
                continue

            reachable = gpd.GeoSeries(node_points).union_all().convex_hull
            records.append({"name": row["略称"], "time": minutes, "geometry": reachable})

    if not records:
        raise RuntimeError("到達圏ポリゴンを生成できませんでした。条件を見直してください。")
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


@st.cache_data(show_spinner=False)
def precompute_isochrones(
    station_df: pd.DataFrame,
    trip_times: tuple[int, ...],
    graph_version: float,
) -> gpd.GeoDataFrame:
    """Cache-intensive isochrone computation so UI updates stay responsive."""
    geometry = gpd.points_from_xy(station_df["経度"], station_df["緯度"])
    stations = gpd.GeoDataFrame(station_df, geometry=geometry, crs="EPSG:4326")

    graph = ox.load_graphml(GRAPHML_PATH)
    graph = ox.add_edge_speeds(graph, hwy_speeds={
        "residential": 30,
        "secondary": 40,
        "tertiary": 40,
        "primary": 50,
        "motorway": 80,
    })
    graph = ox.add_edge_travel_times(graph)
    return compute_isochrones(graph, stations, trip_times)


def render_map_html(
    isochrones: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    tiles: str = "CartoDB Positron",
) -> str:
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=tiles)

    color_map = {5: "#ff6b6b", 10: "#4361ee", 15: "#2ec4b6", 20: "#f4a261"}
    for minutes in sorted({*isochrones["time"]}):
        layer = isochrones[isochrones["time"] == minutes]
        if layer.empty:
            continue
        color = color_map.get(minutes, "#4a4a4a")
        folium.GeoJson(
            data=layer.__geo_interface__,
            name=f"{minutes}分圏",
            style_function=lambda _feature, c=color, m=minutes: {
                "fillColor": c,
                "color": c,
                "weight": 1.2,
                "opacity": 0.8,
                "fillOpacity": 0.25 if m >= 10 else 0.45,
            },
            tooltip=folium.GeoJsonTooltip(fields=["name", "time"], aliases=["拠点", "到達時間(分)"])
        ).add_to(fmap)

    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row["緯度"], row["経度"]],
            radius=7,
            color="#1f1f1f",
            weight=2,
            fill=True,
            fill_color="#f6bd60",
            fill_opacity=0.9,
            popup=f"{row['略称']}",
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap.get_root().render()


def main() -> None:
    st.set_page_config(page_title="愛媛救急車 到達圏ビューア", layout="wide")
    st.title("🚑 愛媛県 救急車到達圏ビューア")
    st.caption("map.xlsx を元に消防署の到達圏を可視化します。")

    stations = load_station_data("map.xlsx")
    stations_plain = stations.drop(columns="geometry").copy()
    station_names = sorted(stations["略称"].unique())
    trip_options = [5, 10, 15, 20]

    col_left, col_right = st.columns([2, 1])
    with col_left:
        selected_names = st.multiselect(
            "表示する消防署",
            options=station_names,
            default=station_names,
            help="複数選択で到達圏を比較できます。",
        )
    with col_right:
        selected_times = st.multiselect(
            "到達時間 (分)",
            options=trip_options,
            default=[5, 10],
        )

    if not selected_names:
        st.warning("少なくとも1つの消防署を選択してください。")
        st.stop()
    if not selected_times:
        st.warning("少なくとも1つの到達時間を選択してください。")
        st.stop()

    filtered = stations[stations["略称"].isin(selected_names)].copy()

    padding_deg = 0.1
    west_all, south_all, east_all, north_all = stations.total_bounds
    bbox = (north_all + padding_deg, south_all - padding_deg, east_all + padding_deg, west_all - padding_deg)

    with st.spinner("道路ネットワークを読み込み中..."):
        load_graph_cached(bbox)

    graph_version = graph_data_version()

    with st.spinner("到達圏データを準備しています..."):
        all_isochrones = precompute_isochrones(
            station_df=stations_plain,
            trip_times=tuple(trip_options),
            graph_version=graph_version,
        )

    display_isochrones = all_isochrones[
        (all_isochrones["name"].isin(selected_names)) &
        (all_isochrones["time"].isin(selected_times))
    ].copy()

    if display_isochrones.empty:
        st.error("選択条件に合致する到達圏がありません。")
        st.stop()

    html_map = render_map_html(display_isochrones, filtered)
    st.components.v1.html(html_map, height=720)

    st.info("アプリを終了するには、実行中のターミナルで Ctrl+C を押してください。")


if __name__ == "__main__":
    main()
