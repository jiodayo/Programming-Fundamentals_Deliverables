"""Streamlit app that visualizes EMS isochrones on a Google-like map.

$ streamlit run app.py で実行
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Iterable, Callable
import re

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
from shapely.geometry import Point
from shapely.geometry import MultiPoint

ox.settings.use_cache = True
GRAPHML_PATH = Path("cache/ehime_drive.graphml")
GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
STATIONS_DB_PATH = Path("map.sqlite")
ISOCHRONE_CACHE_PATH = Path("cache/isochrones.parquet")
GEOCODE_CACHE_PATH = Path("cache/incident_geocode.parquet")


def graph_data_version() -> float:
    """Return a timestamp that reflects the cached graph version."""
    return GRAPHML_PATH.stat().st_mtime if GRAPHML_PATH.exists() else 0.0


def station_data_version(db_path: Path = STATIONS_DB_PATH, excel_path: str = "map.xlsx") -> float:
    """Return mtime of the current station datasource for cache invalidation."""
    if db_path.exists():
        return db_path.stat().st_mtime
    return Path(excel_path).stat().st_mtime if Path(excel_path).exists() else 0.0


@st.cache_data(show_spinner=False)
def load_incident_data(excel_path: str = "R6.xlsx") -> pd.DataFrame:
    """Load incident records; keeps only rows with a valid 発生日時."""
    if not Path(excel_path).exists():
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(excel_path)
    df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
    df = df[df["覚知"].notna()].copy()
    df["date"] = df["覚知"].dt.date
    return df


def _load_geocode_cache(path: Path = GEOCODE_CACHE_PATH) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame(columns=["address", "lat", "lon"])
    return pd.DataFrame(columns=["address", "lat", "lon"])


def _save_geocode_cache(df: pd.DataFrame, path: Path = GEOCODE_CACHE_PATH) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        pass


def geocode_addresses(addresses: list[str], region_prefix: str = "愛媛県") -> pd.DataFrame:
    """Geocode addresses with osmnx + Nominatim and persist results locally."""
    cache = _load_geocode_cache().copy()
    seen = set(cache["address"].tolist())
    missing = [a for a in addresses if a not in seen]

    new_records: list[dict] = []
    for addr in missing:
        query = f"{region_prefix} {addr}" if region_prefix else addr
        try:
            lat, lon = ox.geocode(query)
            new_records.append({"address": addr, "lat": lat, "lon": lon})
        except Exception:
            new_records.append({"address": addr, "lat": None, "lon": None})

    if new_records:
        cache = pd.concat([cache, pd.DataFrame(new_records)], ignore_index=True)
        cache = cache.drop_duplicates(subset=["address"], keep="last")
        _save_geocode_cache(cache)

    return cache[cache["address"].isin(addresses)].copy()


@st.cache_data(show_spinner=False)
def load_station_data(
    db_path: Path,
    excel_path: str,
    source_mtime: float,
) -> gpd.GeoDataFrame:
    """Load station records from SQLite when available, otherwise fallback to Excel."""
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM stations", conn)
    else:
        df = pd.read_excel(excel_path)

    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


@st.cache_resource(show_spinner=False)
def load_graph_cached(bbox: tuple[float, float, float, float]) -> nx.MultiDiGraph:
    north, south, east, west = bbox
    if GRAPHML_PATH.exists():
        graph = ox.load_graphml(GRAPHML_PATH)
    else:
        print("道路データを準備中...（初回取得は時間がかかります）")
        try:
            graph = ox.graph_from_bbox(bbox=bbox, network_type="drive")
        except ValueError as exc:
            if "no graph nodes" not in str(exc).lower():
                raise
            graph = ox.graph_from_place("Ehime, Japan", network_type="drive")
        ox.save_graphml(graph, GRAPHML_PATH)

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

    return graph


def compute_isochrones(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    trip_times: Iterable[int],
    progress_cb: Callable[[float], None] | None = None,
) -> gpd.GeoDataFrame:
    records: list[dict] = []

    # Vectorized nearest-node lookup to avoid per-row KDTree rebuilds
    xs = stations["経度"].to_list()
    ys = stations["緯度"].to_list()
    try:
        center_nodes = ox.distance.nearest_nodes(graph, xs, ys)
    except Exception:
        # Fallback to per-point lookup if vectorized call fails
        center_nodes = [ox.distance.nearest_nodes(graph, x, y) for x, y in zip(xs, ys)]

    total = len(center_nodes)

    # Pre-extract node coordinates to avoid repeated attribute lookups
    node_xy = {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}
    trip_times_sorted = sorted(trip_times)
    max_radius = trip_times_sorted[-1] * 60 if trip_times_sorted else 0

    def _one_station(payload: tuple[int, tuple]) -> list[dict]:
        _idx, (row, center_node) = payload
        out: list[dict] = []

        # Single-source Dijkstra once up to the maximum requested time
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
            points = [node_xy[nid] for nid in reachable_nodes if nid in node_xy]
            if not points:
                continue
            hull = MultiPoint(points).convex_hull
            out.append({"name": row.略称, "time": minutes, "geometry": hull})

        return out

    with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 2))) as ex:  # type: ignore[name-defined]
        futures = [
            ex.submit(_one_station, payload)
            for payload in enumerate(zip(stations.itertuples(index=False), center_nodes))
        ]
        completed = 0
        for fut in as_completed(futures):
            records.extend(fut.result())
            completed += 1
            if progress_cb:
                progress_cb(completed / total)

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
    return compute_isochrones(graph, stations, trip_times)


@st.cache_data(show_spinner=False)
def load_precomputed_isochrones(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return gpd.read_parquet(path)


def append_virtual_stations(base: gpd.GeoDataFrame, virtuals: list[dict]) -> gpd.GeoDataFrame:
    """Append in-session virtual stations (lat/lon) to the loaded stations."""
    if not virtuals:
        return base

    df_new = pd.DataFrame(virtuals)
    geom = gpd.points_from_xy(df_new["経度"], df_new["緯度"])
    gdf_new = gpd.GeoDataFrame(df_new, geometry=geom, crs="EPSG:4326")

    # Ensure all columns exist and order matches base
    for col in base.columns:
        if col not in gdf_new.columns:
            gdf_new[col] = None
    gdf_new = gdf_new[base.columns]

    return pd.concat([base, gdf_new], ignore_index=True)


def render_map_html(
    isochrones: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    tiles: str = "CartoDB Positron",
) -> str:
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=tiles)

    # Softer palette for better visibility when overlapping
    color_map = {5: "#ff9e9e", 10: "#8aa5ff", 15: "#7dd8c6", 20: "#f7caa0"}
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
                "weight": 1.0,
                "opacity": 0.6,
                "fillOpacity": 0.18 if m >= 10 else 0.30,
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
    st.title("🚑 愛媛県 救急車ビューア")
    st.caption("map.xlsx で拠点到達圏、R6.xlsx で出動地点を可視化します。")

    stations = load_station_data(
        db_path=STATIONS_DB_PATH,
        excel_path="map.xlsx",
        source_mtime=station_data_version(),
    )

    if "virtual_stations" not in st.session_state:
        st.session_state["virtual_stations"] = []

    tab_iso, tab_inc = st.tabs(["到達圏", "出動地点 (R6)" ])

    with tab_iso:
        with st.expander("仮想消防署を追加（このセッションのみ）"):
            with st.form("virtual_station_form"):
                default_name = f"仮想署{len(st.session_state['virtual_stations']) + 1}"
                v_name = st.text_input("略称", value=default_name)
                v_lat = st.number_input("緯度", value=float(stations["緯度"].mean()))
                v_lon = st.number_input("経度", value=float(stations["経度"].mean()))
                submitted = st.form_submit_button("追加")
                if submitted:
                    st.session_state["virtual_stations"].append({
                        "略称": v_name.strip() or default_name,
                        "緯度": v_lat,
                        "経度": v_lon,
                    })
                    st.success(f"仮想消防署を追加: {v_name}")
            if st.button("仮想消防署をクリア", type="secondary"):
                st.session_state["virtual_stations"] = []
                st.info("仮想消防署をクリアしました。")

        has_virtual = bool(st.session_state["virtual_stations"])
        stations_view = append_virtual_stations(stations, st.session_state["virtual_stations"])
        station_names = sorted(stations_view["略称"].unique())
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

        filtered = stations_view[stations_view["略称"].isin(selected_names)].copy()

        padding_deg = 0.1
        west_all, south_all, east_all, north_all = stations_view.total_bounds
        bbox = (north_all + padding_deg, south_all - padding_deg, east_all + padding_deg, west_all - padding_deg)

        with st.spinner("道路ネットワークを読み込み中..."):
            graph = load_graph_cached(bbox)

        if ISOCHRONE_CACHE_PATH.exists() and not has_virtual:
            try:
                with st.spinner("到達圏キャッシュを読み込み中..."):
                    all_isochrones = load_precomputed_isochrones(ISOCHRONE_CACHE_PATH)
                display_isochrones = all_isochrones[
                    (all_isochrones["name"].isin(selected_names)) &
                    (all_isochrones["time"].isin(selected_times))
                ].copy()
            except Exception as exc:
                st.warning(f"事前計算キャッシュの読み込みに失敗したため再計算します: {exc}")
                display_isochrones = None
        else:
            display_isochrones = None

        if display_isochrones is None:
            with st.spinner("到達圏を計算しています..."):
                prog = st.progress(0)
                display_isochrones = compute_isochrones(
                    graph=graph,
                    stations=filtered,
                    trip_times=selected_times,
                    progress_cb=lambda p: prog.progress(int(p * 100)),
                )

        if display_isochrones.empty:
            st.error("選択条件に合致する到達圏がありません。")
            st.stop()

        html_map = render_map_html(display_isochrones, filtered)
        st.components.v1.html(html_map, height=720)

    with tab_inc:
        try:
            incidents = load_incident_data("R6.xlsx")
        except FileNotFoundError:
            st.error("R6.xlsx が見つかりません。ルートに配置してください。")
            st.stop()

        date_options = sorted(incidents["date"].unique())
        if not date_options:
            st.warning("R6.xlsx に日付データがありません。")
            st.stop()

        default_date = date_options[0]
        selected_date = st.selectbox(
            "表示する日付 (覚知日)",
            options=date_options,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            index=0,
        )

        day_inc = incidents[incidents["date"] == selected_date].copy()
        addr_series = day_inc["出動場所"].dropna().astype(str)
        addr_unique = sorted(addr_series.unique())

        st.write(f"{selected_date} の出動件数: {len(day_inc)} 件 (ユニーク地点 {len(addr_unique)} 箇所)")

        with st.spinner("住所をジオコーディングしています (キャッシュ利用) ..."):
            geo_df = geocode_addresses(addr_unique, region_prefix="愛媛県")

        merged = day_inc.merge(geo_df, left_on="出動場所", right_on="address", how="left")
        mapped = merged.dropna(subset=["lat", "lon"]).copy()
        missing_count = len(day_inc) - len(mapped)

        if mapped.empty:
            st.error("この日の地点をジオコーディングできませんでした。")
            st.stop()

        st.write(f"地図にプロットできた件数: {len(mapped)} / {len(day_inc)} (未特定 {missing_count} 件)")

        center_lat = mapped["lat"].mean()
        center_lon = mapped["lon"].mean()
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB Positron")

        # Softer color by weekday to help visually group clusters
        weekday_colors = {
            "月": "#f94144",
            "火": "#f3722c",
            "水": "#f9c74f",
            "木": "#90be6d",
            "金": "#43aa8b",
            "土": "#577590",
            "日": "#9d4edd",
        }

        for _, row in mapped.iterrows():
            wk = str(row.get("曜日", "?"))
            color = weekday_colors.get(wk, "#4a4a4a")
            label_time = row["覚知"].strftime("%H:%M") if not pd.isna(row.get("覚知")) else "--:--"
            popup = f"{row.get('出動隊', '不明')} | {label_time} | {row.get('搬送区分(事案)', '')}"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=1.0,
                popup=popup,
            ).add_to(fmap)

        st.components.v1.html(fmap.get_root().render(), height=720)

    st.info("アプリを終了するには、実行中のターミナルで Ctrl+C を押してください。")


if __name__ == "__main__":
    main()
