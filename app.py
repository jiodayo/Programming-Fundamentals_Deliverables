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
import logging

# WebSocketClosedErrorのログを抑制（Streamlit再描画時に発生する無害なエラー）
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

# Traffic-aware isochrones
from traffic_analysis import (
    load_delay_factors,
    get_delay_factor,
    TIME_SLOT_LABELS,
    DOW_LABELS,
    DELAY_FACTORS_PATH,
)

# リソース（救急車台数）情報 - R6（2024年）: R4〜R6で3隊増隊後
STATION_RESOURCES_R6 = {
    "東消防署": 4,
    "中央消防署": 3,
    "西消防署": 3,
    "南消防署": 3,
    "城北支署": 1,
    "城東支署": 1,
    "西部支署": 1,
    "東部支署": 1,
    "北条支署": 1,
    "湯山出張所": 1,
    "久谷出張所": 1,
    "消防局": 1,
    "WS": 1,
}
# 合計: 22台

# リソース（救急車台数）情報 - H27（2015年）: 増隊前（推定）
# R4〜R6で3隊増隊、H25時点ではWS常駐隊なし
STATION_RESOURCES_H27_DEFAULT = {
    "東消防署": 3,  # 推定
    "中央消防署": 2,  # 推定
    "西消防署": 2,  # 推定
    "南消防署": 3,
    "城北支署": 1,
    "城東支署": 1,
    "西部支署": 1,
    "東部支署": 1,
    "北条支署": 1,
    "湯山出張所": 1,
    "久谷出張所": 1,
    "消防局": 1,
    "WS": 0,  # H25時点では常駐隊なし
}
# 合計: 18台（R6より4台少ない・推定値）

# 後方互換性のためのエイリアス
STATION_RESOURCES = STATION_RESOURCES_R6
DEFAULT_AMBULANCES = 1

ox.settings.use_cache = True
GRAPHML_PATH = Path("cache/matsuyama_drive.graphml")
GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)
STATIONS_DB_PATH = Path("map.sqlite")
INCIDENTS_DB_PATH = Path("incidents.sqlite")
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
def load_incident_data(excel_path: str = "R6.xlsx", db_path: Path = INCIDENTS_DB_PATH) -> pd.DataFrame:
    """Load R6 incident records from SQLite if available, otherwise from Excel."""
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            # Check if table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents_r6'")
            if cursor.fetchone():
                df = pd.read_sql("SELECT * FROM incidents_r6", conn)
                df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df
    
    # Fallback to Excel
    if not Path(excel_path).exists():
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(excel_path)
    df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
    df = df[df["覚知"].notna()].copy()
    df["date"] = df["覚知"].dt.date
    return df


@st.cache_data(show_spinner=False)
def load_incident_data_h27(excel_path: str = "H27.xls", db_path: Path = INCIDENTS_DB_PATH) -> pd.DataFrame:
    """Load H27 incident records from SQLite if available, otherwise from Excel."""
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            # Check if table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents_h27'")
            if cursor.fetchone():
                df = pd.read_sql("SELECT * FROM incidents_h27", conn)
                df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df
    
    # Fallback to Excel
    if not Path(excel_path).exists():
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(excel_path)
    
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
    df["date"] = df["覚知"].dt.date
    
    # Normalize column names to match R6 format
    df["出動場所"] = df["出場場所-1"]
    df["出動隊"] = df["出場隊名"]
    df["曜日"] = df["覚知曜日名"]
    
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
            graph = ox.graph_from_place("Matsuyama, Ehime, Japan", network_type="drive")
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
    delay_factor: float = 1.0,
) -> gpd.GeoDataFrame:
    """Compute isochrones with optional traffic delay factor.
    
    Args:
        delay_factor: Multiplier for travel times (>1 means slower due to traffic)
    """
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
    # Adjust max_radius by delay factor (if delay_factor > 1, effective range shrinks)
    max_radius = (trip_times_sorted[-1] * 60 / delay_factor) if trip_times_sorted else 0

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
            # With delay_factor > 1, effective reach shrinks (slower travel)
            cutoff = minutes * 60 / delay_factor
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


def create_location_picker_map(
    stations: gpd.GeoDataFrame,
    virtual_stations: list[dict] | None = None,
) -> folium.Map:
    """Create an interactive map for picking locations to add virtual stations."""
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB Positron")

    # Show existing stations
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
            tooltip=f"既存: {row['略称']}",
        ).add_to(fmap)

    # Show virtual stations (if any)
    if virtual_stations:
        for vs in virtual_stations:
            folium.CircleMarker(
                location=[vs["緯度"], vs["経度"]],
                radius=8,
                color="#e63946",
                weight=2,
                fill=True,
                fill_color="#e63946",
                fill_opacity=0.9,
                popup=f"仮想: {vs['略称']}",
                tooltip=f"仮想: {vs['略称']}",
            ).add_to(fmap)

    return fmap


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

    tab_summary, tab_iso, tab_inc, tab_coverage, tab_resource, tab_optimize = st.tabs([
        "📊 サマリー", "到達圏", "出動地点 (R6)", "カバー率分析", "🚑 リソース分析", "⭐ 配置最適化"
    ])

    # ========== タブ0: サマリー ==========
    with tab_summary:
        st.header("📊 松山市 救急搬送データ サマリー")
        
        # データ読み込み
        @st.cache_data
        def load_summary_data():
            summary = {}
            
            # 消防署数
            if os.path.exists("map.sqlite"):
                conn = sqlite3.connect("map.sqlite")
                stations = pd.read_sql_query("SELECT * FROM stations", conn)
                conn.close()
                summary["stations_count"] = len(stations)
            else:
                summary["stations_count"] = 0
            
            # R6出動データ
            if os.path.exists("incidents.sqlite"):
                conn = sqlite3.connect("incidents.sqlite")
                try:
                    r6_df = pd.read_sql_query("SELECT * FROM incidents_r6", conn)
                    summary["r6_total"] = len(r6_df)
                    if "出動日" in r6_df.columns:
                        r6_df["出動日"] = pd.to_datetime(r6_df["出動日"], errors="coerce")
                        summary["r6_days"] = r6_df["出動日"].nunique()
                    else:
                        summary["r6_days"] = 0
                except:
                    summary["r6_total"] = 0
                    summary["r6_days"] = 0
                
                try:
                    h27_df = pd.read_sql_query("SELECT * FROM incidents_h27", conn)
                    summary["h27_total"] = len(h27_df)
                except:
                    summary["h27_total"] = 0
                conn.close()
            else:
                summary["r6_total"] = 0
                summary["h27_total"] = 0
                summary["r6_days"] = 0
            
            # ジオコーディング済みデータ
            if os.path.exists("cache/incident_geocode.parquet"):
                geo_df = pd.read_parquet("cache/incident_geocode.parquet")
                summary["geocoded"] = len(geo_df)
            else:
                summary["geocoded"] = 0
            
            return summary
        
        summary = load_summary_data()
        
        # メトリクスカード
        st.markdown("### 📈 基本統計")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏢 消防署数", f"{summary['stations_count']} 署")
        with col2:
            st.metric("🚑 R6 出動件数", f"{summary['r6_total']:,} 件")
        with col3:
            st.metric("📅 R6 データ日数", f"{summary['r6_days']} 日")
        with col4:
            st.metric("📍 位置特定済み", f"{summary['geocoded']:,} 件")
        
        st.markdown("---")
        
        # 比較
        st.markdown("### 📊 年度別比較")
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            st.metric("H27 (2015年)", f"{summary['h27_total']:,} 件")
        with col_comp2:
            st.metric("R6 (2024年)", f"{summary['r6_total']:,} 件")
        with col_comp3:
            if summary['h27_total'] > 0:
                change = summary['r6_total'] - summary['h27_total']
                change_pct = (change / summary['h27_total']) * 100
                st.metric("変化", f"{change:+,} 件", delta=f"{change_pct:+.1f}%")
            else:
                st.metric("変化", "N/A")
        
        st.markdown("---")
        
        # クイックリンク
        st.markdown("### 🔗 各機能へ")
        st.markdown("""
        | タブ | 説明 |
        |------|------|
        | **到達圏** | 消防署からの5/10/15/20分到達圏を表示。仮想消防署の追加も可能 |
        | **出動地点 (R6)** | 日別の出動地点をプロット。🎬 アニメーション表示対応 |
        | **カバー率分析** | H27とR6のカバー率を比較。改善度を数値で確認 |
        | **リソース分析** | 各消防署のリソース（救急車台数）を分析 |
        | **配置最適化** | 新規消防署の最適配置をシミュレーション |
        """)
        
        st.info("💡 各タブをクリックして詳細な分析を行ってください")

    with tab_iso:
        with st.expander("🗺️ 仮想消防署を追加（このセッションのみ）", expanded=False):
            st.markdown("**地図をクリックして仮想消防署の場所を選択してください**")
            st.caption("クリック後、名前を入力して「追加」ボタンを押してください。")

            # クリック選択用の地図を表示
            picker_map = create_location_picker_map(
                stations,
                st.session_state["virtual_stations"],
            )
            map_data = st_folium(
                picker_map,
                width=700,
                height=400,
                key="location_picker",
                returned_objects=["last_clicked"],
            )

            # クリックした座標を取得
            clicked_lat = None
            clicked_lon = None
            if map_data and map_data.get("last_clicked"):
                clicked_lat = map_data["last_clicked"]["lat"]
                clicked_lon = map_data["last_clicked"]["lng"]

            col_info, col_add = st.columns([2, 1])
            with col_info:
                if clicked_lat is not None:
                    st.success(f"📍 選択位置: 緯度 {clicked_lat:.6f}, 経度 {clicked_lon:.6f}")
                else:
                    st.info("💡 地図をクリックして場所を選択してください")

            with col_add:
                default_name = f"仮想署{len(st.session_state['virtual_stations']) + 1}"
                v_name = st.text_input("名前", value=default_name, key="virtual_name_input")

            # 追加ボタン
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ この場所に追加", type="primary", disabled=(clicked_lat is None)):
                    if clicked_lat is not None:
                        st.session_state["virtual_stations"].append({
                            "略称": v_name.strip() or default_name,
                            "緯度": clicked_lat,
                            "経度": clicked_lon,
                        })
                        st.success(f"仮想消防署を追加しました: {v_name}")
                        st.rerun()
            with col_btn2:
                if st.button("🗑️ 全てクリア", type="secondary"):
                    st.session_state["virtual_stations"] = []
                    st.info("仮想消防署をクリアしました。")
                    st.rerun()

            # 追加済み仮想消防署一覧
            if st.session_state["virtual_stations"]:
                st.markdown("---")
                st.markdown("**追加済み仮想消防署:**")
                for i, vs in enumerate(st.session_state["virtual_stations"]):
                    col_name, col_coord, col_del = st.columns([2, 3, 1])
                    with col_name:
                        st.write(f"🔴 {vs['略称']}")
                    with col_coord:
                        st.caption(f"({vs['緯度']:.5f}, {vs['経度']:.5f})")
                    with col_del:
                        if st.button("削除", key=f"del_vs_{i}"):
                            st.session_state["virtual_stations"].pop(i)
                            st.rerun()

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

        # ========== 🚦 渋滞考慮モード ==========
        with st.expander("🚦 渋滞考慮モード（実データ学習済み）", expanded=False):
            factors_exist = DELAY_FACTORS_PATH.exists()
            if factors_exist:
                st.success("✅ R6実データから学習した遅延係数を使用")
            else:
                st.warning("⚠️ misc/learn_delays.py を実行すると学習できます")

            traffic_enabled = st.toggle("渋滞を考慮する", value=False, key="traffic_enabled")

            if traffic_enabled:
                col_time, col_dow = st.columns(2)
                with col_time:
                    time_slot = st.selectbox(
                        "時間帯",
                        options=list(TIME_SLOT_LABELS.keys()),
                        index=3,  # 朝ラッシュ
                        key="traffic_time_slot",
                    )
                    # 代表的な時間を取得
                    slot_hours = TIME_SLOT_LABELS[time_slot]
                    selected_hour = slot_hours[len(slot_hours) // 2]

                with col_dow:
                    use_dow = st.checkbox("曜日も考慮", value=False, key="traffic_use_dow")
                    if use_dow:
                        dow_label = st.selectbox(
                            "曜日",
                            options=DOW_LABELS,
                            index=0,
                            key="traffic_dow",
                        )
                        selected_dow = DOW_LABELS.index(dow_label)
                    else:
                        selected_dow = None

                delay_factor = get_delay_factor(selected_hour, selected_dow)
                
                # 係数の意味を表示（実データ: 日中が最速、深夜が遅い）
                if delay_factor < 1.05:
                    emoji = "🟢"
                    desc = "最速（日中帯）"
                elif delay_factor < 1.2:
                    desc = "やや遅い"
                    emoji = "🟡"
                else:
                    desc = "遅い（深夜帯）"
                    emoji = "🔴"
                
                st.info(f"{emoji} 遅延係数: **{delay_factor:.3f}** ({desc})")
                st.caption(f"→ 例: 5分圏が実質 {5 * delay_factor:.1f}分圏 に縮小")
            else:
                delay_factor = 1.0
        # ========================================

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

        # 渋滞考慮モードではキャッシュを使わず再計算（係数が異なるため）
        use_cache = ISOCHRONE_CACHE_PATH.exists() and not has_virtual and delay_factor == 1.0
        
        if use_cache:
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
            spinner_msg = "到達圏を計算しています..."
            if delay_factor != 1.0:
                spinner_msg = f"渋滞考慮で到達圏を計算中（係数: {delay_factor:.3f}）..."
            with st.spinner(spinner_msg):
                prog = st.progress(0)
                display_isochrones = compute_isochrones(
                    graph=graph,
                    stations=filtered,
                    trip_times=selected_times,
                    progress_cb=lambda p: prog.progress(int(p * 100)),
                    delay_factor=delay_factor,
                )

        if display_isochrones.empty:
            st.error("選択条件に合致する到達圏がありません。")
            st.stop()

        html_map = render_map_html(display_isochrones, filtered)
        st.components.v1.html(html_map, height=720)

    with tab_inc:
        st.subheader("🗓️ 出動地点プロット")
        
        try:
            incidents = load_incident_data("R6.xlsx")
        except FileNotFoundError:
            st.error("R6.xlsx が見つかりません。ルートに配置してください。")
            st.stop()

        date_options = sorted(incidents["date"].unique())
        if not date_options:
            st.warning("R6.xlsx に日付データがありません。")
            st.stop()

        # 日付選択
        col_date, col_mode = st.columns([2, 1])
        with col_date:
            default_date = date_options[0]
            selected_date = st.selectbox(
                "表示する日付 (覚知日)",
                options=date_options,
                format_func=lambda d: d.strftime("%Y-%m-%d"),
                index=0,
            )
        with col_mode:
            display_mode = st.radio(
                "表示モード",
                ["📍 静的表示", "🎬 アニメーション"],
                horizontal=True,
                help="アニメーションでは時系列で出動を再生できます"
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

        # 曜日別カラーマップ
        weekday_colors = {
            "月": "#f94144",
            "火": "#f3722c",
            "水": "#f9c74f",
            "木": "#90be6d",
            "金": "#43aa8b",
            "土": "#577590",
            "日": "#9d4edd",
        }

        center_lat = mapped["lat"].mean()
        center_lon = mapped["lon"].mean()

        if display_mode == "📍 静的表示":
            # 従来の静的表示
            fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB Positron")

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

        else:
            # 🎬 アニメーション表示
            st.markdown("---")
            st.markdown("### 🎬 出動アニメーション設定")
            
            col_anim1, col_anim2, col_anim3 = st.columns(3)
            with col_anim1:
                period_min = st.slider(
                    "再生ステップ (分)",
                    min_value=5,
                    max_value=60,
                    value=15,
                    step=5,
                    help="タイムラインの1ステップあたりの時間"
                )
            with col_anim2:
                duration_min = st.slider(
                    "ポイント表示時間 (分)",
                    min_value=30,
                    max_value=180,
                    value=60,
                    step=30,
                    help="出動地点が地図上に表示され続ける時間"
                )
            with col_anim3:
                auto_play = st.checkbox("自動再生", value=False)
            
            # GeoJSON FeatureCollection を生成
            from folium.plugins import TimestampedGeoJson
            
            # 時刻でソート
            mapped_sorted = mapped.sort_values("覚知")
            
            features = []
            for _, row in mapped_sorted.iterrows():
                if pd.isna(row.get("覚知")):
                    continue
                
                wk = str(row.get("曜日", "?"))
                color = weekday_colors.get(wk, "#4a4a4a")
                
                # ISO8601形式の時刻
                time_str = row["覚知"].strftime("%Y-%m-%dT%H:%M:%S")
                
                label_time = row["覚知"].strftime("%H:%M")
                popup_text = f"{row.get('出動隊', '不明')} | {label_time} | {row.get('搬送区分(事案)', '')}"
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row["lon"], row["lat"]],  # GeoJSON: [lon, lat]
                    },
                    "properties": {
                        "time": time_str,
                        "popup": popup_text,
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": color,
                            "fillOpacity": 0.8,
                            "stroke": "true",
                            "color": color,
                            "radius": 8,
                        },
                    },
                }
                features.append(feature)
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": features,
            }
            
            # 地図生成
            fmap_anim = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="CartoDB Positron"
            )
            
            # TimestampedGeoJson追加
            TimestampedGeoJson(
                geojson_data,
                period=f"PT{period_min}M",
                duration=f"PT{duration_min}M",
                auto_play=auto_play,
                loop=True,
                loop_button=True,
                date_options="HH:mm",
                time_slider_drag_update=True,
            ).add_to(fmap_anim)
            
            # 消防署も表示
            for _, station in stations.iterrows():
                folium.Marker(
                    location=[station["緯度"], station["経度"]],
                    popup=f"🏥 {station['略称']}",
                    icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
                ).add_to(fmap_anim)
            
            st.components.v1.html(fmap_anim.get_root().render(), height=720)
            
            st.caption("""
            **操作方法**: 
            - ▶️ 再生ボタンでアニメーション開始
            - スライダーをドラッグして時刻を移動
            - 出動地点は時間経過で出現・消滅します
            """)
            
            # 時間帯別の出動件数サマリ
            with st.expander("📊 時間帯別 出動件数", expanded=False):
                mapped_sorted["hour"] = pd.to_datetime(mapped_sorted["覚知"]).dt.hour
                hourly_counts = mapped_sorted.groupby("hour").size().reset_index(name="件数")
                hourly_counts.columns = ["時", "件数"]
                
                import altair as alt
                chart = alt.Chart(hourly_counts).mark_bar(color="#f94144").encode(
                    x=alt.X("時:O", title="時刻"),
                    y=alt.Y("件数:Q", title="出動件数"),
                    tooltip=["時", "件数"],
                ).properties(width=600, height=200)
                st.altair_chart(chart, width="stretch")

    with tab_coverage:
        st.subheader("📊 出動地点の到達圏カバー率分析")
        st.caption("現在の消防署配置でH27・R6の出動データを分析します。")

        # Check file availability
        r6_available = Path("R6.xlsx").exists()
        h27_available = Path("H27.xls").exists()

        if not r6_available and not h27_available:
            st.error("出動データファイルが見つかりません。R6.xlsx または H27.xls を配置してください。")
            st.stop()

        # Dataset selection
        dataset_options = []
        if r6_available:
            dataset_options.append("R6 (2024年)")
        if h27_available:
            dataset_options.append("H27 (2015年)")
        if r6_available and h27_available:
            dataset_options.append("⭐ 比較モード (R6 vs H27)")

        col_mode, col_resource = st.columns([2, 1])
        with col_mode:
            selected_mode = st.radio(
                "分析モード",
                options=dataset_options,
                horizontal=True,
            )
        
        is_comparison = "比較モード" in selected_mode

        # ========== 🚑 リソース考慮オプション ==========
        with col_resource:
            resource_mode = st.checkbox(
                "🚑 リソース考慮",
                value=False,
                help="各出動地点で到達可能な救急車台数を考慮したカバー率を計算します",
            )
        
        if resource_mode:
            st.info("""
            **リソース考慮モード**: 単純な到達圏内/外ではなく、各出動地点に「何台の救急車が到達可能か」を分析します。
            - 🟢 **2台以上**: 冗長性あり（1台出動中でも対応可能）
            - 🟡 **1台のみ**: カバーされているが冗長性なし
            - 🔴 **0台**: 到達圏外
            """)
            
            # H27リソース設定の調整UI
            with st.expander("⚙️ H27リソース設定を調整", expanded=False):
                st.caption("H27当時の正確な配置が不明なため、手動で調整できます")
                
                h27_resources_custom = {}
                cols_h27 = st.columns(3)
                station_names = list(STATION_RESOURCES_H27_DEFAULT.keys())
                for i, station in enumerate(station_names):
                    default_val = STATION_RESOURCES_H27_DEFAULT[station]
                    with cols_h27[i % 3]:
                        h27_resources_custom[station] = st.number_input(
                            station,
                            min_value=0,
                            max_value=10,
                            value=default_val,
                            key=f"h27_res_{station}"
                        )
                
                total_h27 = sum(h27_resources_custom.values())
                total_r6 = sum(STATION_RESOURCES_R6.values())
                st.metric("H27 合計", f"{total_h27}台", delta=f"{total_h27 - total_r6}台 vs R6")
                
                # session_stateに保存
                st.session_state["h27_resources_custom"] = h27_resources_custom

        # ========== 🚦 渋滞考慮 & 時間帯別分析 ==========
        with st.expander("🚦 時間帯別カバー率分析", expanded=False):
            factors_exist_cov = DELAY_FACTORS_PATH.exists()
            if factors_exist_cov:
                st.success("✅ R6実データから学習した遅延係数を使用")
            else:
                st.warning("⚠️ misc/learn_delays.py を実行すると学習できます")

            analysis_type = st.radio(
                "分析タイプ",
                ["通常（渋滞なし）", "🕐 時間帯別カバー率", "🚦 特定時間帯の渋滞考慮"],
                horizontal=True,
                key="coverage_analysis_type",
            )

            cov_delay_factor = 1.0
            selected_hour_cov = None
            hourly_analysis = False

            if analysis_type == "🕐 時間帯別カバー率":
                hourly_analysis = True
                st.info("📊 出動時刻に基づいて時間帯別のカバー率を集計します")
            
            elif analysis_type == "🚦 特定時間帯の渋滞考慮":
                col_slot, col_info = st.columns([1, 1])
                with col_slot:
                    time_slot_cov = st.selectbox(
                        "時間帯",
                        options=list(TIME_SLOT_LABELS.keys()),
                        index=3,
                        key="coverage_time_slot",
                    )
                    slot_hours_cov = TIME_SLOT_LABELS[time_slot_cov]
                    selected_hour_cov = slot_hours_cov[len(slot_hours_cov) // 2]
                    cov_delay_factor = get_delay_factor(selected_hour_cov)
                
                with col_info:
                    if cov_delay_factor < 1.0:
                        emoji_cov = "🟢"
                        desc_cov = "救急優先走行で速い"
                    elif cov_delay_factor < 1.1:
                        emoji_cov = "🟡"
                        desc_cov = "通常"
                    else:
                        emoji_cov = "🔴"
                        desc_cov = "やや混雑"
                    st.metric(
                        "遅延係数",
                        f"{cov_delay_factor:.3f}",
                        delta=desc_cov,
                    )
        # ================================================

        # Load or compute isochrones (shared for all modes)
        stations_cov = load_station_data(
            db_path=STATIONS_DB_PATH,
            excel_path="map.xlsx",
            source_mtime=station_data_version(),
        )
        trip_times_cov = [5, 10]

        # 渋滞考慮モードではキャッシュを使わず再計算
        use_cache_cov = ISOCHRONE_CACHE_PATH.exists() and cov_delay_factor == 1.0

        if use_cache_cov:
            try:
                with st.spinner("到達圏キャッシュを読み込み中..."):
                    isochrones_cov = load_precomputed_isochrones(ISOCHRONE_CACHE_PATH)
            except Exception as exc:
                st.warning(f"キャッシュ読み込み失敗: {exc}")
                isochrones_cov = None
        else:
            isochrones_cov = None

        if isochrones_cov is None:
            padding_deg = 0.1
            west_cov, south_cov, east_cov, north_cov = stations_cov.total_bounds
            bbox_cov = (north_cov + padding_deg, south_cov - padding_deg, east_cov + padding_deg, west_cov - padding_deg)
            with st.spinner("道路ネットワークを読み込み中..."):
                graph_cov = load_graph_cached(bbox_cov)
            spinner_msg_cov = "到達圏を計算中..."
            if cov_delay_factor != 1.0:
                spinner_msg_cov = f"渋滞考慮で到達圏を計算中（係数: {cov_delay_factor:.3f}）..."
            with st.spinner(spinner_msg_cov):
                prog_cov = st.progress(0)
                isochrones_cov = compute_isochrones(
                    graph=graph_cov,
                    stations=stations_cov,
                    trip_times=trip_times_cov,
                    progress_cb=lambda p: prog_cov.progress(int(p * 100)),
                    delay_factor=cov_delay_factor,
                )

        def analyze_coverage(incidents_df: pd.DataFrame, label: str, isochrones: gpd.GeoDataFrame = None, with_resources: bool = False) -> dict:
            """Analyze coverage for a given incident dataset.
            
            Args:
                incidents_df: 出動データ
                label: データラベル
                isochrones: 到達圏データ
                with_resources: リソース（救急車台数）を考慮するか
            """
            if isochrones is None:
                isochrones = isochrones_cov
            
            # データラベルに応じてリソース設定を選択
            if "H27" in label or "2015" in label:
                # カスタム設定があればそれを使用、なければデフォルト
                station_resources = st.session_state.get("h27_resources_custom", STATION_RESOURCES_H27_DEFAULT)
                total_amb = sum(station_resources.values())
                resource_label = f"H27（推定{total_amb}台）"
            else:
                station_resources = STATION_RESOURCES_R6
                total_amb = sum(station_resources.values())
                resource_label = f"R6（{total_amb}台）"
            
            addr_series = incidents_df["出動場所"].dropna().astype(str)
            addr_unique = sorted(addr_series.unique())

            geo_df = geocode_addresses(addr_unique, region_prefix="愛媛県")
            merged = incidents_df.merge(geo_df, left_on="出動場所", right_on="address", how="left")
            mapped = merged.dropna(subset=["lat", "lon"]).copy()

            if mapped.empty:
                return None

            incident_points = gpd.GeoDataFrame(
                mapped,
                geometry=gpd.points_from_xy(mapped["lon"], mapped["lat"]),
                crs="EPSG:4326"
            )

            results = {"label": label, "total": len(incidents_df), "geocoded": len(mapped), "resource_config": resource_label}
            
            for minutes in trip_times_cov:
                iso_layer = isochrones[isochrones["time"] == minutes]
                if iso_layer.empty:
                    results[f"covered_{minutes}"] = 0
                    if with_resources:
                        incident_points[f"ambulances_{minutes}min"] = 0
                    continue
                combined_polygon = unary_union(iso_layer.geometry)
                within_mask = incident_points.geometry.within(combined_polygon)
                results[f"covered_{minutes}"] = within_mask.sum()
                incident_points[f"within_{minutes}min"] = within_mask
                
                # リソース考慮モード：各出動地点で到達可能な救急車台数を計算
                if with_resources:
                    ambulance_counts = []
                    for idx, row in incident_points.iterrows():
                        point = row.geometry
                        count = 0
                        # 各消防署の到達圏に含まれるか確認
                        for _, iso_row in iso_layer.iterrows():
                            station_name = iso_row["name"] if "name" in iso_row.index else ""
                            if point.within(iso_row.geometry):
                                # その署の救急車台数を加算（年度別リソース設定を使用）
                                count += station_resources.get(station_name, DEFAULT_AMBULANCES)
                        ambulance_counts.append(count)
                    incident_points[f"ambulances_{minutes}min"] = ambulance_counts
                    
                    # リソース別カバー率
                    results[f"covered_{minutes}_0amb"] = (incident_points[f"ambulances_{minutes}min"] == 0).sum()  # 圏外
                    results[f"covered_{minutes}_1amb"] = (incident_points[f"ambulances_{minutes}min"] == 1).sum()  # 1台のみ
                    results[f"covered_{minutes}_2amb"] = (incident_points[f"ambulances_{minutes}min"] >= 2).sum()  # 2台以上

            results["incident_points"] = incident_points
            results["mapped"] = mapped
            return results

        def analyze_hourly_coverage(incidents_df: pd.DataFrame) -> pd.DataFrame:
            """Analyze coverage by hour of day."""
            # 時間帯の定義
            time_bins = {
                "深夜 (0-5時)": list(range(0, 5)),
                "早朝 (5-7時)": list(range(5, 7)),
                "朝ラッシュ (7-9時)": list(range(7, 9)),
                "午前 (9-12時)": list(range(9, 12)),
                "昼 (12-14時)": list(range(12, 14)),
                "午後 (14-17時)": list(range(14, 17)),
                "夕ラッシュ (17-19時)": list(range(17, 19)),
                "夜 (19-22時)": list(range(19, 22)),
                "深夜 (22-24時)": list(range(22, 24)),
            }
            
            # ジオコーディング
            addr_series = incidents_df["出動場所"].dropna().astype(str)
            addr_unique = sorted(addr_series.unique())
            geo_df = geocode_addresses(addr_unique, region_prefix="愛媛県")
            merged = incidents_df.merge(geo_df, left_on="出動場所", right_on="address", how="left")
            mapped = merged.dropna(subset=["lat", "lon"]).copy()
            
            if mapped.empty:
                return None
            
            # 時間帯列を追加
            mapped["hour"] = pd.to_datetime(mapped["覚知"], errors="coerce").dt.hour
            
            incident_points = gpd.GeoDataFrame(
                mapped,
                geometry=gpd.points_from_xy(mapped["lon"], mapped["lat"]),
                crs="EPSG:4326"
            )
            
            # 時間帯ごとにカバー率を計算
            results = []
            for slot_name, hours in time_bins.items():
                slot_points = incident_points[incident_points["hour"].isin(hours)]
                if slot_points.empty:
                    continue
                
                total = len(slot_points)
                row = {"時間帯": slot_name, "件数": total}
                
                for minutes in trip_times_cov:
                    iso_layer = isochrones_cov[isochrones_cov["time"] == minutes]
                    if iso_layer.empty:
                        row[f"{minutes}分圏カバー"] = 0
                        row[f"{minutes}分圏率"] = 0.0
                        continue
                    combined_polygon = unary_union(iso_layer.geometry)
                    within_mask = slot_points.geometry.within(combined_polygon)
                    covered = within_mask.sum()
                    row[f"{minutes}分圏カバー"] = covered
                    row[f"{minutes}分圏率"] = covered / total * 100 if total > 0 else 0
                
                results.append(row)
            
            return pd.DataFrame(results)

        # ========== 時間帯別分析モード ==========
        if hourly_analysis:
            st.markdown("---")
            st.subheader("🕐 時間帯別カバー率分析")
            
            import altair as alt
            
            # 比較モードの場合は両方のデータを分析
            if is_comparison:
                col_r6_h, col_h27_h = st.columns(2)
                
                with st.spinner("R6データの時間帯別カバー率を計算中..."):
                    incidents_r6_hourly = load_incident_data("R6.xlsx")
                    hourly_df_r6 = analyze_hourly_coverage(incidents_r6_hourly)
                
                with st.spinner("H27データの時間帯別カバー率を計算中..."):
                    incidents_h27_hourly = load_incident_data_h27("H27.xls")
                    hourly_df_h27 = analyze_hourly_coverage(incidents_h27_hourly)
                
                if hourly_df_r6 is not None and hourly_df_h27 is not None:
                    # 並べて表示
                    with col_r6_h:
                        st.markdown("### 🟢 R6 (2024年)")
                        display_df_r6 = hourly_df_r6[["時間帯", "件数", "5分圏率", "10分圏率"]].copy()
                        display_df_r6["5分圏率"] = display_df_r6["5分圏率"].apply(lambda x: f"{x:.1f}%")
                        display_df_r6["10分圏率"] = display_df_r6["10分圏率"].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df_r6, width="stretch", hide_index=True)
                    
                    with col_h27_h:
                        st.markdown("### 🟡 H27 (2015年)")
                        display_df_h27 = hourly_df_h27[["時間帯", "件数", "5分圏率", "10分圏率"]].copy()
                        display_df_h27["5分圏率"] = display_df_h27["5分圏率"].apply(lambda x: f"{x:.1f}%")
                        display_df_h27["10分圏率"] = display_df_h27["10分圏率"].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df_h27, width="stretch", hide_index=True)
                    
                    # 比較グラフ（5分圏）
                    st.markdown("### 📈 5分圏カバー率比較グラフ")
                    
                    # データを結合
                    hourly_df_r6["データ"] = "R6 (2024年)"
                    hourly_df_h27["データ"] = "H27 (2015年)"
                    combined_hourly = pd.concat([hourly_df_r6, hourly_df_h27], ignore_index=True)
                    
                    chart_5min = alt.Chart(combined_hourly).mark_bar().encode(
                        x=alt.X("時間帯:N", sort=list(TIME_SLOT_LABELS.keys()), title="時間帯"),
                        y=alt.Y("5分圏率:Q", title="5分圏カバー率 (%)"),
                        color=alt.Color("データ:N", scale=alt.Scale(
                            domain=["R6 (2024年)", "H27 (2015年)"],
                            range=["#2ecc71", "#f1c40f"]
                        )),
                        xOffset="データ:N",
                        tooltip=["時間帯", "データ", alt.Tooltip("5分圏率:Q", format=".1f"), "件数"],
                    ).properties(
                        width=600,
                        height=350,
                    )
                    st.altair_chart(chart_5min, width="stretch")
                    
                    # 差分テーブル
                    st.markdown("### 📊 時間帯別 改善度（R6 - H27）")
                    diff_data = []
                    for slot in hourly_df_r6["時間帯"].unique():
                        r6_row = hourly_df_r6[hourly_df_r6["時間帯"] == slot]
                        h27_row = hourly_df_h27[hourly_df_h27["時間帯"] == slot]
                        if not r6_row.empty and not h27_row.empty:
                            diff_5 = r6_row["5分圏率"].values[0] - h27_row["5分圏率"].values[0]
                            diff_10 = r6_row["10分圏率"].values[0] - h27_row["10分圏率"].values[0]
                            diff_data.append({
                                "時間帯": slot,
                                "5分圏改善": f"{diff_5:+.1f}%",
                                "10分圏改善": f"{diff_10:+.1f}%",
                            })
                    
                    diff_df = pd.DataFrame(diff_data)
                    st.dataframe(diff_df, width="stretch", hide_index=True)
                    
                    # サマリ
                    st.markdown("### 🔍 分析サマリ")
                    avg_diff_5 = (hourly_df_r6["5分圏率"].mean() - hourly_df_h27["5分圏率"].mean())
                    avg_diff_10 = (hourly_df_r6["10分圏率"].mean() - hourly_df_h27["10分圏率"].mean())
                    
                    if avg_diff_5 > 0:
                        st.success(f"✅ 全時間帯平均 5分圏カバー率: **{avg_diff_5:+.1f}%** 改善")
                    else:
                        st.warning(f"⚠️ 全時間帯平均 5分圏カバー率: **{avg_diff_5:+.1f}%**")
                    
                    if avg_diff_10 > 0:
                        st.success(f"✅ 全時間帯平均 10分圏カバー率: **{avg_diff_10:+.1f}%** 改善")
                    else:
                        st.warning(f"⚠️ 全時間帯平均 10分圏カバー率: **{avg_diff_10:+.1f}%**")
                else:
                    st.error("時間帯別分析に失敗しました。")
            
            else:
                # 単一データセットモード
                if "R6" in selected_mode:
                    incidents_hourly = load_incident_data("R6.xlsx")
                    data_label_hourly = "R6 (2024年)"
                else:
                    incidents_hourly = load_incident_data_h27("H27.xls")
                    data_label_hourly = "H27 (2015年)"
                
                with st.spinner(f"{data_label_hourly} の時間帯別カバー率を計算中..."):
                    hourly_df = analyze_hourly_coverage(incidents_hourly)
                
                if hourly_df is not None:
                    st.markdown(f"### 📊 {data_label_hourly} 時間帯別カバー率")
                    
                    # テーブル表示
                    display_df = hourly_df.copy()
                    display_df["5分圏率"] = display_df["5分圏率"].apply(lambda x: f"{x:.1f}%")
                    display_df["10分圏率"] = display_df["10分圏率"].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_df, width="stretch", hide_index=True)
                    
                    # グラフ表示
                    st.markdown("### 📈 時間帯別カバー率グラフ")
                    
                    chart_data = hourly_df.melt(
                        id_vars=["時間帯", "件数"],
                        value_vars=["5分圏率", "10分圏率"],
                        var_name="到達圏",
                        value_name="カバー率",
                    )
                    
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("時間帯:N", sort=list(TIME_SLOT_LABELS.keys()), title="時間帯"),
                        y=alt.Y("カバー率:Q", title="カバー率 (%)"),
                        color=alt.Color("到達圏:N", scale=alt.Scale(
                            domain=["5分圏率", "10分圏率"],
                            range=["#ff9e9e", "#8aa5ff"]
                        )),
                        xOffset="到達圏:N",
                        tooltip=["時間帯", "到達圏", alt.Tooltip("カバー率:Q", format=".1f"), "件数"],
                    ).properties(
                        width=600,
                        height=400,
                    )
                    st.altair_chart(chart, width="stretch")
                    
                    # 時間帯間の差を分析
                    st.markdown("### 🔍 分析サマリ")
                    best_5min = hourly_df.loc[hourly_df["5分圏率"].idxmax()]
                    worst_5min = hourly_df.loc[hourly_df["5分圏率"].idxmin()]
                    
                    col_best, col_worst = st.columns(2)
                    with col_best:
                        st.success(f"✅ 5分圏カバー率 最高: **{best_5min['時間帯']}** ({best_5min['5分圏率']:.1f}%)")
                    with col_worst:
                        st.warning(f"⚠️ 5分圏カバー率 最低: **{worst_5min['時間帯']}** ({worst_5min['5分圏率']:.1f}%)")
                    
                    st.info(f"📊 時間帯による5分圏カバー率の差: **{best_5min['5分圏率'] - worst_5min['5分圏率']:.1f}%**")
                else:
                    st.error("時間帯別分析に失敗しました。")
            
            st.stop()  # 時間帯別分析モードでは以降の処理をスキップ
        # ========================================

        if is_comparison:
            # Comparison mode: analyze both datasets
            st.markdown("---")
            col_r6, col_h27 = st.columns(2)

            with col_r6:
                st.markdown("### 🟢 R6 (2024年)")
            with col_h27:
                st.markdown("### 🟡 H27 (2015年)")

            with st.spinner("R6データを分析中..."):
                incidents_r6 = load_incident_data("R6.xlsx")
                results_r6 = analyze_coverage(incidents_r6, "R6", with_resources=resource_mode)

            with st.spinner("H27データを分析中..."):
                incidents_h27 = load_incident_data_h27("H27.xls")
                results_h27 = analyze_coverage(incidents_h27, "H27", with_resources=resource_mode)

            if results_r6 is None or results_h27 is None:
                st.error("データの分析に失敗しました。")
                st.stop()

            # Build comparison table
            comparison_data = []
            for minutes in trip_times_cov:
                r6_covered = results_r6[f"covered_{minutes}"]
                r6_total = results_r6["geocoded"]
                r6_pct = r6_covered / r6_total * 100 if r6_total > 0 else 0

                h27_covered = results_h27[f"covered_{minutes}"]
                h27_total = results_h27["geocoded"]
                h27_pct = h27_covered / h27_total * 100 if h27_total > 0 else 0

                diff = r6_pct - h27_pct

                comparison_data.append({
                    "到達時間": f"{minutes}分",
                    "R6 カバー率": f"{r6_pct:.1f}%",
                    "R6 (件数)": f"{r6_covered}/{r6_total}",
                    "H27 カバー率": f"{h27_pct:.1f}%",
                    "H27 (件数)": f"{h27_covered}/{h27_total}",
                    "差分": f"{diff:+.1f}%",
                })

            st.markdown("### 📊 カバー率比較（到達圏内/外）")
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width="stretch", hide_index=True)

            # ========== リソース考慮モードの追加表示 ==========
            if resource_mode:
                st.markdown("---")
                st.markdown("### 🚑 リソース考慮カバー率比較")
                st.caption("各出動地点に到達可能な救急車台数で分類")
                
                # 使用リソース設定を表示
                st.info(f"""
                **使用リソース設定**
                - R6: {results_r6.get('resource_config', 'R6')}
                - H27: {results_h27.get('resource_config', 'H27')}
                """)
                
                resource_comparison = []
                for minutes in trip_times_cov:
                    r6_total = results_r6["geocoded"]
                    h27_total = results_h27["geocoded"]
                    
                    # R6のリソース別カバー
                    r6_0 = results_r6.get(f"covered_{minutes}_0amb", 0)
                    r6_1 = results_r6.get(f"covered_{minutes}_1amb", 0)
                    r6_2 = results_r6.get(f"covered_{minutes}_2amb", 0)
                    
                    # H27のリソース別カバー
                    h27_0 = results_h27.get(f"covered_{minutes}_0amb", 0)
                    h27_1 = results_h27.get(f"covered_{minutes}_1amb", 0)
                    h27_2 = results_h27.get(f"covered_{minutes}_2amb", 0)
                    
                    resource_comparison.append({
                        "到達時間": f"{minutes}分",
                        "R6 🔴圏外": f"{r6_0} ({r6_0/r6_total*100:.1f}%)" if r6_total > 0 else "0",
                        "R6 🟡1台": f"{r6_1} ({r6_1/r6_total*100:.1f}%)" if r6_total > 0 else "0",
                        "R6 🟢2台+": f"{r6_2} ({r6_2/r6_total*100:.1f}%)" if r6_total > 0 else "0",
                        "H27 🔴圏外": f"{h27_0} ({h27_0/h27_total*100:.1f}%)" if h27_total > 0 else "0",
                        "H27 🟡1台": f"{h27_1} ({h27_1/h27_total*100:.1f}%)" if h27_total > 0 else "0",
                        "H27 🟢2台+": f"{h27_2} ({h27_2/h27_total*100:.1f}%)" if h27_total > 0 else "0",
                    })
                
                resource_df = pd.DataFrame(resource_comparison)
                st.dataframe(resource_df, width="stretch", hide_index=True)
                
                # リソースメトリクス
                st.markdown("### 📊 リソース冗長性の比較")
                for minutes in trip_times_cov:
                    r6_total = results_r6["geocoded"]
                    h27_total = results_h27["geocoded"]
                    
                    r6_redundant = results_r6.get(f"covered_{minutes}_2amb", 0) / r6_total * 100 if r6_total > 0 else 0
                    h27_redundant = results_h27.get(f"covered_{minutes}_2amb", 0) / h27_total * 100 if h27_total > 0 else 0
                    diff_redundant = r6_redundant - h27_redundant
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"R6 {minutes}分 冗長性あり", f"{r6_redundant:.1f}%")
                    with col2:
                        st.metric(f"H27 {minutes}分 冗長性あり", f"{h27_redundant:.1f}%")
                    with col3:
                        st.metric(
                            f"{minutes}分 冗長性改善",
                            f"{diff_redundant:+.1f}%",
                            delta=f"{diff_redundant:+.1f}%" if diff_redundant != 0 else None,
                            delta_color="normal" if diff_redundant >= 0 else "inverse"
                        )
            # ================================================

            # Metrics side by side
            st.markdown("### 📈 差分メトリクス")
            for minutes in trip_times_cov:
                r6_pct = results_r6[f"covered_{minutes}"] / results_r6["geocoded"] * 100
                h27_pct = results_h27[f"covered_{minutes}"] / results_h27["geocoded"] * 100
                diff = r6_pct - h27_pct

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"R6 {minutes}分到達圏", f"{r6_pct:.1f}%")
                with col2:
                    st.metric(f"H27 {minutes}分到達圏", f"{h27_pct:.1f}%")
                with col3:
                    st.metric(
                        f"{minutes}分圏 差分",
                        f"{diff:+.1f}%",
                        delta=f"{diff:+.1f}%" if diff != 0 else None,
                        delta_color="normal" if diff >= 0 else "inverse"
                    )

            # Summary
            st.markdown("---")
            st.markdown("### 📝 分析サマリ")
            r6_5min_pct = results_r6["covered_5"] / results_r6["geocoded"] * 100
            h27_5min_pct = results_h27["covered_5"] / results_h27["geocoded"] * 100
            r6_10min_pct = results_r6["covered_10"] / results_r6["geocoded"] * 100
            h27_10min_pct = results_h27["covered_10"] / results_h27["geocoded"] * 100

            diff_5 = r6_5min_pct - h27_5min_pct
            diff_10 = r6_10min_pct - h27_10min_pct
            
            st.markdown("""
            **注意**: この比較は**現在の消防署配置**で両年度の出動データを分析したものです。
            実際の配置変更（R4〜R6で3隊増隊）の効果を直接測定したものではありません。
            """)
            
            if diff_5 > 0:
                st.success(f"✅ 5分到達圏カバー率: R6がH27より {diff_5:+.1f}% 高い")
            else:
                st.warning(f"⚠️ 5分到達圏カバー率: R6がH27より {diff_5:+.1f}%")

            if diff_10 > 0:
                st.success(f"✅ 10分到達圏カバー率: R6がH27より {diff_10:+.1f}% 高い")
            else:
                st.warning(f"⚠️ 10分到達圏カバー率: R6がH27より {diff_10:+.1f}%")

            st.info(f"""
            **データ件数**
            - R6 (2024年): {results_r6['total']:,} 件 (ジオコーディング成功: {results_r6['geocoded']:,} 件)
            - H27 (2015年): {results_h27['total']:,} 件 (ジオコーディング成功: {results_h27['geocoded']:,} 件)
            """)

        else:
            # Single dataset mode
            if "R6" in selected_mode:
                incidents_cov = load_incident_data("R6.xlsx")
                data_label = "R6 (2024年)"
            else:
                incidents_cov = load_incident_data_h27("H27.xls")
                data_label = "H27 (2015年)"

            st.markdown(f"### 📅 {data_label} のカバー率分析")
            st.write(f"全出動件数: {len(incidents_cov):,} 件")

            with st.spinner("住所をジオコーディング中..."):
                results = analyze_coverage(incidents_cov, data_label, with_resources=resource_mode)

            if results is None:
                st.error("出動地点をジオコーディングできませんでした。")
                st.stop()

            incident_points = results["incident_points"]
            mapped_cov = results["mapped"]

            geocoded_rate = results["geocoded"] / results["total"] * 100
            st.write(f"📍 ジオコーディング成功: {results['geocoded']:,} / {results['total']:,} 件 ({geocoded_rate:.1f}%)")

            # Calculate coverage for each time threshold
            st.markdown("---")
            st.subheader("📊 カバー率結果")

            coverage_results = []
            for minutes in trip_times_cov:
                covered_count = results[f"covered_{minutes}"]
                total_count = results["geocoded"]
                coverage_pct = covered_count / total_count * 100 if total_count > 0 else 0

                coverage_results.append({
                    "到達時間": f"{minutes}分",
                    "カバー数": covered_count,
                    "全件数": total_count,
                    "カバー率": f"{coverage_pct:.1f}%",
                })

            if coverage_results:
                coverage_df = pd.DataFrame(coverage_results)
                st.dataframe(coverage_df, width="stretch", hide_index=True)

                # Show metrics
                cols = st.columns(len(coverage_results))
                for i, res in enumerate(coverage_results):
                    with cols[i]:
                        st.metric(
                            label=f"{res['到達時間']}到達圏",
                            value=res["カバー率"],
                            delta=f"{res['カバー数']}/{res['全件数']}件"
                        )

            # ========== リソース考慮モードの追加表示（単一データセット） ==========
            if resource_mode:
                st.markdown("---")
                st.subheader("🚑 リソース考慮カバー率")
                st.caption("各出動地点に到達可能な救急車台数で分類")
                
                # 使用リソース設定を表示
                st.info(f"""
                **使用リソース設定**: {results.get('resource_config', '')}
                """)
                
                resource_results = []
                for minutes in trip_times_cov:
                    total = results["geocoded"]
                    amb_0 = results.get(f"covered_{minutes}_0amb", 0)
                    amb_1 = results.get(f"covered_{minutes}_1amb", 0)
                    amb_2 = results.get(f"covered_{minutes}_2amb", 0)
                    
                    resource_results.append({
                        "到達時間": f"{minutes}分",
                        "🔴 圏外 (0台)": f"{amb_0} ({amb_0/total*100:.1f}%)" if total > 0 else "0",
                        "🟡 1台のみ": f"{amb_1} ({amb_1/total*100:.1f}%)" if total > 0 else "0",
                        "🟢 2台以上": f"{amb_2} ({amb_2/total*100:.1f}%)" if total > 0 else "0",
                    })
                
                resource_df = pd.DataFrame(resource_results)
                st.dataframe(resource_df, width="stretch", hide_index=True)
                
                # リソースメトリクス
                for minutes in trip_times_cov:
                    total = results["geocoded"]
                    amb_0 = results.get(f"covered_{minutes}_0amb", 0)
                    amb_1 = results.get(f"covered_{minutes}_1amb", 0)
                    amb_2 = results.get(f"covered_{minutes}_2amb", 0)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{minutes}分 圏外", f"{amb_0/total*100:.1f}%" if total > 0 else "0%", delta_color="inverse")
                    with col2:
                        st.metric(f"{minutes}分 1台カバー", f"{amb_1/total*100:.1f}%" if total > 0 else "0%")
                    with col3:
                        st.metric(f"{minutes}分 冗長性あり", f"{amb_2/total*100:.1f}%" if total > 0 else "0%")
                
                st.info("""
                **リソース考慮の意味**:
                - **圏外 (0台)**: 指定時間内に到達可能な救急車がない
                - **1台のみ**: カバーされているが、その1台が出動中だと対応不可
                - **2台以上**: 冗長性あり。1台が出動中でも別の車両で対応可能
                """)
            # ================================================

            # Render map with coverage visualization
            st.markdown("---")
            st.subheader("🗺️ カバー状況マップ")

            center_lat_cov = mapped_cov["lat"].mean()
            center_lon_cov = mapped_cov["lon"].mean()
            fmap_cov = folium.Map(location=[center_lat_cov, center_lon_cov], zoom_start=11, tiles="CartoDB Positron")

            # Add isochrone layers
            color_map_cov = {5: "#ff9e9e", 10: "#8aa5ff"}
            for minutes in trip_times_cov:
                iso_layer = isochrones_cov[isochrones_cov["time"] == minutes]
                if iso_layer.empty:
                    continue
                color = color_map_cov.get(minutes, "#4a4a4a")
                folium.GeoJson(
                    data=iso_layer.__geo_interface__,
                    name=f"{minutes}分到達圏",
                    style_function=lambda _f, c=color, m=minutes: {
                        "fillColor": c,
                        "color": c,
                        "weight": 1.0,
                        "opacity": 0.5,
                        "fillOpacity": 0.15 if m >= 10 else 0.25,
                    },
                ).add_to(fmap_cov)

            # Add station markers
            for _, row in stations_cov.iterrows():
                folium.CircleMarker(
                    location=[row["緯度"], row["経度"]],
                    radius=6,
                    color="#1f1f1f",
                    weight=2,
                    fill=True,
                    fill_color="#f6bd60",
                    fill_opacity=0.9,
                    popup=f"{row['略称']}",
                ).add_to(fmap_cov)

            # Add incident markers with coverage status
            for _, row in incident_points.iterrows():
                within_5 = row.get("within_5min", False)
                within_10 = row.get("within_10min", False)

                if within_5:
                    color = "#2ecc71"  # Green - covered by 5min
                    status = "5分内"
                elif within_10:
                    color = "#f39c12"  # Orange - covered by 10min
                    status = "10分内"
                else:
                    color = "#e74c3c"  # Red - not covered
                    status = "圏外"

                label_time = row["覚知"].strftime("%H:%M") if not pd.isna(row.get("覚知")) else "--:--"
                popup = f"{status} | {row.get('出動隊', '不明')} | {label_time}"
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    weight=1.0,
                    popup=popup,
                ).add_to(fmap_cov)

            # Add legend
            legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                        background-color: white; padding: 10px; border-radius: 5px;
                        border: 2px solid gray; font-size: 12px;">
                <b>出動地点</b><br>
                <span style="color: #2ecc71;">●</span> 5分内到達<br>
                <span style="color: #f39c12;">●</span> 10分内到達<br>
                <span style="color: #e74c3c;">●</span> 到達圏外
            </div>
            '''
            fmap_cov.get_root().html.add_child(folium.Element(legend_html))

            folium.LayerControl(collapsed=False).add_to(fmap_cov)
            st.components.v1.html(fmap_cov.get_root().render(), height=720)

            # Show uncovered incidents detail
            if "within_10min" in incident_points.columns:
                uncovered = incident_points[~incident_points["within_10min"]]
                if not uncovered.empty:
                    st.markdown("---")
                    st.subheader(f"⚠️ 10分到達圏外の出動 ({len(uncovered)} 件)")
                    uncovered_display = uncovered[["date", "覚知", "出動場所", "出動隊"]].copy()
                    uncovered_display.columns = ["日付", "覚知時刻", "出動場所", "出動隊"]
                    st.dataframe(uncovered_display, width="stretch", hide_index=True)

    # ========================================
    # 🚑 リソース分析タブ
    # ========================================
    with tab_resource:
        st.header("🚑 リソースベース カバレッジ分析")
        st.markdown("""
        各地点で「**n分以内に到達可能な救急車が何台あるか**」を分析し、
        リソース配置の最適化提案を行います。
        """)
        
        # キャッシュ確認
        from misc.coverage_analysis import (
            load_coverage_cache,
            compute_coverage_quality,
            compute_optimization_suggestions,
            load_stations as load_stations_with_resources,
            create_coverage_map,
            STATION_RESOURCES,
        )
        
        cache = load_coverage_cache()
        
        if cache is None:
            st.warning("⚠️ カバレッジ分析のキャッシュがないか、読み込みに失敗しました（NumPyバージョン不整合の可能性）。")
            st.code("python3 misc/coverage_analysis.py", language="bash")
            st.info("上記コマンドを実行してカバレッジ分析を事前計算（または再生成）してください。")
        else:
            grid, travel_times = cache
            
            # 消防署データ（リソース情報付き）を読み込み
            stations_res = load_stations_with_resources()
            
            # 閾値選択
            threshold_min = st.selectbox(
                "到達時間の閾値",
                options=[5, 8, 10],
                index=1,
                format_func=lambda x: f"{x}分以内",
            )
            
            # カバレッジ計算
            grid = compute_coverage_quality(travel_times, grid, [5, 8, 10])
            col = f"ambulances_{threshold_min}min"
            
            # 統計表示
            st.subheader("📊 現状のカバレッジ状況")
            
            col1, col2, col3, col4 = st.columns(4)
            total_points = len(grid)
            zero_cov = (grid[col] == 0).sum()
            single_cov = (grid[col] == 1).sum()
            multi_cov = (grid[col] >= 2).sum()
            
            with col1:
                st.metric("分析ポイント数", f"{total_points:,}")
            with col2:
                st.metric("カバレッジなし", f"{zero_cov:,}", delta=f"{zero_cov/total_points*100:.1f}%", delta_color="inverse")
            with col3:
                st.metric("1台のみ", f"{single_cov:,}", delta=f"{single_cov/total_points*100:.1f}%", delta_color="off")
            with col4:
                st.metric("2台以上", f"{multi_cov:,}", delta=f"{multi_cov/total_points*100:.1f}%", delta_color="normal")
            
            # リソース配置表示
            st.subheader("🏥 消防署別リソース配置")
            resource_df = stations_res[["略称", "救急車台数", "区分"]].copy()
            resource_df.columns = ["消防署", "救急車台数", "区分"]
            st.dataframe(resource_df, width="stretch", hide_index=True)
            st.caption(f"合計: {stations_res['救急車台数'].sum()}台")
            
            # 最適化提案
            st.subheader("💡 リソース配置 最適化提案")
            suggestions = compute_optimization_suggestions(
                grid, stations_res, travel_times, target_threshold_min=threshold_min
            )
            
            # 弱点エリア
            if suggestions["weak_areas"]:
                st.markdown("**⚠️ 弱点エリア:**")
                for area in suggestions["weak_areas"]:
                    if area["severity"] == "高":
                        st.error(f"🔴 {area['type']}: {area['count']}ポイント")
                    else:
                        st.warning(f"🟡 {area['type']}: {area['count']}ポイント")
            
            # 増強推奨
            st.markdown("**📈 救急車1台追加時の改善効果（上位5署）:**")
            for i, s in enumerate(suggestions["suggestions"], 1):
                improvement_score = s["total_improvement"]
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                st.markdown(
                    f"{emoji} **{i}. {s['station_name']}** (現{s['current_ambulances']}台) "
                    f"→ 新規カバー: {s['newly_covered_points']}pt, 冗長性追加: {s['redundancy_improved_points']}pt"
                )
            
            # ヒートマップ表示
            st.subheader("🗺️ カバレッジマップ")
            st.markdown(f"**{threshold_min}分以内に到達可能な救急車台数**")
            
            with st.spinner("マップ生成中..."):
                coverage_map = create_coverage_map(grid, stations_res, threshold_min)
                st.components.v1.html(coverage_map.get_root().render(), height=600)
            
            st.markdown("---")
            st.caption("""
            **凡例**: 🔴 0台（カバレッジなし）, 🟠 1台（冗長性なし）, 🟡 2台, 🟢 3台以上  
            **データソース**: R6出動データから各消防署の救急車台数を推定
            """)

    # ========================================
    # ⭐ 配置最適化タブ
    # ========================================
    with tab_optimize:
        st.header("⭐ リソース考慮 配置最適化")
        st.markdown("""
        出動データと既存リソースを分析し、**新規消防署の最適な候補地点**を自動で提案します。
        
        **特徴:**
        - 📍 出動密度とカバレッジギャップから候補地点を自動抽出
        - 🚑 既存の救急車配置を考慮したシミュレーション
        - ⚡ 高速な貪欲法アルゴリズム（デモ向け）
        """)
        
        # インポート（遅延読み込み）
        try:
            from optimization import (
                load_stations as opt_load_stations,
                load_incident_locations,
                generate_candidate_locations,
                optimize_placement,
                create_optimization_map,
                CandidateLocation,
                load_candidates_cache,
                save_candidates_cache,
            )
            optimization_available = True
        except ImportError as e:
            optimization_available = False
            st.error(f"最適化モジュールの読み込みに失敗しました: {e}")
        
        if optimization_available:
            # 設定
            st.subheader("⚙️ 最適化設定")
            
            col_set1, col_set2, col_set3 = st.columns(3)
            with col_set1:
                n_candidates = st.slider(
                    "候補地点数",
                    min_value=5,
                    max_value=20,
                    value=10,
                    help="生成する候補地点の数。多いほど精度が上がりますが計算時間が増えます。"
                )
            with col_set2:
                threshold_min_opt = st.selectbox(
                    "到達時間の閾値",
                    options=[5, 8, 10],
                    index=1,
                    format_func=lambda x: f"{x}分以内",
                    help="この時間内に到達できることを目標とします。"
                )
            with col_set3:
                new_ambulances = st.number_input(
                    "新規消防署の救急車台数",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="新しく設置する消防署に配備する救急車の台数。"
                )
            
            use_cache = st.checkbox("キャッシュを使用（高速化）", value=True, help="以前の計算結果を再利用します。")
            
            # 実行ボタン
            st.markdown("---")
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                run_optimization = st.button("🚀 最適化を実行", type="primary")
            with col_btn2:
                if st.button("🗑️ キャッシュクリア", type="secondary"):
                    from optimization import OPTIMIZATION_CACHE_PATH
                    if OPTIMIZATION_CACHE_PATH.exists():
                        OPTIMIZATION_CACHE_PATH.unlink()
                        st.success("キャッシュをクリアしました。")
                        st.rerun()
            
            if run_optimization:
                st.markdown("---")
                
                # データ読み込み
                with st.spinner("データを読み込み中..."):
                    stations_opt = opt_load_stations()
                    incidents_opt = load_incident_locations()
                
                if incidents_opt.empty:
                    st.error("出動データが見つかりません。先にジオコーディングを実行してください。")
                    st.code("python scripts/precompute_incident_geocode.py", language="bash")
                    st.stop()
                
                st.info(f"📊 分析対象: 消防署 {len(stations_opt)}箇所, 出動地点 {len(incidents_opt):,}件")
                
                # 候補地点生成
                st.subheader("📍 候補地点の抽出")
                
                candidates = None
                if use_cache:
                    candidates = load_candidates_cache()
                    if candidates:
                        st.success(f"✅ キャッシュから{len(candidates)}候補を読み込みました")
                
                if candidates is None:
                    with st.spinner("候補地点を生成中..."):
                        prog_cand = st.progress(0)
                        candidates = generate_candidate_locations(
                            stations_opt, incidents_opt,
                            n_candidates=n_candidates,
                            progress_cb=lambda p: prog_cand.progress(int(p * 100)),
                        )
                        save_candidates_cache(candidates)
                        st.success(f"✅ {len(candidates)}候補地点を生成しました")
                
                # 候補一覧表示
                if candidates:
                    candidate_data = []
                    for i, c in enumerate(candidates, 1):
                        candidate_data.append({
                            "順位": i,
                            "緯度": f"{c.lat:.5f}",
                            "経度": f"{c.lon:.5f}",
                            "理由": c.reason,
                            "出動密度": f"{c.incident_density:.2f}",
                            "ギャップ": f"{c.current_coverage_gap:.2f}",
                            "スコア": f"{c.priority_score:.3f}",
                        })
                    
                    with st.expander(f"📋 候補地点一覧（{len(candidates)}件）", expanded=False):
                        st.dataframe(pd.DataFrame(candidate_data), width="stretch", hide_index=True)
                
                # 最適化実行
                st.subheader("🎯 最適化シミュレーション")
                
                with st.spinner("シミュレーション実行中..."):
                    prog_opt = st.progress(0)
                    result = optimize_placement(
                        stations_opt, incidents_opt, candidates,
                        threshold_min=threshold_min_opt,
                        new_ambulances=new_ambulances,
                        progress_cb=lambda p: prog_opt.progress(int(p * 100)),
                    )
                
                st.success(f"✅ 最適化完了（{result.computation_time_sec:.2f}秒）")
                
                # 結果表示
                st.markdown("---")
                st.subheader("⭐ 最適化結果")
                
                if result.best_location:
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.markdown("### 🏆 最適候補地点")
                        st.markdown(f"""
                        - **位置**: ({result.best_location['lat']:.5f}, {result.best_location['lon']:.5f})
                        - **選定理由**: {result.best_location['reason']}
                        - **新規カバー件数**: {result.best_location['newly_covered_incidents']:,}件
                        - **効率スコア**: {result.best_location['efficiency_score']:.1f}件/台
                        """)
                        
                        # Google Maps リンク
                        gmap_url = f"https://www.google.com/maps?q={result.best_location['lat']},{result.best_location['lon']}"
                        st.markdown(f"[📍 Google Mapsで開く]({gmap_url})")
                    
                    with col_res2:
                        st.markdown("### 📈 改善効果")
                        st.metric(
                            "新規カバー件数",
                            f"{result.best_location['newly_covered_incidents']:,}件",
                            delta=f"+{result.best_location['efficiency_score']:.1f}件/救急車1台"
                        )
                        st.metric(
                            "投入リソース",
                            f"救急車 {new_ambulances}台",
                        )
                        st.info(f"""
                        **リソース効率**: 救急車1台あたり約{result.best_location['efficiency_score']:.0f}件の
                        出動をカバー可能になります。
                        """)
                
                else:
                    st.warning("最適な候補地点を特定できませんでした。")
                
                # マップ表示
                st.subheader("🗺️ 最適化結果マップ")
                
                with st.spinner("マップ生成中..."):
                    opt_map = create_optimization_map(
                        stations_opt, incidents_opt, candidates, result.best_location
                    )
                    st.components.v1.html(opt_map.get_root().render(), height=600)
                
                st.caption("""
                **凡例**: 📍青=既存消防署, ⭐赤=最適候補地点, ●橙=その他候補, ・灰=出動地点  
                **赤い円**: 8分到達圏（概算）
                """)
                
                # 詳細結果
                with st.expander("📊 詳細分析データ", expanded=False):
                    st.markdown("**カバレッジ改善:**")
                    st.json(result.coverage_improvement)
                    
                    st.markdown("**リソース効率:**")
                    st.json(result.resource_efficiency)
                    
                    st.markdown("**全候補地点スコア:**")
                    all_candidates_df = pd.DataFrame(result.candidate_locations)
                    st.dataframe(all_candidates_df, width="stretch")
            
            else:
                # 実行前の説明
                st.markdown("---")
                st.info("""
                **使い方:**
                1. 上記の設定を調整（デフォルトでもOK）
                2. 「🚀 最適化を実行」ボタンをクリック
                3. 候補地点の生成 → シミュレーション → 結果表示
                
                **所要時間**: 約10〜30秒（キャッシュ使用時は数秒）
                """)
                
                # 現在のリソース配置を表示
                st.subheader("📊 現在のリソース配置")
                try:
                    stations_preview = opt_load_stations()
                    preview_df = stations_preview[["略称", "救急車台数", "区分"]].copy()
                    preview_df.columns = ["消防署", "救急車台数", "区分"]
                    
                    col_p1, col_p2 = st.columns([2, 1])
                    with col_p1:
                        st.dataframe(preview_df, width="stretch", hide_index=True)
                    with col_p2:
                        st.metric("総救急車台数", f"{preview_df['救急車台数'].sum()}台")
                        st.metric("消防署数", f"{len(preview_df)}箇所")
                except Exception as e:
                    st.warning(f"プレビュー表示に失敗: {e}")

    st.info("アプリを終了するには、実行中のターミナルで Ctrl+C を押してください。")


if __name__ == "__main__":
    main()
