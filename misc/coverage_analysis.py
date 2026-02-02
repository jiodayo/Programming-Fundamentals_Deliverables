"""リソースベースのカバレッジ分析モジュール

機能:
- 各地点から「n分以内に到達可能な救急車台数」を計算
- カバレッジ品質のヒートマップ生成
- リソース配置の最適化提案
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import folium
from shapely.geometry import Point, box

# 各消防署の救急車台数（R6データから）
STATION_RESOURCES = {
    "東消防署": 4,
    "中央消防署": 3,
    "西消防署": 3,
    "南消防署": 3,
    "城北支署": 1,  # 支署は1台想定
    "城東支署": 1,
    "西部支署": 1,
    "東部支署": 1,
    "北条支署": 1,
    "湯山出張所": 1,
    "久谷出張所": 1,
    "消防局": 1,  # 消防局も1台として計算
    "WS": 1,  # ワークステーション
}

# デフォルト台数（不明な署の場合）
DEFAULT_AMBULANCES = 1

CACHE_DIR = Path(__file__).parent.parent / "cache"
GRAPH_PATH = CACHE_DIR / "matsuyama_drive.graphml"  # または ehime_drive.graphml


def load_stations() -> gpd.GeoDataFrame:
    """消防署データを読み込み"""
    db_path = Path(__file__).parent.parent / "map.sqlite"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM stations", conn)
    
    # リソース情報を追加
    df["救急車台数"] = df["略称"].map(STATION_RESOURCES).fillna(DEFAULT_AMBULANCES).astype(int)
    
    geometry = [Point(lon, lat) for lon, lat in zip(df["経度"], df["緯度"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def load_graph() -> nx.MultiDiGraph:
    """道路ネットワークグラフを読み込み"""
    if GRAPH_PATH.exists():
        return ox.load_graphml(GRAPH_PATH)
    raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")


def generate_grid_points(
    bounds: tuple[float, float, float, float],
    resolution_km: float = 1.0,
) -> gpd.GeoDataFrame:
    """分析用のグリッドポイントを生成
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        resolution_km: グリッド間隔（km）
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # 緯度経度をkmに変換（おおよそ）
    lat_step = resolution_km / 111.0  # 1度 ≈ 111km
    lon_step = resolution_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
    
    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    
    points = []
    for lat in lats:
        for lon in lons:
            points.append({"lat": lat, "lon": lon, "geometry": Point(lon, lat)})
    
    return gpd.GeoDataFrame(points, crs="EPSG:4326")


def compute_travel_times_from_stations(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    max_time_sec: float = 600,  # 10分
) -> pd.DataFrame:
    """各消防署から各グリッドポイントへの到達時間を計算
    
    Returns:
        DataFrame with columns: grid_idx, station_name, travel_time_sec, ambulances
    """
    # 消防署の最寄りノード
    station_nodes = ox.distance.nearest_nodes(
        graph, 
        stations["経度"].tolist(), 
        stations["緯度"].tolist()
    )
    
    # グリッドの最寄りノード
    grid_nodes = ox.distance.nearest_nodes(
        graph,
        grid["lon"].tolist(),
        grid["lat"].tolist()
    )
    
    results = []
    
    for i, (_, station) in enumerate(stations.iterrows()):
        print(f"  計算中: {station['略称']} ({i+1}/{len(stations)})")
        station_node = station_nodes[i]
        
        # 消防署からの最短経路（逆方向なので注意）
        # 救急車は「消防署から現場へ」向かうので、消防署を起点とした最短経路
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph,
                station_node,
                cutoff=max_time_sec,
                weight="travel_time",
            )
        except Exception:
            continue
        
        for j, grid_node in enumerate(grid_nodes):
            if grid_node in lengths:
                travel_time = lengths[grid_node]
                results.append({
                    "grid_idx": j,
                    "station_name": station["略称"],
                    "travel_time_sec": travel_time,
                    "ambulances": station["救急車台数"],
                })
    
    return pd.DataFrame(results)


def compute_coverage_quality(
    travel_times: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    time_thresholds: list[int] = [5, 8, 10],  # 分
) -> gpd.GeoDataFrame:
    """各グリッドポイントのカバレッジ品質を計算
    
    カバレッジ品質 = n分以内に到達可能な救急車の合計台数
    """
    grid = grid.copy()
    
    for threshold_min in time_thresholds:
        threshold_sec = threshold_min * 60
        col_name = f"ambulances_{threshold_min}min"
        
        # 各グリッドポイントで、閾値以内に到達可能な救急車台数を集計
        reachable = travel_times[travel_times["travel_time_sec"] <= threshold_sec]
        
        # 同じ署からの重複を排除し、台数を合計
        ambulance_counts = (
            reachable.groupby("grid_idx")
            .apply(lambda x: x.drop_duplicates("station_name")["ambulances"].sum())
        )
        
        grid[col_name] = grid.index.map(ambulance_counts).fillna(0).astype(int)
    
    return grid


def compute_optimization_suggestions(
    grid: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    travel_times: pd.DataFrame,
    target_threshold_min: int = 8,
) -> dict:
    """リソース配置の最適化提案を生成
    
    Returns:
        dict with:
        - current_stats: 現状の統計
        - weak_areas: カバレッジが弱いエリア
        - suggestions: 改善提案
    """
    col = f"ambulances_{target_threshold_min}min"
    
    # 現状統計
    current_stats = {
        "total_grid_points": len(grid),
        "zero_coverage": int((grid[col] == 0).sum()),
        "single_coverage": int((grid[col] == 1).sum()),
        "multi_coverage": int((grid[col] >= 2).sum()),
        "avg_ambulances": float(grid[col].mean()),
        "min_ambulances": int(grid[col].min()),
        "max_ambulances": int(grid[col].max()),
    }
    
    # カバレッジ0のエリア
    zero_coverage_points = grid[grid[col] == 0].copy()
    
    # カバレッジ1のエリア（冗長性がない）
    single_coverage_points = grid[grid[col] == 1].copy()
    
    # 弱点エリアの中心を計算
    weak_areas = []
    if len(zero_coverage_points) > 0:
        weak_areas.append({
            "type": "カバレッジなし",
            "count": len(zero_coverage_points),
            "center_lat": zero_coverage_points["lat"].mean(),
            "center_lon": zero_coverage_points["lon"].mean(),
            "severity": "高",
        })
    
    if len(single_coverage_points) > 0:
        weak_areas.append({
            "type": "冗長性なし（1台のみ）",
            "count": len(single_coverage_points),
            "center_lat": single_coverage_points["lat"].mean(),
            "center_lon": single_coverage_points["lon"].mean(),
            "severity": "中",
        })
    
    # 各消防署の増強効果をシミュレーション
    suggestions = []
    for _, station in stations.iterrows():
        station_name = station["略称"]
        
        # この署から到達可能なポイントを取得
        station_times = travel_times[travel_times["station_name"] == station_name]
        reachable_in_threshold = station_times[
            station_times["travel_time_sec"] <= target_threshold_min * 60
        ]["grid_idx"].unique()
        
        # 現在カバレッジ0のポイントでこの署が救える数
        zero_points_idx = set(grid[grid[col] == 0].index)
        newly_covered = len(zero_points_idx & set(reachable_in_threshold))
        
        # 現在カバレッジ1のポイントで冗長性を追加できる数
        single_points_idx = set(grid[grid[col] == 1].index)
        redundancy_added = len(single_points_idx & set(reachable_in_threshold))
        
        suggestions.append({
            "station_name": station_name,
            "current_ambulances": int(station["救急車台数"]),
            "newly_covered_points": newly_covered,
            "redundancy_improved_points": redundancy_added,
            "total_improvement": newly_covered * 2 + redundancy_added,  # 重み付き
        })
    
    # 改善効果が高い順にソート
    suggestions = sorted(suggestions, key=lambda x: x["total_improvement"], reverse=True)
    
    return {
        "current_stats": current_stats,
        "weak_areas": weak_areas,
        "suggestions": suggestions[:5],  # 上位5つ
    }


def save_coverage_cache(
    grid: gpd.GeoDataFrame,
    travel_times: pd.DataFrame,
    cache_name: str = "coverage_analysis",
):
    """分析結果をキャッシュ"""
    CACHE_DIR.mkdir(exist_ok=True)
    
    grid.to_pickle(CACHE_DIR / f"{cache_name}_grid.pkl")
    travel_times.to_pickle(CACHE_DIR / f"{cache_name}_times.pkl")
    print(f"✅ キャッシュ保存: {CACHE_DIR / cache_name}_*.pkl")


def load_coverage_cache(
    cache_name: str = "coverage_analysis",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame] | None:
    """キャッシュから読み込み"""
    grid_path = CACHE_DIR / f"{cache_name}_grid.pkl"
    times_path = CACHE_DIR / f"{cache_name}_times.pkl"
    
    if grid_path.exists() and times_path.exists():
        try:
            grid = pd.read_pickle(grid_path)
            travel_times = pd.read_pickle(times_path)
            return grid, travel_times
        except (ModuleNotFoundError, ImportError, pickle.UnpicklingError) as e:
            # NumPyバージョン不整合などでpickle読み込み失敗
            print(f"⚠️ キャッシュ読み込みエラー（バージョン不整合の可能性）: {e}")
            print("  → キャッシュを再生成してください: python3 misc/coverage_analysis.py")
            return None
        except Exception as e:
            print(f"⚠️ キャッシュ読み込みエラー: {e}")
            return None
    return None


def create_coverage_map(
    grid: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    threshold_min: int = 8,
) -> "folium.Map":
    """カバレッジヒートマップを作成
    
    Args:
        grid: カバレッジ計算済みのグリッド
        stations: 消防署データ
        threshold_min: 表示する閾値（分）
    """
    import folium
    from folium.plugins import HeatMap
    
    col = f"ambulances_{threshold_min}min"
    
    # 地図の中心
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )
    
    # カバレッジに応じた色分け
    # 0台: 赤, 1台: オレンジ, 2台: 黄, 3台以上: 緑
    colors = {0: "red", 1: "orange", 2: "yellow"}
    
    for _, row in grid.iterrows():
        ambulances = row[col]
        if ambulances == 0:
            color = "red"
            opacity = 0.7
        elif ambulances == 1:
            color = "orange"
            opacity = 0.5
        elif ambulances == 2:
            color = "yellow"
            opacity = 0.4
        else:
            color = "green"
            opacity = 0.3
        
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            opacity=opacity,
            popup=f"救急車: {ambulances}台",
        ).add_to(m)
    
    # 消防署マーカー
    for _, station in stations.iterrows():
        folium.Marker(
            location=[station["緯度"], station["経度"]],
            popup=f"{station['略称']} ({station['救急車台数']}台)",
            icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
        ).add_to(m)
    
    # 凡例
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>{threshold_min}分以内到達可能な救急車台数</b><br>
        <span style="color: red;">●</span> 0台（カバレッジなし）<br>
        <span style="color: orange;">●</span> 1台（冗長性なし）<br>
        <span style="color: yellow;">●</span> 2台<br>
        <span style="color: green;">●</span> 3台以上
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def main():
    print("🚑 リソースベース カバレッジ分析")
    print("=" * 50)
    
    # データ読み込み
    print("\n📍 消防署データ読み込み...")
    stations = load_stations()
    print(f"   {len(stations)}署")
    print("\n   救急車配置:")
    for _, s in stations.iterrows():
        print(f"   - {s['略称']}: {s['救急車台数']}台")
    print(f"   合計: {stations['救急車台数'].sum()}台")
    
    # キャッシュ確認
    cache = load_coverage_cache()
    if cache:
        print("\n📦 キャッシュから読み込み...")
        grid, travel_times = cache
    else:
        print("\n🗺️ 道路グラフ読み込み...")
        graph = load_graph()
        
        # グリッド生成
        print("\n📊 分析グリッド生成...")
        bounds = (
            stations["経度"].min() - 0.05,
            stations["緯度"].min() - 0.05,
            stations["経度"].max() + 0.05,
            stations["緯度"].max() + 0.05,
        )
        grid = generate_grid_points(bounds, resolution_km=0.5)
        print(f"   {len(grid)}ポイント")
        
        # 到達時間計算
        print("\n⏱️ 到達時間計算...")
        travel_times = compute_travel_times_from_stations(
            graph, stations, grid, max_time_sec=900  # 15分
        )
        print(f"   {len(travel_times)}レコード")
        
        # キャッシュ保存
        save_coverage_cache(grid, travel_times)
    
    # カバレッジ品質計算
    print("\n📈 カバレッジ品質計算...")
    grid = compute_coverage_quality(travel_times, grid, [5, 8, 10])
    
    # 結果表示
    for threshold in [5, 8, 10]:
        col = f"ambulances_{threshold}min"
        print(f"\n   【{threshold}分以内到達】")
        print(f"   - 0台: {(grid[col] == 0).sum()}ポイント")
        print(f"   - 1台: {(grid[col] == 1).sum()}ポイント")
        print(f"   - 2台以上: {(grid[col] >= 2).sum()}ポイント")
        print(f"   - 平均: {grid[col].mean():.2f}台")
    
    # 最適化提案
    print("\n" + "=" * 50)
    print("🎯 リソース配置 最適化提案")
    print("=" * 50)
    
    suggestions = compute_optimization_suggestions(grid, stations, travel_times, target_threshold_min=8)
    
    stats = suggestions["current_stats"]
    print(f"\n📊 現状（8分圏）:")
    print(f"   - カバレッジなし: {stats['zero_coverage']}ポイント ({stats['zero_coverage']/stats['total_grid_points']*100:.1f}%)")
    print(f"   - 1台のみ: {stats['single_coverage']}ポイント ({stats['single_coverage']/stats['total_grid_points']*100:.1f}%)")
    print(f"   - 2台以上: {stats['multi_coverage']}ポイント ({stats['multi_coverage']/stats['total_grid_points']*100:.1f}%)")
    
    if suggestions["weak_areas"]:
        print(f"\n⚠️ 弱点エリア:")
        for area in suggestions["weak_areas"]:
            print(f"   - {area['type']}: {area['count']}ポイント（中心: {area['center_lat']:.4f}, {area['center_lon']:.4f}）")
    
    print(f"\n💡 増強推奨（救急車1台追加時の効果）:")
    for i, s in enumerate(suggestions["suggestions"], 1):
        print(f"   {i}. {s['station_name']} (現{s['current_ambulances']}台)")
        print(f"      → 新規カバー: {s['newly_covered_points']}pt, 冗長性追加: {s['redundancy_improved_points']}pt")
    
    # 結果をJSONで保存
    result = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "stats": stats,
        "weak_areas": suggestions["weak_areas"],
        "suggestions": suggestions["suggestions"],
    }
    with open(CACHE_DIR / "coverage_analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 結果保存: {CACHE_DIR / 'coverage_analysis_result.json'}")


if __name__ == "__main__":
    main()
