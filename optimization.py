"""リソースを考慮した配置最適化モジュール

機能:
- 出動データの密度分析から候補地点を自動生成
- 既存リソース（救急車台数）を考慮した最適配置シミュレーション
- 高速な貪欲法ベースの最適化アルゴリズム
- デモンストレーション向けの高速動作

作成: 2026/02/02
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union

# =============================================================================
# 定数
# =============================================================================

CACHE_DIR = Path("cache")
GRAPH_PATH = CACHE_DIR / "matsuyama_drive.graphml"
STATIONS_DB_PATH = Path("map.sqlite")
INCIDENTS_DB_PATH = Path("incidents.sqlite")
GEOCODE_CACHE_PATH = CACHE_DIR / "incident_geocode.parquet"

# 各消防署の救急車台数（実データ推定）
STATION_RESOURCES = {
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

# 新規消防署の想定リソース
NEW_STATION_AMBULANCES = 2

# デフォルト台数
DEFAULT_AMBULANCES = 1


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class OptimizationResult:
    """最適化結果を格納するデータクラス"""
    candidate_locations: list[dict]  # 候補地点リスト
    best_location: dict | None  # 最適地点
    coverage_improvement: dict  # カバレッジ改善
    resource_efficiency: dict  # リソース効率
    computation_time_sec: float  # 計算時間


@dataclass
class CandidateLocation:
    """候補地点"""
    lat: float
    lon: float
    name: str
    reason: str  # 候補理由
    priority_score: float  # 優先度スコア（高いほど良い）
    incident_density: float  # 周辺出動密度
    current_coverage_gap: float  # 現在のカバレッジギャップ


# =============================================================================
# ユーティリティ関数
# =============================================================================

def load_graph() -> nx.MultiDiGraph:
    """道路ネットワークグラフを読み込み"""
    if GRAPH_PATH.exists():
        return ox.load_graphml(GRAPH_PATH)
    raise FileNotFoundError(f"Graph not found: {GRAPH_PATH}")


def load_stations() -> gpd.GeoDataFrame:
    """消防署データを読み込み（リソース情報付き）"""
    if STATIONS_DB_PATH.exists():
        with sqlite3.connect(STATIONS_DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM stations", conn)
    else:
        df = pd.read_excel("map.xlsx")
    
    df["救急車台数"] = df["略称"].map(STATION_RESOURCES).fillna(DEFAULT_AMBULANCES).astype(int)
    df["区分"] = df["略称"].apply(lambda x: "署" if "消防署" in x or "消防局" in x else "支署・出張所")
    
    geometry = gpd.points_from_xy(df["経度"], df["緯度"])
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def load_incident_locations() -> gpd.GeoDataFrame:
    """出動地点のジオコーディング済みデータを読み込み"""
    if GEOCODE_CACHE_PATH.exists():
        df = pd.read_parquet(GEOCODE_CACHE_PATH)
        df = df.dropna(subset=["lat", "lon"])
        geometry = gpd.points_from_xy(df["lon"], df["lat"])
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gpd.GeoDataFrame(columns=["address", "lat", "lon", "geometry"], crs="EPSG:4326")


# =============================================================================
# 候補地点生成ロジック
# =============================================================================

def generate_candidate_locations(
    stations: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    travel_times_cache: dict | None = None,
    n_candidates: int = 10,
    resolution_km: float = 0.5,
    progress_cb: Callable[[float], None] | None = None,
) -> list[CandidateLocation]:
    """候補地点を自動生成
    
    生成ロジック:
    1. 出動密度が高い地点を抽出
    2. 現在のカバレッジが弱い地点を抽出
    3. 両者を組み合わせて優先度スコアを計算
    
    Args:
        stations: 消防署データ
        incidents: 出動地点データ
        travel_times_cache: 事前計算した到達時間キャッシュ
        n_candidates: 生成する候補数
        resolution_km: グリッド解像度（km）
        progress_cb: 進捗コールバック
    
    Returns:
        優先度順にソートされた候補地点リスト
    """
    start_time = time.time()
    
    if incidents.empty:
        return []
    
    # 分析範囲を決定
    bounds = _compute_bounds(stations, incidents, buffer_km=3.0)
    min_lon, min_lat, max_lon, max_lat = bounds
    
    if progress_cb:
        progress_cb(0.1)
    
    # Step 1: 出動密度マップを作成
    density_map = _compute_incident_density_map(
        incidents, bounds, resolution_km
    )
    
    if progress_cb:
        progress_cb(0.3)
    
    # Step 2: 現在のカバレッジギャップを計算
    coverage_gap_map = _compute_coverage_gap_map(
        stations, bounds, resolution_km, travel_times_cache
    )
    
    if progress_cb:
        progress_cb(0.6)
    
    # Step 3: 複合スコアを計算し候補を抽出
    candidates = _extract_candidates(
        density_map, coverage_gap_map, stations,
        bounds, resolution_km, n_candidates
    )
    
    if progress_cb:
        progress_cb(1.0)
    
    elapsed = time.time() - start_time
    print(f"候補地点生成完了: {len(candidates)}地点 ({elapsed:.2f}秒)")
    
    return candidates


def _compute_bounds(
    stations: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    buffer_km: float = 3.0,
) -> tuple[float, float, float, float]:
    """分析範囲を計算"""
    all_lons = list(stations["経度"]) + list(incidents["lon"])
    all_lats = list(stations["緯度"]) + list(incidents["lat"])
    
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(np.mean(all_lats))))
    
    return (
        min(all_lons) - lon_buffer,
        min(all_lats) - lat_buffer,
        max(all_lons) + lon_buffer,
        max(all_lats) + lat_buffer,
    )


def _compute_incident_density_map(
    incidents: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    resolution_km: float,
) -> dict:
    """出動密度マップを作成（KDE）"""
    min_lon, min_lat, max_lon, max_lat = bounds
    
    lat_step = resolution_km / 111.0
    lon_step = resolution_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
    
    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    
    # グリッドを作成
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 出動地点の座標
    incident_coords = np.column_stack([incidents["lon"], incidents["lat"]])
    
    if len(incident_coords) == 0:
        return {
            "density": np.zeros_like(lon_grid),
            "lats": lats,
            "lons": lons,
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
        }
    
    # KDTreeで近傍密度を高速計算
    tree = cKDTree(incident_coords)
    
    density = np.zeros_like(lon_grid)
    bandwidth_km = 2.0  # カーネル帯域幅
    bandwidth_deg = bandwidth_km / 111.0
    
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = [lon_grid[i, j], lat_grid[i, j]]
            # 近傍の出動件数をカウント
            neighbors = tree.query_ball_point(point, bandwidth_deg)
            density[i, j] = len(neighbors)
    
    # 正規化
    if density.max() > 0:
        density = density / density.max()
    
    return {
        "density": density,
        "lats": lats,
        "lons": lons,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }


def _compute_coverage_gap_map(
    stations: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    resolution_km: float,
    travel_times_cache: dict | None = None,
) -> dict:
    """カバレッジギャップマップを作成
    
    簡易版: 各グリッドから最寄り消防署までの距離をベースに計算
    （正確な到達時間はキャッシュがある場合のみ使用）
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    lat_step = resolution_km / 111.0
    lon_step = resolution_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))
    
    lats = np.arange(min_lat, max_lat, lat_step)
    lons = np.arange(min_lon, max_lon, lon_step)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # 消防署座標
    station_coords = np.column_stack([stations["経度"], stations["緯度"]])
    station_ambulances = stations["救急車台数"].values
    
    tree = cKDTree(station_coords)
    
    # 各グリッドポイントで「8分圏内の救急車台数」を推定
    coverage = np.zeros_like(lon_grid)
    target_distance_km = 8 * 40 / 60  # 8分 × 40km/h ≈ 5.3km
    target_distance_deg = target_distance_km / 111.0
    
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = [lon_grid[i, j], lat_grid[i, j]]
            
            # 到達可能な消防署を特定
            indices = tree.query_ball_point(point, target_distance_deg)
            
            # 救急車台数を合計
            total_ambulances = sum(station_ambulances[idx] for idx in indices)
            coverage[i, j] = total_ambulances
    
    # カバレッジギャップ（低いほどギャップが大きい）
    # 2台以上を目標として、不足分をギャップとする
    target_ambulances = 2
    gap = np.maximum(0, target_ambulances - coverage)
    
    # 正規化
    if gap.max() > 0:
        gap = gap / gap.max()
    
    return {
        "gap": gap,
        "coverage": coverage,
        "lats": lats,
        "lons": lons,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }


def _extract_candidates(
    density_map: dict,
    coverage_gap_map: dict,
    stations: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    resolution_km: float,
    n_candidates: int,
) -> list[CandidateLocation]:
    """複合スコアから候補地点を抽出"""
    
    density = density_map["density"]
    gap = coverage_gap_map["gap"]
    coverage = coverage_gap_map["coverage"]
    lat_grid = density_map["lat_grid"]
    lon_grid = density_map["lon_grid"]
    
    # 複合スコア: 出動密度 × カバレッジギャップ
    # 両方が高い地点が優先される
    combined_score = density * (gap + 0.1)  # gapが0でも密度が高ければ候補に
    
    # 既存消防署の近くは除外（最低1km離れた地点のみ）
    station_coords = np.column_stack([stations["経度"], stations["緯度"]])
    station_tree = cKDTree(station_coords)
    min_distance_km = 1.0
    min_distance_deg = min_distance_km / 111.0
    
    # マスク作成
    mask = np.ones_like(combined_score, dtype=bool)
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = [lon_grid[i, j], lat_grid[i, j]]
            dist, _ = station_tree.query(point)
            if dist < min_distance_deg:
                mask[i, j] = False
    
    combined_score = np.where(mask, combined_score, 0)
    
    # ピーク検出（ローカルマキシマ）
    # 単純に上位N個のスコアを取得
    flat_indices = np.argsort(combined_score.ravel())[::-1]
    
    candidates = []
    used_positions = set()
    min_separation_deg = 0.5 / 111.0  # 候補同士は最低0.5km離す
    
    for flat_idx in flat_indices:
        if len(candidates) >= n_candidates:
            break
        
        i, j = np.unravel_index(flat_idx, combined_score.shape)
        lat = lat_grid[i, j]
        lon = lon_grid[i, j]
        
        # 既存候補と近すぎないかチェック
        too_close = False
        for used_lat, used_lon in used_positions:
            dist = np.sqrt((lat - used_lat)**2 + (lon - used_lon)**2)
            if dist < min_separation_deg:
                too_close = True
                break
        
        if too_close:
            continue
        
        score = combined_score[i, j]
        if score <= 0:
            continue
        
        # 候補理由を生成
        density_val = density[i, j]
        gap_val = gap[i, j]
        coverage_val = coverage[i, j]
        
        if density_val > 0.7 and gap_val > 0.5:
            reason = "出動密度高 & カバレッジ不足"
        elif density_val > 0.5:
            reason = "出動密度が高いエリア"
        elif gap_val > 0.7:
            reason = "カバレッジギャップが大きい"
        elif coverage_val < 2:
            reason = "冗長性が不足（救急車1台以下）"
        else:
            reason = "バランス改善"
        
        candidates.append(CandidateLocation(
            lat=float(lat),
            lon=float(lon),
            name=f"候補地点{len(candidates) + 1}",
            reason=reason,
            priority_score=float(score),
            incident_density=float(density_val),
            current_coverage_gap=float(gap_val),
        ))
        
        used_positions.add((lat, lon))
    
    return candidates


# =============================================================================
# 最適化シミュレーション
# =============================================================================

def simulate_new_station(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    candidate: CandidateLocation,
    threshold_min: int = 8,
    new_ambulances: int = NEW_STATION_AMBULANCES,
) -> dict:
    """新規消防署追加のシミュレーション
    
    高速化のため、簡易計算を行う
    """
    # 候補地点の最寄りノード
    try:
        candidate_node = ox.distance.nearest_nodes(graph, candidate.lon, candidate.lat)
    except Exception:
        return {
            "candidate": candidate,
            "error": "グラフ上のノードを特定できません",
        }
    
    # 出動地点の最寄りノード（サンプリングで高速化）
    max_samples = min(500, len(incidents))
    sampled_incidents = incidents.sample(n=max_samples, random_state=42) if len(incidents) > max_samples else incidents
    
    try:
        incident_nodes = ox.distance.nearest_nodes(
            graph,
            sampled_incidents["lon"].tolist(),
            sampled_incidents["lat"].tolist()
        )
    except Exception:
        incident_nodes = [
            ox.distance.nearest_nodes(graph, lon, lat)
            for lon, lat in zip(sampled_incidents["lon"], sampled_incidents["lat"])
        ]
    
    # 候補地点からの到達時間を計算
    threshold_sec = threshold_min * 60
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph,
            candidate_node,
            cutoff=threshold_sec,
            weight="travel_time",
        )
    except Exception:
        return {
            "candidate": candidate,
            "error": "到達時間の計算に失敗",
        }
    
    # カバー可能な出動件数
    covered_count = sum(1 for node in incident_nodes if node in lengths)
    coverage_rate = covered_count / len(sampled_incidents) * 100
    
    # 既存消防署でカバーできていない件数を特定
    # （簡易版: 距離ベース）
    station_coords = np.column_stack([stations["経度"], stations["緯度"]])
    station_tree = cKDTree(station_coords)
    
    target_distance_km = threshold_min * 40 / 60
    target_distance_deg = target_distance_km / 111.0
    
    newly_covered = 0
    for _, inc in sampled_incidents.iterrows():
        inc_point = [inc["lon"], inc["lat"]]
        
        # 既存カバー確認
        dist, _ = station_tree.query(inc_point)
        existing_covered = dist <= target_distance_deg
        
        # 候補地点からの距離
        cand_dist = np.sqrt(
            (inc["lon"] - candidate.lon)**2 + 
            (inc["lat"] - candidate.lat)**2
        )
        new_covered = cand_dist <= target_distance_deg
        
        if new_covered and not existing_covered:
            newly_covered += 1
    
    # スケールアップ（サンプリングの場合）
    scale_factor = len(incidents) / len(sampled_incidents)
    estimated_newly_covered = int(newly_covered * scale_factor)
    
    return {
        "candidate": candidate,
        "coverage_rate": coverage_rate,
        "covered_incidents": covered_count,
        "newly_covered_incidents": estimated_newly_covered,
        "sampled_total": len(sampled_incidents),
        "actual_total": len(incidents),
        "new_ambulances": new_ambulances,
        "efficiency_score": estimated_newly_covered / new_ambulances if new_ambulances > 0 else 0,
    }


def optimize_placement(
    stations: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    candidates: list[CandidateLocation],
    threshold_min: int = 8,
    new_ambulances: int = NEW_STATION_AMBULANCES,
    progress_cb: Callable[[float], None] | None = None,
) -> OptimizationResult:
    """最適配置を決定
    
    貪欲法: 各候補をシミュレーションし、最も効果が高い地点を選択
    """
    start_time = time.time()
    
    if not candidates:
        return OptimizationResult(
            candidate_locations=[],
            best_location=None,
            coverage_improvement={},
            resource_efficiency={},
            computation_time_sec=0,
        )
    
    graph = load_graph()
    
    results = []
    for i, candidate in enumerate(candidates):
        result = simulate_new_station(
            graph, stations, incidents, candidate,
            threshold_min=threshold_min,
            new_ambulances=new_ambulances,
        )
        results.append(result)
        
        if progress_cb:
            progress_cb((i + 1) / len(candidates))
    
    # エラーのない結果のみ
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return OptimizationResult(
            candidate_locations=[c.__dict__ for c in candidates],
            best_location=None,
            coverage_improvement={"error": "シミュレーション失敗"},
            resource_efficiency={},
            computation_time_sec=time.time() - start_time,
        )
    
    # 効率スコアでソート
    valid_results.sort(key=lambda x: x["efficiency_score"], reverse=True)
    best = valid_results[0]
    
    elapsed = time.time() - start_time
    
    return OptimizationResult(
        candidate_locations=[r["candidate"].__dict__ for r in valid_results],
        best_location={
            "lat": best["candidate"].lat,
            "lon": best["candidate"].lon,
            "name": best["candidate"].name,
            "reason": best["candidate"].reason,
            "newly_covered_incidents": best["newly_covered_incidents"],
            "efficiency_score": best["efficiency_score"],
        },
        coverage_improvement={
            "newly_covered_incidents": best["newly_covered_incidents"],
            "coverage_rate": best["coverage_rate"],
            "sampled_total": best["sampled_total"],
        },
        resource_efficiency={
            "new_ambulances": best["new_ambulances"],
            "incidents_per_ambulance": best["efficiency_score"],
        },
        computation_time_sec=elapsed,
    )


# =============================================================================
# 可視化
# =============================================================================

def create_optimization_map(
    stations: gpd.GeoDataFrame,
    incidents: gpd.GeoDataFrame,
    candidates: list[CandidateLocation],
    best_location: dict | None = None,
) -> folium.Map:
    """最適化結果の可視化マップ"""
    
    center_lat = stations["緯度"].mean()
    center_lon = stations["経度"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB Positron",
    )
    
    # 出動地点をヒートマップ風に表示（サンプリング）
    max_display = 1000
    if len(incidents) > max_display:
        display_incidents = incidents.sample(n=max_display, random_state=42)
    else:
        display_incidents = incidents
    
    for _, inc in display_incidents.iterrows():
        folium.CircleMarker(
            location=[inc["lat"], inc["lon"]],
            radius=2,
            color="#888888",
            fill=True,
            fill_opacity=0.3,
            opacity=0.3,
        ).add_to(m)
    
    # 既存消防署
    for _, station in stations.iterrows():
        folium.Marker(
            location=[station["緯度"], station["経度"]],
            popup=f"{station['略称']} ({station['救急車台数']}台)",
            icon=folium.Icon(color="blue", icon="plus", prefix="fa"),
        ).add_to(m)
    
    # 候補地点
    for i, cand in enumerate(candidates):
        is_best = best_location and abs(cand.lat - best_location["lat"]) < 0.0001 and abs(cand.lon - best_location["lon"]) < 0.0001
        
        if is_best:
            # 最適地点は強調
            folium.Marker(
                location=[cand.lat, cand.lon],
                popup=f"⭐ {cand.name}<br>{cand.reason}<br>スコア: {cand.priority_score:.3f}",
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
            ).add_to(m)
            
            # 到達圏の概略（円で表示）
            folium.Circle(
                location=[cand.lat, cand.lon],
                radius=5000,  # 8分 × 40km/h ≈ 5km
                color="red",
                fill=True,
                fill_opacity=0.1,
                popup="8分到達圏（概算）",
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[cand.lat, cand.lon],
                radius=10,
                color="orange",
                fill=True,
                fill_color="orange",
                fill_opacity=0.7,
                popup=f"{cand.name}<br>{cand.reason}<br>スコア: {cand.priority_score:.3f}",
            ).add_to(m)
    
    # 凡例
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>配置最適化</b><br>
        <span style="color: blue;">📍</span> 既存消防署<br>
        <span style="color: red;">⭐</span> 最適候補地点<br>
        <span style="color: orange;">●</span> その他候補地点<br>
        <span style="color: gray;">・</span> 出動地点
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


# =============================================================================
# キャッシュ
# =============================================================================

OPTIMIZATION_CACHE_PATH = CACHE_DIR / "optimization_candidates.json"


def save_candidates_cache(candidates: list[CandidateLocation]) -> None:
    """候補地点をキャッシュ"""
    CACHE_DIR.mkdir(exist_ok=True)
    data = [c.__dict__ for c in candidates]
    with open(OPTIMIZATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_candidates_cache() -> list[CandidateLocation] | None:
    """キャッシュから候補地点を読み込み"""
    if OPTIMIZATION_CACHE_PATH.exists():
        with open(OPTIMIZATION_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [CandidateLocation(**d) for d in data]
    return None


# =============================================================================
# メイン処理
# =============================================================================

def run_optimization(
    n_candidates: int = 10,
    threshold_min: int = 8,
    use_cache: bool = True,
    progress_cb: Callable[[float], None] | None = None,
) -> OptimizationResult:
    """最適化を実行
    
    Args:
        n_candidates: 生成する候補地点数
        threshold_min: 到達時間の閾値（分）
        use_cache: キャッシュを使用するか
        progress_cb: 進捗コールバック
    
    Returns:
        最適化結果
    """
    start_time = time.time()
    
    # データ読み込み
    stations = load_stations()
    incidents = load_incident_locations()
    
    if incidents.empty:
        print("⚠️ 出動データがありません")
        return OptimizationResult(
            candidate_locations=[],
            best_location=None,
            coverage_improvement={},
            resource_efficiency={},
            computation_time_sec=0,
        )
    
    # 候補地点生成
    candidates = None
    if use_cache:
        candidates = load_candidates_cache()
    
    if candidates is None:
        if progress_cb:
            progress_cb(0.1)
        candidates = generate_candidate_locations(
            stations, incidents, n_candidates=n_candidates
        )
        save_candidates_cache(candidates)
    
    if progress_cb:
        progress_cb(0.3)
    
    # 最適化実行
    result = optimize_placement(
        stations, incidents, candidates,
        threshold_min=threshold_min,
        progress_cb=lambda p: progress_cb(0.3 + p * 0.7) if progress_cb else None,
    )
    
    result.computation_time_sec = time.time() - start_time
    
    return result


# =============================================================================
# CLI実行
# =============================================================================

def main():
    print("🚑 リソース考慮 配置最適化")
    print("=" * 50)
    
    result = run_optimization(
        n_candidates=10,
        threshold_min=8,
        use_cache=False,
    )
    
    print(f"\n⏱️ 計算時間: {result.computation_time_sec:.2f}秒")
    
    if result.best_location:
        print(f"\n⭐ 最適候補地点:")
        print(f"   位置: ({result.best_location['lat']:.5f}, {result.best_location['lon']:.5f})")
        print(f"   理由: {result.best_location['reason']}")
        print(f"   新規カバー: {result.best_location['newly_covered_incidents']}件")
        print(f"   効率スコア: {result.best_location['efficiency_score']:.1f}件/台")
    
    print(f"\n📍 候補地点一覧:")
    for i, cand in enumerate(result.candidate_locations[:5], 1):
        print(f"   {i}. ({cand['lat']:.4f}, {cand['lon']:.4f}) - {cand['reason']}")
    
    # マップ保存
    stations = load_stations()
    incidents = load_incident_locations()
    candidates = [CandidateLocation(**c) for c in result.candidate_locations]
    
    m = create_optimization_map(stations, incidents, candidates, result.best_location)
    output_path = Path("optimization_result.html")
    m.save(str(output_path))
    print(f"\n✅ マップ保存: {output_path.resolve()}")


if __name__ == "__main__":
    main()
