"""Traffic-aware isochrone computation using real incident data patterns.

実データから学習した遅延パターンを使って、時間帯を考慮した到達圏を計算する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable
import copy

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPoint

# デフォルトの遅延係数（実データ分析前の推定値）
# 深夜3時を1.0基準として、各時間帯の遅延を表す
DEFAULT_DELAY_FACTORS = {
    0: 0.95, 1: 0.93, 2: 0.90, 3: 1.00, 4: 0.95, 5: 1.00,
    6: 1.10, 7: 1.25, 8: 1.40, 9: 1.30, 10: 1.15, 11: 1.10,
    12: 1.15, 13: 1.10, 14: 1.10, 15: 1.15, 16: 1.25, 17: 1.35,
    18: 1.45, 19: 1.30, 20: 1.15, 21: 1.05, 22: 1.00, 23: 0.98,
}

# 曜日別の補正係数（月=0, 日=6）
DEFAULT_DOW_FACTORS = {
    0: 1.02,  # 月
    1: 1.00,  # 火
    2: 1.00,  # 水
    3: 1.00,  # 木
    4: 1.05,  # 金
    5: 0.92,  # 土
    6: 0.90,  # 日
}

DELAY_FACTORS_PATH = Path("cache/delay_factors.json")


def load_delay_factors() -> dict:
    """Load learned delay factors from cache, or return defaults."""
    if DELAY_FACTORS_PATH.exists():
        with open(DELAY_FACTORS_PATH) as f:
            data = json.load(f)
            # JSON keys are strings, convert to int
            return {
                "hourly": {int(k): v for k, v in data.get("hourly", {}).items()},
                "dow": {int(k): v for k, v in data.get("dow", {}).items()},
                "matrix": data.get("matrix"),  # hour x dow matrix if available
            }
    return {
        "hourly": DEFAULT_DELAY_FACTORS,
        "dow": DEFAULT_DOW_FACTORS,
        "matrix": None,
    }


def save_delay_factors(hourly: dict, dow: dict, matrix: dict | None = None) -> None:
    """Save learned delay factors to cache."""
    DELAY_FACTORS_PATH.parent.mkdir(exist_ok=True)
    with open(DELAY_FACTORS_PATH, "w") as f:
        json.dump({
            "hourly": hourly,
            "dow": dow,
            "matrix": matrix,
        }, f, indent=2, ensure_ascii=False)


def get_delay_factor(hour: int, dow: int | None = None) -> float:
    """Get the delay factor for a specific hour and day of week.
    
    Args:
        hour: Hour of day (0-23)
        dow: Day of week (0=Monday, 6=Sunday). If None, uses hourly only.
    
    Returns:
        Delay factor (1.0 = baseline, >1.0 = slower)
    """
    factors = load_delay_factors()
    
    # Try matrix first (most accurate)
    if factors.get("matrix") and dow is not None:
        matrix = factors["matrix"]
        key = f"{hour}_{dow}"
        if key in matrix:
            return matrix[key]
    
    # Fall back to hourly * dow
    hourly_factor = factors["hourly"].get(hour, 1.0)
    if dow is not None:
        dow_factor = factors["dow"].get(dow, 1.0)
        return hourly_factor * dow_factor
    
    return hourly_factor


def adjust_graph_for_traffic(
    graph: nx.MultiDiGraph,
    hour: int,
    dow: int | None = None,
) -> nx.MultiDiGraph:
    """Create a copy of the graph with travel times adjusted for traffic.
    
    Args:
        graph: Original road network graph
        hour: Hour of day (0-23)
        dow: Day of week (0=Monday, 6=Sunday)
    
    Returns:
        New graph with adjusted travel_time attributes
    """
    factor = get_delay_factor(hour, dow)
    
    # Create a copy to avoid modifying the original
    adjusted = graph.copy()
    
    for u, v, key, data in adjusted.edges(keys=True, data=True):
        if "travel_time" in data:
            data["travel_time_adjusted"] = data["travel_time"] * factor
    
    return adjusted


def compute_traffic_aware_isochrones(
    graph: nx.MultiDiGraph,
    stations: gpd.GeoDataFrame,
    trip_times: Iterable[int],
    hour: int = 8,
    dow: int | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> gpd.GeoDataFrame:
    """Compute isochrones considering traffic conditions at specific time.
    
    Args:
        graph: Road network graph (should have travel_time attribute)
        stations: GeoDataFrame with station locations (緯度, 経度, 略称)
        trip_times: List of travel time thresholds in minutes
        hour: Hour of day (0-23)
        dow: Day of week (0=Monday, 6=Sunday)
        progress_cb: Optional progress callback
    
    Returns:
        GeoDataFrame with isochrone polygons
    """
    factor = get_delay_factor(hour, dow)
    records: list[dict] = []

    # Vectorized nearest-node lookup
    xs = stations["経度"].to_list()
    ys = stations["緯度"].to_list()
    try:
        center_nodes = ox.distance.nearest_nodes(graph, xs, ys)
    except Exception:
        center_nodes = [ox.distance.nearest_nodes(graph, x, y) for x, y in zip(xs, ys)]

    total = len(center_nodes)
    node_xy = {n: (data["x"], data["y"]) for n, data in graph.nodes(data=True)}
    trip_times_sorted = sorted(trip_times)
    
    # Adjust max radius by delay factor
    max_radius = trip_times_sorted[-1] * 60 * factor if trip_times_sorted else 0

    for idx, (row, center_node) in enumerate(zip(stations.itertuples(index=False), center_nodes)):
        lengths = nx.single_source_dijkstra_path_length(
            graph,
            center_node,
            cutoff=max_radius,
            weight="travel_time",
        )

        for minutes in trip_times_sorted:
            # Adjust cutoff by delay factor
            cutoff = minutes * 60 * factor
            reachable_nodes = [nid for nid, dist in lengths.items() if dist <= cutoff]
            if not reachable_nodes:
                continue
            points = [node_xy[nid] for nid in reachable_nodes if nid in node_xy]
            if not points:
                continue
            hull = MultiPoint(points).convex_hull
            records.append({
                "name": row.略称,
                "time": minutes,
                "hour": hour,
                "dow": dow,
                "delay_factor": factor,
                "geometry": hull,
            })

        if progress_cb:
            progress_cb((idx + 1) / total)

    if not records:
        raise RuntimeError("到達圏ポリゴンを生成できませんでした。")
    
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# =============================================================================
# 時間帯ラベル用ユーティリティ
# =============================================================================

TIME_SLOT_LABELS = {
    "深夜 (0-5時)": (0, 5),
    "早朝 (5-7時)": (5, 7),
    "朝ラッシュ (7-9時)": (7, 9),
    "午前 (9-12時)": (9, 12),
    "昼 (12-14時)": (12, 14),
    "午後 (14-17時)": (14, 17),
    "夕ラッシュ (17-19時)": (17, 19),
    "夜 (19-22時)": (19, 22),
    "深夜 (22-24時)": (22, 24),
}

DOW_LABELS = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]


def get_representative_hour(slot_label: str) -> int:
    """Get representative hour for a time slot label."""
    if slot_label not in TIME_SLOT_LABELS:
        return 12  # default to noon
    start, end = TIME_SLOT_LABELS[slot_label]
    return (start + end) // 2


def format_delay_info(hour: int, dow: int | None = None) -> str:
    """Format delay factor info for display."""
    factor = get_delay_factor(hour, dow)
    if factor < 1.0:
        return f"🟢 {factor:.2f}x（通常より速い）"
    elif factor < 1.1:
        return f"🟡 {factor:.2f}x（通常）"
    elif factor < 1.3:
        return f"🟠 {factor:.2f}x（やや混雑）"
    else:
        return f"🔴 {factor:.2f}x（混雑）"


# =============================================================================
# 遅延パターン学習（実データから）
# =============================================================================

def learn_delay_patterns_from_incidents(
    df: pd.DataFrame,
    time_col: str = "覚知",
    arrival_time_col: str = "覚知－現場到着",
    distance_col: str = "出動－現場",
    baseline_hour: int = 3,
) -> dict:
    """Learn delay patterns from real incident data.
    
    Args:
        df: Incident DataFrame with time and arrival info
        time_col: Column name for incident timestamp
        arrival_time_col: Column name for arrival time (minutes)
        distance_col: Column name for distance (km)
        baseline_hour: Hour to use as baseline (default 3am = least traffic)
    
    Returns:
        Dictionary with hourly, dow, and matrix delay factors
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()
    df["hour"] = df[time_col].dt.hour
    df["dow"] = df[time_col].dt.dayofweek
    
    # Calculate speed proxy: time per distance (min/km)
    # Higher = slower = more delay
    df["min_per_km"] = np.where(
        df[distance_col] > 0,
        df[arrival_time_col] / df[distance_col],
        np.nan
    )
    
    # Filter outliers
    q_low, q_high = df["min_per_km"].quantile([0.05, 0.95])
    df_clean = df[(df["min_per_km"] >= q_low) & (df["min_per_km"] <= q_high)]
    
    # Compute hourly averages
    hourly_avg = df_clean.groupby("hour")["min_per_km"].mean()
    baseline = hourly_avg.get(baseline_hour, hourly_avg.mean())
    hourly_factors = (hourly_avg / baseline).to_dict()
    
    # Compute day-of-week averages
    dow_avg = df_clean.groupby("dow")["min_per_km"].mean()
    dow_baseline = dow_avg.mean()
    dow_factors = (dow_avg / dow_baseline).to_dict()
    
    # Compute hour x dow matrix
    matrix_avg = df_clean.groupby(["hour", "dow"])["min_per_km"].mean()
    matrix_baseline = matrix_avg.mean()
    matrix_factors = {
        f"{h}_{d}": v / matrix_baseline 
        for (h, d), v in matrix_avg.items()
    }
    
    return {
        "hourly": {int(k): round(v, 3) for k, v in hourly_factors.items()},
        "dow": {int(k): round(v, 3) for k, v in dow_factors.items()},
        "matrix": {k: round(v, 3) for k, v in matrix_factors.items()},
    }


if __name__ == "__main__":
    # テスト用
    print("=== 遅延係数テスト ===")
    for hour in [3, 8, 12, 18]:
        print(f"{hour}時: {format_delay_info(hour)}")
    
    print("\n=== 曜日込み ===")
    for dow in range(7):
        print(f"{DOW_LABELS[dow]} 8時: {format_delay_info(8, dow)}")
