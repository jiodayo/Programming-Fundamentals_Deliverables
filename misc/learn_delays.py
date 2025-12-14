"""Lightweight delay pattern learning script (no OSMnx dependency)."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Constants
DELAY_FACTORS_PATH = Path(__file__).parent.parent / "cache" / "delay_factors.json"


def learn_delay_patterns_from_incidents(
    df: pd.DataFrame,
    time_col: str = "覚知",
    arrival_time_col: str = "覚知－現場到着",
    distance_col: str = "出動－現場",
    baseline_hour: int = 3,
) -> dict:
    """Learn delay patterns from incident data.
    
    Args:
        df: DataFrame with incident data
        time_col: Column name for incident time
        arrival_time_col: Column name for arrival time (minutes)
        distance_col: Column name for distance (km)
        baseline_hour: Hour to use as baseline (default 3am = lowest traffic)
    
    Returns:
        dict with 'hourly', 'dow', 'matrix' delay factors
    """
    df = df.copy()
    
    # Parse datetime
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df[df[time_col].notna()].copy()
    
    df['hour'] = df[time_col].dt.hour
    df['dow'] = df[time_col].dt.dayofweek
    
    # Calculate speed proxy: arrival_time / distance (min/km)
    # Higher value = slower = more traffic
    df['min_per_km'] = np.where(
        df[distance_col] > 0,
        df[arrival_time_col] / df[distance_col],
        np.nan
    )
    
    # Remove outliers (e.g., very short distances or very long times)
    df = df[(df['min_per_km'] > 0.5) & (df['min_per_km'] < 30)]
    
    # Hourly patterns
    hourly_speed = df.groupby('hour')['min_per_km'].mean()
    baseline_speed = hourly_speed.get(baseline_hour, hourly_speed.mean())
    hourly_factors = (hourly_speed / baseline_speed).to_dict()
    
    # Day of week patterns
    dow_speed = df.groupby('dow')['min_per_km'].mean()
    baseline_dow = dow_speed.mean()
    dow_factors = (dow_speed / baseline_dow).to_dict()
    
    # Hour x DOW matrix
    matrix_speed = df.groupby(['hour', 'dow'])['min_per_km'].mean().unstack()
    baseline_matrix = matrix_speed.loc[baseline_hour].mean() if baseline_hour in matrix_speed.index else matrix_speed.values.mean()
    matrix_factors = {}
    for hour in range(24):
        for dow in range(7):
            if hour in matrix_speed.index and dow in matrix_speed.columns:
                val = matrix_speed.loc[hour, dow]
                if pd.notna(val):
                    matrix_factors[f"{hour}_{dow}"] = round(val / baseline_matrix, 3)
    
    # Round values
    hourly_factors = {k: round(v, 3) for k, v in hourly_factors.items()}
    dow_factors = {k: round(v, 3) for k, v in dow_factors.items()}
    
    return {
        "hourly": hourly_factors,
        "dow": dow_factors,
        "matrix": matrix_factors,
    }


def save_delay_factors(hourly: dict, dow: dict = None, matrix: dict = None):
    """Save delay factors to JSON file."""
    data = {
        "hourly": {int(k): v for k, v in hourly.items()},
        "dow": {int(k): v for k, v in (dow or {}).items()},
        "matrix": matrix or {},
        "source": "learned_from_R6",
    }
    DELAY_FACTORS_PATH.parent.mkdir(exist_ok=True)
    with open(DELAY_FACTORS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    cache_path = Path(__file__).parent.parent / 'cache' / 'r6_delay_analysis.pkl'
    excel_path = Path(__file__).parent.parent / 'R6.xlsx'
    
    if cache_path.exists():
        print("📦 キャッシュから読み込み中...")
        df = pd.read_pickle(cache_path)
    else:
        print("📊 Excel読み込み中（初回のみ時間がかかります）...")
        usecols = ['覚知', '覚知－現場到着', '出動－現場', '曜日', '出動場所', '出動隊']
        df = pd.read_excel(excel_path, usecols=usecols)
        cache_path.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_path)
        print("✅ キャッシュ保存完了")
    
    print(f"\n読み込み完了: {len(df):,}件")
    
    # Learn patterns
    print("\n🔬 遅延パターン学習中...")
    patterns = learn_delay_patterns_from_incidents(
        df,
        time_col="覚知",
        arrival_time_col="覚知－現場到着",
        distance_col="出動－現場",
        baseline_hour=3,
    )
    
    # Save
    save_delay_factors(
        hourly=patterns["hourly"],
        dow=patterns["dow"],
        matrix=patterns["matrix"],
    )
    print("✅ 遅延係数を cache/delay_factors.json に保存しました")
    
    # Display results
    print("\n" + "="*50)
    print("📈 時間帯別 遅延係数（深夜3時=1.0基準）")
    print("="*50)
    for h in range(24):
        factor = patterns["hourly"].get(h, 1.0)
        bar = "█" * int(factor * 10)
        label = "🔴" if factor > 1.2 else "🟡" if factor > 1.1 else "🟢"
        print(f"{h:02d}時: {factor:.3f} {label} {bar}")
    
    print("\n" + "="*50)
    print("📅 曜日別 遅延係数")
    print("="*50)
    days = ['月', '火', '水', '木', '金', '土', '日']
    for d in range(7):
        factor = patterns["dow"].get(d, 1.0)
        bar = "█" * int(factor * 10)
        print(f"{days[d]}: {factor:.3f} {bar}")
    
    # Summary statistics
    df['覚知'] = pd.to_datetime(df['覚知'], errors='coerce')
    df = df[df['覚知'].notna()].copy()
    df['hour'] = df['覚知'].dt.hour
    
    print("\n" + "="*50)
    print("📊 時間帯別 平均現着時間（分）")
    print("="*50)
    hourly_arrival = df.groupby('hour')['覚知－現場到着'].agg(['mean', 'count'])
    for h in range(24):
        if h in hourly_arrival.index:
            mean = hourly_arrival.loc[h, 'mean']
            count = hourly_arrival.loc[h, 'count']
            print(f"{h:02d}時: {mean:.1f}分 (n={count:,})")


if __name__ == '__main__':
    main()
