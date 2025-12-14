"""Analyze delay patterns from incident data for traffic-aware isochrones."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_analysis import learn_delay_patterns_from_incidents, save_delay_factors


def main():
    cache_path = Path('cache/r6_delay_analysis.pkl')
    
    if cache_path.exists():
        print("キャッシュから読み込み中...")
        df = pd.read_pickle(cache_path)
    else:
        print("Excel読み込み中（必要な列のみ）...")
        # 必要な列のみ読む
        usecols = ['覚知', '覚知－現場到着', '出動－現場', '曜日', '出動場所', '出動隊']
        df = pd.read_excel('R6.xlsx', usecols=usecols)
        cache_path.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_path)
        print("キャッシュ保存完了")
    
    print(f"\n読み込み完了: {len(df)}件")
    
    # =================================================================
    # 遅延パターン学習
    # =================================================================
    print("\n=== 遅延パターン学習中 ===")
    patterns = learn_delay_patterns_from_incidents(
        df,
        time_col="覚知",
        arrival_time_col="覚知－現場到着",
        distance_col="出動－現場",
        baseline_hour=3,
    )
    
    # 保存
    save_delay_factors(
        hourly=patterns["hourly"],
        dow=patterns["dow"],
        matrix=patterns["matrix"],
    )
    print("✅ 遅延係数を cache/delay_factors.json に保存しました")
    
    # =================================================================
    # 分析結果表示
    # =================================================================
    df['覚知'] = pd.to_datetime(df['覚知'], errors='coerce')
    df = df[df['覚知'].notna()].copy()
    df['hour'] = df['覚知'].dt.hour
    df['dayofweek'] = df['覚知'].dt.dayofweek

    # 距離あたりの所要時間（分/km）を計算 = 遅延指標
    df['min_per_km'] = np.where(
        df['出動－現場'] > 0, 
        df['覚知－現場到着'] / df['出動－現場'], 
        np.nan
    )

    print('=== 時間帯別 平均現着時間(分) ===')
    hourly = df.groupby('hour')['覚知－現場到着'].agg(['mean', 'count'])
    print(hourly.to_string())

    print('\n=== 時間帯別 分/km（遅延指標）===')
    delay_hourly = df.groupby('hour')['min_per_km'].mean()
    print(delay_hourly.to_string())
    
    # 基準（深夜3時）に対する遅延係数
    baseline = delay_hourly.loc[3]
    print('\n=== 時間帯別 遅延係数（深夜3時=1.0基準）===')
    delay_factor = delay_hourly / baseline
    print(delay_factor.to_string())

    print('\n=== 曜日別 平均現着時間(分) ===')
    days = ['月', '火', '水', '木', '金', '土', '日']
    dow = df.groupby('dayofweek')['覚知－現場到着'].mean()
    for i, v in dow.items():
        print(f'{days[i]}: {v:.2f}分')

    print('\n=== 時間帯×曜日 マトリクス（平均現着時間）===')
    matrix = df.pivot_table(
        values='覚知－現場到着', 
        index='hour', 
        columns='dayofweek', 
        aggfunc='mean'
    )
    matrix.columns = days
    print(matrix.round(1).to_string())


if __name__ == '__main__':
    main()
