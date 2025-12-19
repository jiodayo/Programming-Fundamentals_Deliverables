"""消防署別リソース分析スクリプト"""
import pandas as pd
from pathlib import Path

df = pd.read_pickle(Path(__file__).parent.parent / 'cache/r6_delay_analysis.pkl')

# 消防署ごとに集計
def get_station(squad):
    if pd.isna(squad):
        return '不明'
    s = str(squad)
    if '東' in s: return '東消防署'
    if '西' in s: return '西消防署'
    if '南' in s: return '南消防署'
    if '中' in s: return '中央消防署'
    if 'ワーク' in s: return 'ワークステーション'
    if '機動' in s: return '機動隊'
    if '局' in s or '消防' in s: return '消防局'
    return s

df['消防署'] = df['出動隊'].apply(get_station)

print('=== 消防署別 リソース状況 ===\n')
for station in ['東消防署', '西消防署', '南消防署', '中央消防署', 'ワークステーション', '機動隊', '消防局']:
    sdf = df[df['消防署'] == station]
    vehicles = sdf['出動隊'].dropna().unique()
    total_calls = len(sdf)
    print(f'{station}:')
    print(f'  救急車: {len(vehicles)}台')
    print(f'  年間出動: {total_calls:,}件')
    print(f'  1台あたり: {total_calls/max(len(vehicles),1):,.0f}件/年')
    print(f'  車両: {list(vehicles)}')
    print()

print('\n=== 出動隊別 出動件数 ===')
print(df['出動隊'].value_counts().to_string())
