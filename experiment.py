import geopandas as gpd
import osmnx as ox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import unary_union
from pathlib import Path
#TODO:地図を出すようにしておく

# ★日本語化ライブラリのインポート
import japanize_matplotlib

ox.settings.use_cache = True
GRAPHML_PATH = Path("cache/ehime_drive.graphml")
GRAPHML_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_or_build_graph(north: float, south: float, east: float, west: float) -> nx.MultiDiGraph:
    """Return cached road network when available, otherwise download and cache it."""
    if GRAPHML_PATH.exists():
        print("キャッシュ済みの道路ネットワークを再利用します。")
        return ox.load_graphml(GRAPHML_PATH)

    print("道路データを準備中...（初回取得は時間がかかります）")
    bbox = (north, south, east, west)
    try:
        graph = ox.graph_from_bbox(bbox=bbox, network_type='drive')
    except ValueError as exc:
        if "no graph nodes" not in str(exc).lower():
            raise
        print("指定範囲に道路ノードが見つからなかったため、愛媛県全域データにフォールバックします。")
        graph = ox.graph_from_place("Ehime, Japan", network_type='drive')
    ox.save_graphml(graph, GRAPHML_PATH)
    print("道路ネットワークを保存しました。次回以降は高速に読み込めます。")
    return graph

# --- 1. データの読み込み ---
file_path = 'map.xlsx'  # ファイル名
df = pd.read_excel(file_path)

# データフレームをGeoDataFrameに変換
geometry = gpd.points_from_xy(df['経度'], df['緯度'])
gdf_stations = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print(f"データ読み込み完了: {len(gdf_stations)}箇所の拠点をロードしました。")

# --- 2. 愛媛県全域の道路ネットワークを取得 ---
print("道路データを準備中...（初回は時間がかかります）")
west, south, east, north = gdf_stations.total_bounds
padding_deg = 0.1  # 約11kmの余白を四方に追加
north += padding_deg
south -= padding_deg
east += padding_deg
west -= padding_deg
graph = load_or_build_graph(north=north, south=south, east=east, west=west)

# 速度設定
hwy_speeds = {'residential': 30, 'secondary': 40, 'tertiary': 40, 'primary': 50, 'motorway': 80}
graph = ox.add_edge_speeds(graph, hwy_speeds=hwy_speeds)
graph = ox.add_edge_travel_times(graph)

print("道路ネットワーク構築完了。到達圏を計算します...")

# --- 3. 到達圏（Isochrone）の計算 ---
trip_times = [5, 10]
iso_colors = {5: '#FF0000', 10: '#0000FF'} 
iso_polygons = []

for _, row in gdf_stations.iterrows():
    center_point = (row['緯度'], row['経度'])
    try:
        center_node = ox.distance.nearest_nodes(graph, center_point[1], center_point[0])
        for trip_time in trip_times:
            subgraph = nx.ego_graph(graph, center_node, radius=trip_time*60, distance='travel_time')
            node_points = [Point((data['x'], data['y'])) for node, data in subgraph.nodes(data=True)]
            if not node_points:
                continue

            reachable_area = unary_union(node_points).convex_hull
            iso_polygons.append({'name': row['略称'], 'time': trip_time, 'geometry': reachable_area})
    except Exception as e:
        print(f"Error at {row['略称']}: {e}")

if not iso_polygons:
    raise RuntimeError("到達圏ポリゴンを生成できませんでした。入力データと旅時間設定を確認してください。")

gdf_iso = gpd.GeoDataFrame(iso_polygons, crs="EPSG:4326")

# --- 4. 地図への描画 ---
print("地図を描画中...")
fig, ax = ox.plot_graph(
    graph,
    show=False,
    close=False,
    edge_color='#4a4a4a',
    edge_linewidth=0.25,
    node_size=0,
    bgcolor='white'
)

# 到達圏の描画
gdf_iso[gdf_iso['time'] == 10].plot(ax=ax, color=iso_colors[10], alpha=0.2, label='10分到達圏')
gdf_iso[gdf_iso['time'] == 5].plot(ax=ax, color=iso_colors[5], alpha=0.4, label='5分到達圏')
gdf_stations.plot(ax=ax, color='black', marker='x', markersize=80, label='消防待機場所', zorder=5)

# --- 5. ズーム設定（ここを変えるだけで範囲が変わります） ---

# ★ここをいじってください！★
target_lat = 33.8416  # 中心の緯度（例：松山市中心部）
target_lon = 132.7653 # 中心の経度
zoom_radius_km = 10   # 半径何キロメートルを表示するか（小さいほど拡大）

# -------------------------------------------------------

# 表示範囲の計算（1度≒111kmと概算）
lat_range = zoom_radius_km / 111
lon_range = zoom_radius_km / (111 * 0.8) # 経度は緯度によって距離が変わるため補正

ax.set_ylim(target_lat - lat_range, target_lat + lat_range)
ax.set_xlim(target_lon - lon_range, target_lon + lon_range)

# タイトルやラベルも日本語で
plt.title(f"愛媛県救急車到達圏マップ (中心: {target_lat}, {target_lon} / 半径: {zoom_radius_km}km)", fontsize=15)
plt.legend(loc='upper right', fontsize=12) # 凡例を表示
plt.axis('on') # 座標軸を表示（位置確認用）
plt.show()