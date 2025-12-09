# EMS Isochrone Visualizer (救急搬送到達圏可視化ツール)

愛媛県の救急拠点（または任意の地点）を中心とした、自動車による到達圏（Isochrone）を地図上に可視化するPythonアプリケーションです。
OpenStreetMapの道路ネットワークデータを使用し、指定した時間内に到達可能なエリアを算出・表示します。

## 機能

- **インタラクティブWebアプリ (`app.py`)**: Streamlitを使用したWebインターフェースで、到達圏を動的に確認できます。
- **静的マップ生成 (`generate_map.py`)**: コマンドラインから実行し、到達圏を描画したHTMLファイルを生成します。
- **キャッシュ機能**: 一度取得した道路ネットワークデータは `cache/` ディレクトリに保存され、次回以降の実行が高速化されます。

## 必要要件

- Python 3.9 以上推奨
- 以下のライブラリ（`requirements.txt` に記載）
    - streamlit
    - folium
    - geopandas
    - networkx
    - osmnx
    - pandas
    - shapely
    - matplotlib
    - japanize-matplotlib
    - openpyxl

## インストール

リポジトリをクローンし、必要なライブラリをインストールしてください。

```bash
pip install -r requirements.txt
```

## データの準備

本ツールを実行するには、拠点データを含むExcelファイル `map.xlsx` が必要です。
プロジェクトのルートディレクトリに配置してください。

**`map.xlsx` の形式:**
以下のカラムを持つ必要があります。
- `緯度`: 拠点の緯度 (例: 33.8416)
- `経度`: 拠点の経度 (例: 132.7653)
- その他、拠点名などのカラムがあっても構いません。

利用規約の都合でこのリポジトリにはあげないです。Teamsにて共有しています。

## 使い方

### 1. Webアプリケーションの起動

以下のコマンドを実行すると、ブラウザが立ち上がりアプリが表示されます。

```bash
streamlit run app.py
```

### 2. 静的マップの生成

コマンドラインから実行してHTMLファイルを生成する場合：

```bash
python generate_map.py
```
※ 引数や設定が必要な場合はスクリプト内の設定を確認してください。

## ファイル構成

- `app.py`: Streamlit Webアプリケーションのメインスクリプト
- `generate_map.py`: Foliumを使用したHTMLマップ生成スクリプト
- `experiment.py`: 実験用スクリプト
- `requirements.txt`: 依存ライブラリ一覧
- `cache/`: 取得した道路ネットワークデータが保存されるディレクトリ（自動生成）