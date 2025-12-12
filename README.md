# 🚑 松山市 救急搬送到達圏可視化ツール

松山市の救急拠点を中心とした自動車による到達圏（Isochrone）を地図上に可視化し、出動データのカバー率を分析するPythonアプリケーションです。

## 主な機能

### 📍 到達圏可視化
- 各消防署から5分/10分/15分/20分で到達可能なエリアを地図上に表示
- 仮想消防署を追加して「もしここに消防署があったら」のシミュレーション

### 📊 カバー率分析
- **R6 (2024年)** と **H27 (2015年)** の出動データを比較
- 配置変更前後でのカバー率改善度を数値で確認
- 到達圏外の出動地点を一覧表示

### 🗓️ 出動地点プロット
- 日別の出動地点を地図上にプロット
- 曜日別に色分けして可視化

## 必要要件

- Python 3.9 以上
- 依存ライブラリは `requirements.txt` に記載

## クイックスタート

### 1. 環境構築

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. データの準備

以下のファイルをプロジェクトルートに配置してください：

| ファイル | 内容 | 必須 |
|----------|------|------|
| `map.xlsx` | 消防署の位置情報（緯度・経度・略称） | ✅ |
| `R6.xlsx` | 2024年の出動データ | カバー率分析に必要 |
| `H27.xls` | 2015年の出動データ | 比較分析に必要 |

※ これらのファイルはTeamsで共有されています（著作権の都合上リポジトリには含みません）

### 3. 事前処理の実行

```bash
./scripts/run_all.sh
```

これにより以下が自動実行されます：
1. Excel → SQLite 変換（高速読み込み用）
2. 道路ネットワークのキャッシュ
3. 住所のジオコーディング
4. Streamlitアプリの起動

### 4. アプリのみ起動（事前処理済みの場合）

```bash
./scripts/run_all.sh --only-app
# または
streamlit run app.py
```

## ファイル構成

```
.
├── app.py                    # メインアプリケーション（Streamlit）
├── generate_map.py           # 静的HTMLマップ生成
├── requirements.txt          # 依存ライブラリ
├── scripts/
│   ├── run_all.sh            # 一括実行スクリプト
│   ├── import_map_to_db.py   # 消防署データのDB化
│   ├── import_incidents_to_db.py  # 出動データのDB化
│   ├── precache_graph.py     # 道路ネットワークキャッシュ
│   ├── precompute_incident_geocode.py  # 住所ジオコーディング
│   └── precompute_isochrones.py  # 到達圏の事前計算
├── cache/                    # キャッシュファイル（自動生成）
│   ├── matsuyama_drive.graphml    # 道路ネットワーク
│   ├── incident_geocode.parquet   # ジオコーディング結果
│   └── isochrones.parquet         # 到達圏ポリゴン
├── misc/                     # 実験用・過去のスクリプト
│   ├── experiment.py
│   ├── simulate_departures.py
│   └── outputs/
└── *.sqlite                  # SQLiteデータベース（自動生成）
```

## 事前処理スクリプト

### まとめて実行

```bash
./scripts/run_all.sh              # 全処理 + アプリ起動
./scripts/run_all.sh --no-app     # 全処理のみ
./scripts/run_all.sh --only-app   # アプリのみ
./scripts/run_all.sh --skip-geocode  # ジオコーディングをスキップ
```

### 個別実行

```bash
# 消防署データをSQLiteに変換
python scripts/import_map_to_db.py --input map.xlsx --output map.sqlite

# 出動データをSQLiteに変換
python scripts/import_incidents_to_db.py --output incidents.sqlite

# 道路ネットワークをキャッシュ
python scripts/precache_graph.py --source map.sqlite --output cache/matsuyama_drive.graphml

# 住所をジオコーディング
python scripts/precompute_incident_geocode.py --sleep 1.0

# 到達圏を事前計算
python scripts/precompute_isochrones.py --times 5 10 --output cache/isochrones.parquet
```

## アプリの使い方

### タブ: 到達圏
- 表示する消防署を選択
- 到達時間（5/10/15/20分）を選択
- 「仮想消防署を追加」で新規拠点のシミュレーション

### タブ: 出動地点 (R6)
- 日付を選択して出動地点をプロット
- 曜日別に色分け表示

### タブ: カバー率分析
- **単独モード**: R6またはH27のカバー率を分析
- **比較モード**: 配置変更前後のカバー率を比較
  - 5分/10分到達圏のカバー率
  - 改善度（差分）の表示
  - 到達圏外の出動リスト

## 技術スタック

- **フロントエンド**: Streamlit, Folium
- **地理データ処理**: GeoPandas, Shapely, OSMnx
- **道路ネットワーク**: NetworkX, OpenStreetMap
- **データベース**: SQLite, Parquet

## 更新履歴

- **2025/12/13**: H27データ対応、カバー率比較機能追加、事前処理の整備
- **2025/12/12**: 松山市周辺データに範囲を限定
- **2025/12/09**: 初版作成
