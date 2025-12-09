# EMS Isochrone Visualizer (救急搬送到達圏可視化ツール)

愛媛県の救急拠点（または任意の地点）を中心とした、自動車による到達圏（Isochrone）を地図上に可視化するPythonアプリケーションです。
OpenStreetMapの道路ネットワークデータを使用し、指定した時間内に到達可能なエリアを算出・表示します。

## 機能

- **インタラクティブWebアプリ (`app.py`)**: Streamlitを使用したWebインターフェースで、到達圏を動的に確認できます。
- **静的マップ生成 (`generate_map.py`)**: コマンドラインから実行し、到達圏を描画したHTMLファイルを生成します。
- **24時間出動タイムライン (`simulate_departures.py`)**: R6.xlsxの実績データを1日分可視化し、救急車ごとの出動〜帰署を時系列で表示します。
- **キャッシュ機能**: 一度取得した道路ネットワークデータは `cache/` ディレクトリに保存され、次回以降の実行が高速化されます。

## 必要要件

- Python 3.9 以上推奨
- いくつかのライブラリ（`requirements.txt` に記載）

## インストール

リポジトリをクローンし、必要なライブラリをインストールしてください。

### 仮想環境（推奨）

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

- `.venv/` は `.gitignore` 済みです。

## データの準備

本ツールを実行するには、拠点データを含むExcelファイル `map.xlsx` が必要です。
プロジェクトのルートディレクトリに配置してください。

**`map.xlsx` の形式:**
以下のカラムを持つ必要があります。
- `緯度`: 拠点の緯度 (例: 33.8416)　
- `経度`: 拠点の経度 (例: 132.7653)
- その他、拠点名などのカラムがあっても構いません。

### SQLiteに変換する

`map.xlsx` の読み込みがボトルネックになる場合は、SQLiteに変換して高速化できます。

```bash
python scripts/import_map_to_db.py --input map.xlsx --output map.sqlite
```

- 変換後は `map.sqlite` を優先的に読み込み、存在しない場合のみ `map.xlsx` を参照します。
- 元データの共有ポリシー上、`map.sqlite` もリポジトリには含めません。
- 以降の全ての工程において、絶対にキャッシュしたほうがいいです。

### 道路ネットワーク取得を事前キャッシュする

初回の道路ネットワーク取得が重い場合は、GraphMLを事前生成してキャッシュできます。

```bash
python scripts/precache_graph.py --source map.sqlite --fallback map.xlsx --output cache/ehime_drive.graphml
```

- 一度作成した `cache/ehime_drive.graphml` を再利用することで、`app.py` / `generate_map.py` の起動が高速化されます。
- GraphMLには速度・到達時間属性を保存するため、再計算を省略できます。

### 到達圏を事前計算して即時表示したい場合

全消防署×時間帯をバッチで計算し、Parquetにキャッシュできます。

```bash
python scripts/precompute_isochrones.py --times 5 10 15 20 --source map.sqlite --fallback map.xlsx --graph cache/ehime_drive.graphml --output cache/isochrones.parquet
```

- `cache/isochrones.parquet` が存在すれば、`app.py` / `generate_map.py` は計算をスキップして即表示します。
- 時間帯を増減する場合は `--times` を変更して再生成してください。

### まとめて実行するスクリプト

よく使う処理をまとめた実行スクリプトを用意しています。

```bash
./scripts/run_all.sh          # 1) Excel→SQLite, 2) グラフ事前取得, 3) Streamlit起動
./scripts/run_all.sh --no-app # 1) と 2) のみ実行
./scripts/run_all.sh --only-app # データとキャッシュがある前提でアプリだけ起動
```

利用規約の都合でmap.xlsxはこのリポジトリにはあげないです。Teamsにて共有しています。

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

### 3. 24時間の出動タイムラインを描画する

実績データ `R6.xlsx` の指定日を1日分プロットし、救急車ごとの出動〜帰署を時間軸上に可視化します。

```bash
# 例: 2024-01-01 の出動を「出動署」単位で可視化して PNG 保存（デフォルト）
python simulate_departures.py --date 2024-01-01 --data R6.xlsx --output outputs/departures_2024-01-01.png

# 保存後に画面表示もしたい場合
python simulate_departures.py --date 2024-01-01 --show
```

- `--data`: データファイルのパス (既定: R6.xlsx)
- `--date`: 対象日 (必須, YYYY-MM-DD)
- `--group-by`: `station` で出動署ごと（既定）, `vehicle` で車両ごとに並べ替え
- `--output`: 保存先 (省略時は `outputs/departures_<date>.png`)

出動重複があっても同じ車両が同時に走らないよう、先行ジョブが終わるまで開始時刻を順送りにしています。

### 4. R6の出動地点を事前ジオコーディングする

地図描画を速くするため、R6.xlsx の出動場所をまとめてジオコーディングしてキャッシュできます。
ちなみにキャッシュは絶対に作ったほうがいいです。

```bash
# すべての出動場所をキャッシュ (Nominatimへの負荷を考慮して1秒スリープ)
python scripts/precompute_incident_geocode.py --input R6.xlsx --output cache/incident_geocode.parquet --sleep 1.0

# まず100件だけ試す場合
python scripts/precompute_incident_geocode.py --input R6.xlsx --limit 100
```

- `--region` で住所の前に付ける地域名を指定（既定: 愛媛県）
- `--sleep` でリクエスト間隔を秒で指定（既定: 1.0。Nominatimのレート制限配慮）
- `--limit` で新規ジオコーディング件数を上限指定（再実行やデバッグ用）

## ファイル構成

- `app.py`: Streamlit Webアプリケーションのメインスクリプト
- `generate_map.py`: Foliumを使用したHTMLマップ生成スクリプト
- `experiment.py`: 実験用スクリプト、上二つの試作版なので、基本的に使用しなくても大丈夫です
- `requirements.txt`: 依存ライブラリ一覧
- `cache/`: 取得した道路ネットワークデータが保存されるディレクトリ（自動生成）

## メモ
- `2025/12/09`: 作成したものを一旦アップロード、高速化したら地図が表示されなくなったので修正予定