# EMS Isochrone Visualizer (救急搬送到達圏可視化ツール)

愛媛県の救急拠点（または任意の地点）を中心とした、自動車による到達圏（Isochrone）を地図上に可視化するPythonアプリケーションです。
OpenStreetMapの道路ネットワークデータを使用し、指定した時間内に到達可能なエリアを算出・表示します。

## 機能

- **インタラクティブWebアプリ (`app.py`)**: Streamlitを使用したWebインターフェースで、到達圏を動的に確認できます。
- **静的マップ生成 (`generate_map.py`)**: コマンドラインから実行し、到達圏を描画したHTMLファイルを生成します。
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

### 道路ネットワーク取得を事前キャッシュする

初回の道路ネットワーク取得が重い場合は、GraphMLを事前生成してキャッシュできます。

```bash
python scripts/precache_graph.py --source map.sqlite --fallback map.xlsx --output cache/ehime_drive.graphml
```

- 一度作成した `cache/ehime_drive.graphml` を再利用することで、`app.py` / `generate_map.py` の起動が高速化されます。
- GraphMLには速度・到達時間属性を保存するため、再計算を省略できます。

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

## ファイル構成

- `app.py`: Streamlit Webアプリケーションのメインスクリプト
- `generate_map.py`: Foliumを使用したHTMLマップ生成スクリプト
- `experiment.py`: 実験用スクリプト、上二つの試作版なので、基本的に使用しなくても大丈夫です
- `requirements.txt`: 依存ライブラリ一覧
- `cache/`: 取得した道路ネットワークデータが保存されるディレクトリ（自動生成）