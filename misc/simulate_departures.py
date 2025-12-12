"""24-hour ambulance departure timeline based on R6.xlsx.

Usage:
    python simulate_departures.py --date 2024-01-01 --data R6.xlsx --output outputs/departures_2024-01-01.png

The script builds a day-long schedule from actual dispatch records and visualizes
when each ambulance left station and returned.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Enable Japanese glyphs when the font package is available.
try:  # pragma: no cover - optional dependency
    import japanize_matplotlib  # type: ignore
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="24h ambulance departure timeline")
    parser.add_argument(
        "--data",
        default="R6.xlsx",
        help="Path to the transport log Excel file (default: R6.xlsx)",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD (uses 覚知の年月日)",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults to outputs/departures_<date>.png",
    )
    parser.add_argument(
        "--group-by",
        choices=["station", "vehicle"],
        default="station",
        help="Group timeline by dispatching station or by vehicle (default: station)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving",
    )
    return parser.parse_args()


def _combine_timestamp(row: pd.Series, hour_col: str, minute_col: str) -> pd.Timestamp:
    """Combine date columns with hour/minute columns.

    Falls back to NaT when source values are missing or invalid.
    """
    h = row.get(hour_col)
    m = row.get(minute_col)
    if pd.isna(h) or pd.isna(m):
        return pd.NaT
    try:
        return pd.Timestamp(
            year=int(row["覚知年"]),
            month=int(row["覚知月"]),
            day=int(row["覚知日"]),
            hour=int(h),
            minute=int(m),
        )
    except Exception:
        return pd.NaT


def load_calls(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["覚知"] = pd.to_datetime(df["覚知"], errors="coerce")
    df = df[df["覚知"].notna()].copy()
    df["date"] = df["覚知"].dt.date
    return df


def _derive_station_label(row: pd.Series) -> str:
    """Approximate dispatching station from 出動隊/出動車両 labels.

    This strips digits and common prefixes (高規格救急車/救急車/救急) and maps
    known prefixes to station-like labels.
    """

    raw = str(row.get("出動隊") or row.get("出動車両") or "").strip()
    if not raw:
        return "未設定"

    cleaned = re.sub(r"[0-9０-９]", "", raw)
    for prefix in ("高規格救急車", "救急車", "救急"):
        cleaned = cleaned.replace(prefix, "")
    cleaned = cleaned.replace("局救急", "局")
    cleaned = cleaned.strip()

    mapping = [
        ("東", "東消防署"),
        ("西", "西消防署"),
        ("中", "中央消防署"),
        ("南", "南消防署"),
        ("北", "北消防署"),
        ("局", "消防局"),
        ("ワーク", "ワーク車"),
        ("機動", "機動隊"),
        ("消防", "消防隊"),
    ]
    for key, label in mapping:
        if cleaned.startswith(key):
            return label

    return cleaned or "未設定"


def build_day_events(df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    day_df = df[df["date"] == target_date].copy()
    if day_df.empty:
        raise ValueError(f"No rows for date {target_date}")

    day_start = pd.Timestamp(target_date)
    # Dispatch time using 出動時/出動分, fallback to 覚知
    dispatch_times: list[pd.Timestamp] = []
    for _, row in day_df.iterrows():
        t = _combine_timestamp(row, "出動時", "出動分")
        if pd.isna(t):
            t = row["覚知"]
        dispatch_times.append(t)
    day_df["dispatch_time"] = dispatch_times

    # Service duration in minutes from 覚知－帰署; falls back to 覚知－引揚 when needed.
    service_min = pd.to_numeric(day_df.get("覚知－帰署"), errors="coerce")
    if service_min.isna().all() and "覚知－引揚" in day_df:
        service_min = pd.to_numeric(day_df.get("覚知－引揚"), errors="coerce")
    day_df["service_min"] = service_min

    median_service = day_df["service_min"].median()

    def _compute_end(row: pd.Series) -> pd.Timestamp:
        if not pd.isna(row["service_min"]):
            return row["dispatch_time"] + pd.to_timedelta(row["service_min"], unit="m")
        end = _combine_timestamp(row, "帰署時", "帰署分")
        if pd.isna(end):
            if pd.isna(row["dispatch_time"]) or pd.isna(median_service):
                return pd.NaT
            return row["dispatch_time"] + pd.to_timedelta(median_service, unit="m")
        if end < row["dispatch_time"]:
            end += pd.Timedelta(days=1)
        return end

    day_df["end_time"] = day_df.apply(_compute_end, axis=1)

    events = day_df.dropna(subset=["dispatch_time", "end_time", "出動車両"]).copy()
    events.sort_values("dispatch_time", inplace=True)

    # Station grouping for plotting
    events["group_station"] = events.apply(_derive_station_label, axis=1)

    # Avoid overlaps per vehicle by pushing starts to when the unit is free.
    busy_until: dict[str, pd.Timestamp] = defaultdict(lambda: day_start)
    adjusted_starts: list[pd.Timestamp] = []
    adjusted_ends: list[pd.Timestamp] = []
    durations_min: list[float] = []

    for _, row in events.iterrows():
        start = max(row["dispatch_time"], busy_until[row["出動車両"]])
        duration = (row["end_time"] - row["dispatch_time"]).total_seconds() / 60.0
        if duration < 0:
            duration = max(row.get("service_min", 0), 0)
        end = start + pd.to_timedelta(duration, unit="m")
        busy_until[row["出動車両"]] = end
        adjusted_starts.append(start)
        adjusted_ends.append(end)
        durations_min.append(duration)

    events["adjusted_start"] = adjusted_starts
    events["adjusted_end"] = adjusted_ends
    events["duration_min"] = durations_min
    return events

def plot_departures(events: pd.DataFrame, target_date: date, output_path: Path, group_by: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    group_col = "group_station" if group_by == "station" else "出動車両"
    order = events[group_col].value_counts().index.tolist()
    y_positions = {name: idx for idx, name in enumerate(order)}

    categories = events["搬送区分(事案)"].fillna("未設定").unique().tolist()
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(len(categories), 2)))
    color_map = {cat: colors[idx % len(colors)] for idx, cat in enumerate(categories)}

    fig_height = max(6, 0.38 * len(order) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    for _, row in events.iterrows():
        start_num = mdates.date2num(row["adjusted_start"])
        width_days = (row["adjusted_end"] - row["adjusted_start"]).total_seconds() / 86400.0
        y = y_positions[row[group_col]]
        cat = row["搬送区分(事案)"] if pd.notna(row["搬送区分(事案)"]) else "未設定"
        ax.broken_barh(
            [(start_num, width_days)],
            (y - 0.4, 0.8),
            facecolors=color_map[cat],
            edgecolors="#222222",
            linewidth=0.6,
        )

    day_start = pd.Timestamp(target_date)
    day_end = day_start + pd.Timedelta(days=1)
    ax.set_xlim(day_start, day_end)
    ax.set_ylim(-1, len(order))
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(order)
    ax.set_xlabel("時間")
    title_scope = "出動署" if group_by == "station" else "車両"
    ax.set_title(f"{target_date} の救急出動タイムライン ({title_scope}別, R6.xlsx)")

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 25, 2)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(axis="x", linestyle="--", color="#cccccc", alpha=0.6)

    # Build legend
    handles = []
    for cat, color in color_map.items():
        handles.append(plt.Line2D([0], [0], color=color, lw=8, label=cat))
    ax.legend(handles=handles, title="搬送区分", loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved timeline to {output_path}")


def summarize(events: pd.DataFrame, group_by: str) -> None:
    total = len(events)
    avg_duration = events["duration_min"].mean()
    busiest_hour = (
        events.groupby(events["adjusted_start"].dt.hour)["出動車両"]
        .count()
        .sort_values(ascending=False)
        .head(1)
    )
    busiest_hour_str = "N/A"
    if not busiest_hour.empty:
        busiest_hour_str = f"{busiest_hour.index[0]:02d}時台 ({busiest_hour.iloc[0]}件)"
    group_col = "group_station" if group_by == "station" else "出動車両"
    top_groups = events[group_col].value_counts().head(5)
    print(f"Total runs: {total}")
    print(f"Average service minutes: {avg_duration:.1f}")
    print(f"Busiest hour: {busiest_hour_str}")
    print("Top groups:")
    for name, cnt in top_groups.items():
        print(f"  {name}: {cnt}件")


def main() -> None:
    args = parse_args()
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = load_calls(data_path)
    events = build_day_events(df, target_date)

    output = Path(args.output) if args.output else Path("outputs") / f"departures_{target_date}.png"
    plot_departures(events, target_date, output, group_by=args.group_by)
    summarize(events, group_by=args.group_by)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
