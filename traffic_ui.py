"""Streamlit UI components for traffic-aware isochrones.

Usage in app.py:
    from traffic_ui import render_traffic_settings, get_selected_traffic_params
"""

import streamlit as st
from traffic_analysis import (
    TIME_SLOT_LABELS,
    DOW_LABELS,
    get_representative_hour,
    get_delay_factor,
    format_delay_info,
    load_delay_factors,
    DELAY_FACTORS_PATH,
)


def render_traffic_settings() -> dict:
    """Render traffic settings UI and return selected parameters.
    
    Returns:
        dict with keys: enabled, hour, dow, factor
    """
    st.markdown("### 🚦 渋滞考慮モード")
    
    # Check if learned factors exist
    factors_exist = DELAY_FACTORS_PATH.exists()
    if factors_exist:
        st.success("✅ 実データから学習した遅延係数を使用")
    else:
        st.warning("⚠️ デフォルトの推定値を使用（misc/analyze_delay_patterns.py を実行すると学習できます）")
    
    enabled = st.toggle("渋滞を考慮する", value=False, key="traffic_enabled")
    
    if not enabled:
        return {"enabled": False, "hour": None, "dow": None, "factor": 1.0}
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_slot = st.selectbox(
            "時間帯",
            options=list(TIME_SLOT_LABELS.keys()),
            index=3,  # Default to 朝ラッシュ
            key="traffic_time_slot",
        )
        hour = get_representative_hour(time_slot)
    
    with col2:
        use_dow = st.checkbox("曜日も考慮", value=False, key="traffic_use_dow")
        if use_dow:
            dow_label = st.selectbox(
                "曜日",
                options=DOW_LABELS,
                index=0,
                key="traffic_dow",
            )
            dow = DOW_LABELS.index(dow_label)
        else:
            dow = None
    
    # Show delay factor info
    factor = get_delay_factor(hour, dow)
    st.info(f"📊 遅延係数: {format_delay_info(hour, dow)}")
    
    # Explanation
    with st.expander("ℹ️ 遅延係数について"):
        st.markdown("""
        遅延係数は、**深夜3時**（最も空いている時間帯）を基準（1.0）として、
        各時間帯でどの程度到達時間が延びるかを表します。
        
        - **1.0未満**: 深夜より速い（ほぼない）
        - **1.0〜1.1**: 通常
        - **1.1〜1.3**: やや混雑
        - **1.3以上**: 混雑
        
        例えば係数が1.4の場合、5分で到達できるエリアが
        実際には7分かかることを意味します。
        """)
    
    return {
        "enabled": True,
        "hour": hour,
        "dow": dow,
        "factor": factor,
    }


def render_traffic_comparison_ui() -> dict | None:
    """Render UI for comparing multiple time slots.
    
    Returns:
        dict with comparison settings, or None if not comparing
    """
    st.markdown("### 📊 時間帯比較モード")
    
    compare = st.toggle("複数時間帯を比較", value=False, key="traffic_compare")
    
    if not compare:
        return None
    
    selected_slots = st.multiselect(
        "比較する時間帯を選択",
        options=list(TIME_SLOT_LABELS.keys()),
        default=["深夜 (0-5時)", "朝ラッシュ (7-9時)", "夕ラッシュ (17-19時)"],
        key="traffic_compare_slots",
    )
    
    if len(selected_slots) < 2:
        st.warning("2つ以上の時間帯を選択してください")
        return None
    
    hours = [get_representative_hour(slot) for slot in selected_slots]
    
    # Show factor comparison
    st.markdown("**選択した時間帯の遅延係数:**")
    for slot, hour in zip(selected_slots, hours):
        factor = get_delay_factor(hour)
        st.write(f"- {slot}: {format_delay_info(hour)}")
    
    return {
        "slots": selected_slots,
        "hours": hours,
    }


def render_delay_heatmap():
    """Render a heatmap of delay factors by hour and day of week."""
    import pandas as pd
    
    st.markdown("### 🗓️ 遅延係数ヒートマップ")
    
    factors = load_delay_factors()
    
    if factors.get("matrix"):
        # Build matrix from learned data
        data = []
        for hour in range(24):
            row = {"時間": f"{hour:02d}時"}
            for dow in range(7):
                key = f"{hour}_{dow}"
                row[DOW_LABELS[dow]] = factors["matrix"].get(key, 1.0)
            data.append(row)
        df = pd.DataFrame(data).set_index("時間")
    else:
        # Build from hourly * dow
        data = []
        for hour in range(24):
            row = {"時間": f"{hour:02d}時"}
            hourly = factors["hourly"].get(hour, 1.0)
            for dow in range(7):
                dow_f = factors["dow"].get(dow, 1.0)
                row[DOW_LABELS[dow]] = round(hourly * dow_f, 2)
            data.append(row)
        df = pd.DataFrame(data).set_index("時間")
    
    # Style the dataframe as heatmap
    def color_delay(val):
        if val < 1.0:
            return "background-color: #90EE90"  # light green
        elif val < 1.1:
            return "background-color: #FFFFE0"  # light yellow
        elif val < 1.3:
            return "background-color: #FFD700"  # gold
        else:
            return "background-color: #FF6347"  # tomato
    
    styled = df.style.applymap(color_delay).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    # Test
    st.set_page_config(page_title="Traffic UI Test", layout="wide")
    
    params = render_traffic_settings()
    st.write("Settings:", params)
    
    st.divider()
    
    compare = render_traffic_comparison_ui()
    if compare:
        st.write("Comparison:", compare)
    
    st.divider()
    
    render_delay_heatmap()
