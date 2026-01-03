# app.py
import streamlit as st
import plotly.graph_objects as go

from mfs_engine import calculate_mfs_score
from ramkar_mock import get_ramkar_top10

# PAGE
st.set_page_config(page_title="MFS + RAMKAR", layout="wide")

# CSS YÜKLE
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# HEADER
st.markdown("## 🎯 MFS + RAMKAR OPERATIONS DESK")

# ENGINE
total, regime, scores = calculate_mfs_score()

# KPI BAR
c1, c2, c3, c4 = st.columns(4)

def kpi(col, title, value, cls=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value {cls}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(c1, "REJİM", regime, "regime-on" if regime=="ON" else "regime-off")
kpi(c2, "MFS SKOR", total)
kpi(c3, "MAX POZ", "12")
kpi(c4, "RİSK", "2.5R")

st.markdown("---")

# GAUGE
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=total,
    gauge={
        "axis": {"range": [0,100]},
        "bar": {"color": "#00e676" if regime=="ON" else "#ff5252"},
        "steps": [
            {"range":[0,40],"color":"#3b1f1f"},
            {"range":[40,60],"color":"#3b331f"},
            {"range":[60,100],"color":"#1f3b2a"},
        ]
    }
))
fig.update_layout(height=250, margin=dict(t=0,b=0))

c1, c2 = st.columns([2,3])
c1.plotly_chart(fig, use_container_width=True)

# RAMKAR PANEL
with c2:
    st.markdown("### 🔥 RAMKAR RADAR")
    for r in get_ramkar_top10():
        st.markdown(f"""
        <div class="glass">
            <b>{r['symbol']}</b> • Skor: {r['score']}/6 • {r['status']}
        </div>
        """, unsafe_allow_html=True)
