# app.py
# RAMKAR MFS v2.7.2 HUD EDITION
# Full Dashboard + Manual Kill + Kara Kutu + Visual Terminal

import math, os, csv, warnings
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="MFS + RAMKAR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# GLOBAL THEME (HUD)
# =============================
st.markdown("""
<style>
:root{
  --bg0:#05060d;
  --bg1:#0b1020;
  --panelA: rgba(16, 23, 42, 0.88);
  --panelB: rgba(7, 9, 18, 0.88);
  --cyan:#00d4ff;
  --green:#00c853;
  --amber:#ffc107;
  --red:#ff1744;
  --purple:#ad1457;
  --text:#e8eefc;
  --muted:#9aa4c1;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 600px at 20% 0%, #101a35 0%, var(--bg0) 60%),
    radial-gradient(900px 500px at 85% 15%, #0d2a2a 0%, var(--bg0) 55%),
    linear-gradient(180deg, var(--bg1), var(--bg0));
  color: var(--text);
}

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(10,14,26,0.98), rgba(5,6,13,0.98));
  border-right: 1px solid rgba(0,212,255,0.18);
}

.block-container{ padding-top:1rem; }

.hud{
  background: linear-gradient(135deg, var(--panelA), var(--panelB));
  border:1px solid rgba(0,212,255,0.22);
  box-shadow:0 0 30px rgba(0,212,255,0.08);
  border-radius:16px;
  padding:14px;
}

.hud-title{
  font-size:0.75rem;
  letter-spacing:1.1px;
  color:var(--muted);
}

.hud-value{
  font-size:1.6rem;
  font-weight:900;
}

.alert{
  border-radius:14px;
  padding:10px;
  font-weight:800;
  text-align:center;
  margin-bottom:10px;
}

.card{
  background: linear-gradient(135deg, var(--panelA), var(--panelB));
  border:1px solid rgba(0,212,255,0.18);
  border-radius:16px;
  padding:12px;
  margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v2.7.2"
LOG_FILE = "mfs_kara_kutu.csv"

# =============================
# DATA
# =============================
KATILIM_HISSELERI = [
    "ASELS","BIMAS","KONTR","EREGL","TUPRS","SASA","HEKTS","GESAN",
    "CWENE","EUPWR","YEOTK","SMART","MAVI","LOGO","GUBRF","BRISA"
]

# =============================
# YFINANCE
# =============================
try:
    import yfinance as yf
    YF = True
except:
    YF = False

# =============================
# STATE
# =============================
if "manual_kill" not in st.session_state:
    st.session_state.manual_kill = False
if "kill_reason" not in st.session_state:
    st.session_state.kill_reason = ""

# =============================
# KARA KUTU
# =============================
def log_event(event, detail, score, regime):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Tarih","Olay","Detay","Skor","Rejim"])
        w.writerow([datetime.now(), event, detail, score, regime])

# =============================
# SCORING (BASİT)
# =============================
def mfs_score():
    return 78

def regime_from_score(score, manual):
    if manual:
        return "OFF-KILL"
    if score >= 60: return "ON"
    if score >= 40: return "NEUTRAL"
    return "OFF"

# =============================
# RAMKAR TARAYICI
# =============================
def ramkar_scan():
    results=[]
    if not YF:
        return results
    for s in KATILIM_HISSELERI:
        try:
            d=yf.Ticker(f"{s}.IS").history(period="6mo",interval="1wk")
            if len(d)<20: continue
            close=d["Close"].iloc[-1]
            ema=d["Close"].ewm(span=20).mean().iloc[-1]
            ok=close>ema
            results.append({
                "symbol":s,
                "price":round(close,2),
                "status":"RADAR" if ok else "PUSU",
                "color":"#00c853" if ok else "#666"
            })
        except:
            pass
    return results

# =============================
# HEADER
# =============================
score = mfs_score()
regime = regime_from_score(score, st.session_state.manual_kill)

if st.session_state.manual_kill:
    st.markdown(f"""
    <div class="alert" style="background:#ff1744;color:white;">
    🚨 MANUEL KILL AKTİF — {st.session_state.kill_reason}
    </div>
    """, unsafe_allow_html=True)

# =============================
# HUD METRICS
# =============================
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="hud">
      <div class="hud-title">REJİM</div>
      <div class="hud-value">{regime}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="hud">
      <div class="hud-title">MFS SKOR</div>
      <div class="hud-value">{score}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="hud">
      <div class="hud-title">VERSİYON</div>
      <div class="hud-value">{APP_VERSION}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================
# TABS
# =============================
tab1,tab2,tab3=st.tabs(["📊 MFS","🔥 RAMKAR","🚨 KILL"])

with tab1:
    st.success("Sistem çalışıyor. Rejim ve skor aktif.")

with tab2:
    if regime=="OFF-KILL":
        st.error("Tarama kilitli.")
    else:
        if st.button("🔄 TARA"):
            res=ramkar_scan()
            log_event("SCAN",f"{len(res)} hisse",score,regime)
            for r in res:
                st.markdown(f"""
                <div class="card">
                  <b>{r['symbol']}</b> — ₺{r['price']}  
                  <span style="color:{r['color']}">{r['status']}</span>
                </div>
                """,unsafe_allow_html=True)

with tab3:
    if not st.session_state.manual_kill:
        reason=st.text_area("Kill sebebi (min 10 karakter)")
        if st.button("AKTİFLEŞTİR") and len(reason)>=10:
            st.session_state.manual_kill=True
            st.session_state.kill_reason=reason
            log_event("MANUAL_KILL",reason,score,regime)
            st.experimental_rerun()
    else:
        if st.button("KALDIR"):
            log_event("KILL_OFF","",score,regime)
            st.session_state.manual_kill=False
            st.session_state.kill_reason=""
            st.experimental_rerun()

st.markdown("---")
st.caption(f"MFS + RAMKAR {APP_VERSION} • {datetime.now()}")
