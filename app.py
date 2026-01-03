import os
import io
import re
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# =========================
# アプリ設定
# =========================
APP_TITLE = "立命館大学硬式野球部"
DB_PATH = "pitch_data.db"
UPLOAD_DIR = "uploads"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)
st.title(APP_TITLE)

# =========================
# CSV列名の候補（表記ゆれ対策）
# =========================
COL_CANDIDATES = {
    "date": ["Date", "date", "測定日", "日付"],
    "pitch_type": ["Pitch Type", "PitchType", "球種"],
    "velocity": ["Velocity", "Velo", "球速"],
    "total_spin": ["Total Spin", "Spin", "回転数"],
    "spin_eff": ["Spin Efficiency", "回転効率"],
    "vb": ["VB (trajectory)", "VB", "縦変化量"],
    "hb": ["HB (trajectory)", "HB", "横変化量"],
    "spin_axis": ["Spin Axis", "回転軸"],
}

HBVB_LIMIT = 70

# =========================
# ユーティリティ関数
# =========================
def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def find_col(df, candidates):
    for c in df.columns:
        for cand in candidates:
            if c.strip() == cand:
                return c
    return None

def detect_and_read_csv(uploaded_file):
    import io
    import pandas as pd

    raw = uploaded_file.getvalue()

    # 1) 文字コード対応（Rapsodo系はcp932のことがある）
    text = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    # 2) 区切り文字推定（, or タブ or ;）
    sample_lines = [ln for ln in text.splitlines() if ln.strip()][:30]
    sample = "\n".join(sample_lines)
    candidates = [",", "\t", ";"]
    sep = max(candidates, key=lambda s: sample.count(s))

    # 3) あなたの条件：5行目=ヘッダー、6行目以降=データ
    #    → skiprows=4, header=0
    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        skiprows=4,
        header=0,
        engine="python",
        on_bad_lines="skip"
    )

    # 4) 念のため：空列/空行っぽいのを掃除
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    return df

def standardize_columns(df):
    rename = {}
    for key, cands in COL_CANDIDATES.items():
        col = find_col(df, cands)
        if col:
            rename[col] = key
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["velocity", "total_spin", "spin_eff", "vb", "hb", "spin_axis"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["date"])

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitch_data (
            date TEXT,
            pitch_type TEXT,
            velocity REAL,
            total_spin REAL,
            spin_eff REAL,
            vb REAL,
            hb REAL,
            spin_axis REAL
        )
    """)
    return conn

def save_to_db(conn, df):
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_sql("pitch_data", conn, if_exists="append", index=False)

# =========================
# メイン処理
# =========================
ensure_dirs()
conn = init_db()

uploaded = st.file_uploader(
    "計測器CSVをアップロード",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded:
    for f in uploaded:
        df = detect_and_read_csv(f)
        df = standardize_columns(df)
        save_to_db(conn, df)
        st.success(f"{f.name} を追加した")

# =========================
# DBから読み込み
# =========================
data = pd.read_sql("SELECT * FROM pitch_data", conn)
data["date"] = pd.to_datetime(data["date"])

if data.empty:
    st.info("まだデータがない。CSVをアップロードしてほしい。")
    st.stop()

# =========================
# フィルタ
# =========================
exclude = st.checkbox("Pitch Type が「-」「Other」を除外", value=True)
if exclude:
    data = data[~data["pitch_type"].isin(["-", "Other"])]

# =========================
# トレンドグラフ
# =========================
st.subheader("球速・回転数トレンド")

col1, col2 = st.columns(2)

with col1:
    g = data.groupby(["date", "pitch_type"])["velocity"].mean().reset_index()
    st.altair_chart(
        alt.Chart(g).mark_line(point=True).encode(
            x="date:T",
            y="velocity:Q",
            color="pitch_type:N"
        ),
        use_container_width=True
    )

with col2:
    g = data.groupby(["date", "pitch_type"])["total_spin"].mean().reset_index()
    st.altair_chart(
        alt.Chart(g).mark_line(point=True).encode(
            x="date:T",
            y="total_spin:Q",
            color="pitch_type:N"
        ),
        use_container_width=True
    )

# =========================
# 散布図
# =========================
st.subheader("散布図")

col3, col4 = st.columns(2)

with col3:
    st.altair_chart(
        alt.Chart(data).mark_circle(size=60).encode(
            x="velocity:Q",
            y="total_spin:Q",
            color="pitch_type:N"
        ),
        use_container_width=True
    )

with col4:
    st.altair_chart(
        alt.Chart(data).mark_circle(size=60).encode(
            x=alt.X("hb:Q", scale=alt.Scale(domain=[-HBVB_LIMIT, HBVB_LIMIT])),
            y=alt.Y("vb:Q", scale=alt.Scale(domain=[-HBVB_LIMIT, HBVB_LIMIT])),
            color="pitch_type:N"
        ),
        use_container_width=True
    )

# =========================
# サマリ表
# =========================
st.subheader("球種別サマリ")

summary = data.groupby("pitch_type").agg(
    球速平均=("velocity", "mean"),
    球速MAX=("velocity", "max"),
    回転数平均=("total_spin", "mean"),
    回転数MAX=("total_spin", "max"),
    回転効率平均=("spin_eff", "mean"),
    縦変化量平均=("vb", "mean"),
    横変化量平均=("hb", "mean"),
    回転軸平均=("spin_axis", "mean"),
).reset_index()

st.dataframe(summary, use_container_width=True)
