import io
import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# アプリ設定
# =========================
APP_TITLE = "立命館大学硬式野球部"
DB_PATH = "pitch_data.db"
HBVB_LIMIT = 70

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# =========================
# 列名候補（表記ゆれ対策）
# ※あなたのCSV（Rapsodo系）に合わせて増やしている
# =========================
COL_CANDIDATES = {
    "date": ["Date", "date", "測定日", "日付"],
    "pitch_type": ["Pitch Type", "PitchType", "球種", "Pitch"],
    "velocity": ["Velocity", "Velo", "球速"],
    "total_spin": ["Total Spin", "Spin", "回転数"],
    "spin_eff": ["Spin Efficiency", "Spin Efficiency (release)", "回転効率"],
    "vb": ["VB (trajectory)", "VB", "縦変化量"],
    "hb": ["HB (trajectory)", "HB", "横変化量"],
    "spin_axis": ["Spin Axis", "Spin Direction", "回転軸"],
}

NUMERIC_INTERNAL_COLS = ["velocity", "total_spin", "spin_eff", "vb", "hb", "spin_axis"]


# =========================
# ユーティリティ
# =========================
def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """df.columns の中から candidates と完全一致する列名を探す。"""
    for c in df.columns:
        for cand in candidates:
            if str(c).strip() == cand:
                return c
    return None


def decode_bytes(raw: bytes) -> str:
    """文字コードを吸収してテキスト化。"""
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def guess_sep(text: str) -> str:
    """区切り文字をざっくり推定。"""
    sample_lines = [ln for ln in text.splitlines() if ln.strip()][:30]
    sample = "\n".join(sample_lines)
    candidates = [",", "\t", ";"]
    return max(candidates, key=lambda s: sample.count(s))

def detect_and_read_csv(uploaded_file):
    raw = uploaded_file.getvalue()

    # 文字コード対応
    text = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        text = raw.decode("utf-8", errors="replace")

    # 区切り文字推定（, / タブ / ;）
    sample_lines = [ln for ln in text.splitlines() if ln.strip()][:30]
    sample = "\n".join(sample_lines)
    candidates = [",", "\t", ";"]
    sep = max(candidates, key=lambda s: sample.count(s))

    # ✅ あなたの条件：5行目ヘッダー、6行目以降データ
    df = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        skiprows=4,
        header=0,
        engine="python",
        on_bad_lines="skip"
    )

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名を内部名に統一し、型変換も行う。"""
    rename = {}
    for key, cands in COL_CANDIDATES.items():
        col = find_col(df, cands)
        if col is not None:
            rename[col] = key

    df = df.rename(columns=rename)

    # 必須列チェック（ここで分かりやすく止める）
    required = ["date", "pitch_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"必須列が見つからない: {missing}\n"
            f"現在の列名: {list(df.columns)}\n"
            f"→ CSVの5行目（ヘッダー行）が本当に列名になっているか確認してほしい。"
        )

    # 日付：datetime化（失敗は NaT）
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 数値列：「-」などを NaN にして数値化
    for c in NUMERIC_INTERNAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 日付が無い行は落とす
    df = df.dropna(subset=["date"])

    # pitch_type 文字列化＆前後空白除去
    df["pitch_type"] = df["pitch_type"].astype(str).str.strip()

    return df


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
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
        """
    )
    conn.commit()
    return conn


def save_to_db(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    # DB保存用に必要列だけにして、無い列は作る（NULL保存）
    for c in ["velocity", "total_spin", "spin_eff", "vb", "hb", "spin_axis"]:
        if c not in df.columns:
            df[c] = np.nan

    out = df[["date", "pitch_type", "velocity", "total_spin", "spin_eff", "vb", "hb", "spin_axis"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    out.to_sql("pitch_data", conn, if_exists="append", index=False)


def load_db(conn: sqlite3.Connection) -> pd.DataFrame:
    data = pd.read_sql("SELECT * FROM pitch_data", conn)
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["date"])


# =========================
# UI：アップロード
# =========================
conn = init_db()

st.subheader("CSVアップロード")
uploaded_files = st.file_uploader(
    "計測器CSVをアップロード（複数可）",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        try:
            df_raw = detect_and_read_csv(f)
            df_std = standardize_columns(df_raw)
            save_to_db(conn, df_std)
            st.success(f"追加完了: {f.name}（{len(df_std)}行）")
        except Exception as e:
            st.error(f"読み込み失敗: {f.name}")
            st.write(str(e))
            st.write("参考：読み込めた列名", df_raw.columns.tolist() if "df_raw" in locals() else None)
            st.stop()

# =========================
# DB読み込み
# =========================
data = load_db(conn)

if data.empty:
    st.info("まだデータがない。CSVをアップロードしてほしい。")
    st.stop()

# =========================
# フィルタ（Pitch Type）
# =========================
st.subheader("表示設定")
exclude_pitchtype = st.checkbox("Pitch Type が「-」「Other」を除外（推奨）", value=True)
if exclude_pitchtype:
    data = data[~data["pitch_type"].isin(["-", "Other"])]

# タイトル用：データに含まれる月
latest_month = data["date"].dt.month.iloc[-1]
month_title = f"{latest_month}月"

# =========================
# トレンドグラフ（球速 / 回転数：日ごとの平均）
# =========================
st.subheader(f"{month_title}のトレンド（球種別）")

c1, c2 = st.columns(2)

with c1:
    st.caption("球速（Velocity）日別平均")
    g = data.groupby(["date", "pitch_type"], as_index=False)["velocity"].mean()
    chart = (
        alt.Chart(g)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="日付"),
            y=alt.Y("velocity:Q", title="球速"),
            color=alt.Color("pitch_type:N", title="Pitch Type"),
            tooltip=["date:T", "pitch_type:N", "velocity:Q"],
        )
        .properties(title=f"{month_title}の球速（Velocity）")
    )
    st.altair_chart(chart, use_container_width=True)

with c2:
    st.caption("総回転数（Total Spin）日別平均")
    g = data.groupby(["date", "pitch_type"], as_index=False)["total_spin"].mean()
    chart = (
        alt.Chart(g)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="日付"),
            y=alt.Y("total_spin:Q", title="総回転数"),
            color=alt.Color("pitch_type:N", title="Pitch Type"),
            tooltip=["date:T", "pitch_type:N", "total_spin:Q"],
        )
        .properties(title=f"{month_title}の総回転数（Total Spin）")
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# 散布図（1図に全部、球種で色分け）
# =========================
st.subheader(f"{month_title}の散布図（球種別）")
c3, c4 = st.columns(2)

with c3:
    st.caption("球速（Velocity）× 総回転数（Total Spin）")
    scat1 = (
        alt.Chart(data)
        .mark_circle(size=65)
        .encode(
            x=alt.X("velocity:Q", title="球速（Velocity）"),
            y=alt.Y("total_spin:Q", title="総回転数（Total Spin）"),
            color=alt.Color("pitch_type:N", title="Pitch Type"),
            tooltip=["date:T", "pitch_type:N", "velocity:Q", "total_spin:Q"],
        )
        .properties(title=f"{month_title}の Velocity × Total Spin")
    )
    st.altair_chart(scat1, use_container_width=True)

with c4:
    st.caption("横変化量（HB）× 縦変化量（VB）※±70固定")
    scat2 = (
        alt.Chart(data)
        .mark_circle(size=65)
        .encode(
            x=alt.X("hb:Q", title="HB (trajectory)", scale=alt.Scale(domain=[-HBVB_LIMIT, HBVB_LIMIT])),
            y=alt.Y("vb:Q", title="VB (trajectory)", scale=alt.Scale(domain=[-HBVB_LIMIT, HBVB_LIMIT])),
            color=alt.Color("pitch_type:N", title="Pitch Type"),
            tooltip=["date:T", "pitch_type:N", "hb:Q", "vb:Q"],
        )
        .properties(title=f"{month_title}の HB × VB（±{HBVB_LIMIT}）")
    )
    st.altair_chart(scat2, use_container_width=True)

# =========================
# 球種別サマリ表（平均＆MAX）
# =========================
st.subheader("球種別サマリ（平均・MAX）")

summary = (
    data.groupby("pitch_type", as_index=False)
    .agg(
        球速平均=("velocity", "mean"),
        回転数平均=("total_spin", "mean"),
        回転効率平均=("spin_eff", "mean"),
        縦変化量平均=("vb", "mean"),
        横変化量平均=("hb", "mean"),
        回転軸平均=("spin_axis", "mean"),
        MAX球速=("velocity", "max"),
        MAX回転数=("total_spin", "max"),
    )
)

st.dataframe(summary, use_container_width=True)
