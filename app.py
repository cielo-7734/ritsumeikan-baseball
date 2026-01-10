from __future__ import annotations

import re
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gdrive_utils import build_drive_service_from_info, upload_png_bytes

# =========================
# 基本設定
# =========================
DATA_DIR = Path("data")
OUT_DIR = Path("out")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

A4_W_INCH = 8.27
A4_H_INCH = 11.69
DPI = 300

# 企画条件
DEFAULT_EXCLUDE_PITCHTYPE = {"-", "Other"}  # 条件追加：Pitch Type除外（必要なら）
DEFAULT_HBVB_LIM = 70                       # 条件：±70

# =========================
# 共通：Pitch Type判定
# =========================
def is_fastball(pt: str) -> bool:
    s = str(pt).lower().strip()
    return any(k in s for k in ["fast", "4-seam", "4 seam", "four", "straight", "ストレ", "直球"])

# =========================
# CSV読み込み（企画書仕様）
#  - 3行目：Player Name
#  - 5行目：ヘッダー
#  - 6行目以降：データ
#  - "－" は欠損扱い
# =========================
def read_player_name_from_csv_bytes(csv_bytes: bytes, encoding_candidates=("utf-8", "cp932")) -> str:
    text = None
    for enc in encoding_candidates:
        try:
            text = csv_bytes.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = csv_bytes.decode("utf-8", errors="ignore")

    lines = text.splitlines()
    if len(lines) < 3:
        return "Unknown"

    line3 = lines[2].strip().strip('"')
    m = re.search(r"Player\s*Name\s*[:,]?\s*(.*)$", line3, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip().strip('"')
        return name if name else "Unknown"

    return line3 if line3 else "Unknown"

def extract_prefix7(filename: str) -> str:
    base = Path(filename).name
    m = re.match(r"(\d{7})", base)
    return m.group(1) if m else "0000000"

def load_rapsodo_csv(csv_bytes: bytes) -> pd.DataFrame:
    # 5行目がヘッダー → skiprows=4
    df = pd.read_csv(BytesIO(csv_bytes), skiprows=4)

    # "－" / "-" を欠損に寄せる（ファイルによって表記ゆれがある可能性）
    df = df.replace({"－": np.nan, "-": np.nan})

    # Dateを日付へ
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    # Pitch Type
    if "Pitch Type" in df.columns:
        df["Pitch Type"] = df["Pitch Type"].astype(str).str.strip()
    else:
        df["Pitch Type"] = "Unknown"

    # 数値列（存在するものだけ）
    for c in ["Velocity", "Total Spin", "Spin Efficiency", "VB", "HB"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================
# 投手ID / 蓄積（投手ごとparquet）
# =========================
def pitcher_id(prefix7: str, player_name: str) -> str:
    safe = re.sub(r"[^\wぁ-んァ-ン一-龥・\-\s]", "", str(player_name)).strip()
    safe = re.sub(r"\s+", "_", safe) if safe else "Unknown"
    return f"{prefix7}_{safe}"

def store_path(pid: str) -> Path:
    return DATA_DIR / f"{pid}.parquet"

def append_to_store(pid: str, new_df: pd.DataFrame) -> pd.DataFrame:
    path = store_path(pid)
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    # 重複除去（ざっくり）
    key_cols = [c for c in ["Date", "Pitch Type", "Velocity", "Total Spin", "HB", "VB"] if c in merged.columns]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols)

    merged.to_parquet(path, index=False)
    return merged

def load_all_pitchers() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for p in DATA_DIR.glob("*.parquet"):
        out[p.stem] = pd.read_parquet(p)
    return out

# =========================
# フィルタ（条件追加込み）
#   - Pitch Type「-」「Other」除外をON/OFF
#   - HB/VBの外れ値除外（±70）
#   - 日付範囲
#   - 球種別最低球数（任意）
# =========================
def build_filter_ui(df: pd.DataFrame) -> dict:
    st.sidebar.header("条件（フィルタ）")

    # Pitch Type 除外
    pitch_types = sorted(df.get("Pitch Type", pd.Series(dtype=str)).dropna().unique().tolist())
    default_excl = [p for p in pitch_types if p in DEFAULT_EXCLUDE_PITCHTYPE]
    exclude_types = st.sidebar.multiselect(
        "除外するPitch Type（条件追加：- / Other）",
        options=pitch_types,
        default=default_excl
    )

    # Date範囲
    date_min: Optional[datetime] = None
    date_max: Optional[datetime] = None
    if "Date" in df.columns:
        dd = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(dd) > 0:
            dmin, dmax = dd.min().date(), dd.max().date()
            d_in = st.sidebar.date_input("日付範囲（累積データ用）", value=(dmin, dmax))
            if isinstance(d_in, (list, tuple)) and len(d_in) == 2:
                date_min, date_max = d_in[0], d_in[1]

    # HB/VB外れ値除外（±70）
    hbvb_lim = st.sidebar.number_input(
        "HB/VB 絶対値上限（条件：±70）",
        min_value=0,
        max_value=200,
        value=DEFAULT_HBVB_LIM,
        step=5
    )

    # 球種別最低球数
    min_pitches = st.sidebar.number_input(
        "球種別の最低球数（これ未満は除外）",
        min_value=1,
        max_value=200,
        value=1,
        step=1
    )

    return {
        "exclude_types": set(exclude_types),
        "date_min": date_min,
        "date_max": date_max,
        "hbvb_lim": float(hbvb_lim),
        "min_pitches": int(min_pitches),
    }

def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    d = df.copy()

    # Pitch Type除外
    if "Pitch Type" in d.columns and f["exclude_types"]:
        d = d[~d["Pitch Type"].isin(f["exclude_types"])].copy()

    # Date範囲
    if "Date" in d.columns and f["date_min"] and f["date_max"]:
        dd = pd.to_datetime(d["Date"], errors="coerce").dt.date
        d = d[(dd >= f["date_min"]) & (dd <= f["date_max"])].copy()

    # HB/VB外れ値除外（±lim）
    if {"HB", "VB"}.issubset(d.columns) and f["hbvb_lim"] is not None:
        lim = float(f["hbvb_lim"])
        d = d[
            (d["HB"].isna() | (d["HB"].abs() <= lim)) &
            (d["VB"].isna() | (d["VB"].abs() <= lim))
        ].copy()

    # 球種別最低球数
    if "Pitch Type" in d.columns and f["min_pitches"] > 1:
        vc = d["Pitch Type"].value_counts(dropna=True)
        keep = set(vc[vc >= f["min_pitches"]].index.tolist())
        d = d[d["Pitch Type"].isin(keep)].copy()

    return d

# =========================
# A4描画ユーティリティ
# =========================
def make_a4_figure() -> plt.Figure:
    return plt.figure(figsize=(A4_W_INCH, A4_H_INCH), dpi=DPI)

def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def page_title(fig, text: str):
    fig.text(0.02, 0.98, text, ha="left", va="top", fontsize=16, weight="bold")

def add_footer(fig, text: str):
    fig.text(0.02, 0.01, text, ha="left", va="bottom", fontsize=9)

def infer_session_date(df: pd.DataFrame) -> str:
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if len(d) > 0:
            return str(d.dt.date.mode().iloc[0])
    return datetime.now().strftime("%Y-%m-%d")

# =========================
# ① トレンド（球種別・日別平均）
# =========================
def trend_by_pitch(df_all: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df_all.dropna(subset=["Date", "Pitch Type", value_col]).copy()
    if d.empty:
        return d
    g = d.groupby(["Date", "Pitch Type"], as_index=False)[value_col].mean()
    return g

def plot_trend_page(pid: str, df_all: pd.DataFrame) -> bytes:
    fig = make_a4_figure()
    page_title(fig, f"トレンド（球種別・日別平均） | {pid}")

    ax1 = fig.add_axes([0.10, 0.55, 0.85, 0.35])
    ax2 = fig.add_axes([0.10, 0.12, 0.85, 0.35])

    t_velo = trend_by_pitch(df_all, "Velocity") if "Velocity" in df_all.columns else pd.DataFrame()
    t_spin = trend_by_pitch(df_all, "Total Spin") if "Total Spin" in df_all.columns else pd.DataFrame()

    if not t_velo.empty:
        for ptype, sub in t_velo.groupby("Pitch Type"):
            ax1.plot(pd.to_datetime(sub["Date"]), sub["Velocity"], marker="o", label=ptype)
        ax1.set_title("Velocity（球速）")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("km/h")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Velocityのトレンドを作成できるデータがありません", ha="center", va="center")
        ax1.axis("off")

    if not t_spin.empty:
        for ptype, sub in t_spin.groupby("Pitch Type"):
            ax2.plot(pd.to_datetime(sub["Date"]), sub["Total Spin"], marker="o", label=ptype)
        ax2.set_title("Total Spin（総回転数）")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("rpm")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Total Spinのトレンドを作成できるデータがありません", ha="center", va="center")
        ax2.axis("off")

    add_footer(fig, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return fig_to_png_bytes(fig)

# =========================
# ② 散布図（単回測定内）
#   - 球速×総回転数
#   - HB×VB（±70）
# =========================
def plot_scatter_page(pid: str, df_session: pd.DataFrame, hbvb_lim: float) -> bytes:
    fig = make_a4_figure()
    page_title(fig, f"散布図（単回測定内） | {pid}")

    d = df_session.copy()

    ax1 = fig.add_axes([0.10, 0.55, 0.85, 0.35])
    ax2 = fig.add_axes([0.10, 0.12, 0.85, 0.35])

    # Velocity x Total Spin
    if {"Velocity", "Total Spin", "Pitch Type"}.issubset(d.columns) and d.dropna(subset=["Velocity", "Total Spin"]).shape[0] > 0:
        for ptype, sub in d.dropna(subset=["Velocity", "Total Spin"]).groupby("Pitch Type"):
            ax1.scatter(sub["Velocity"], sub["Total Spin"], label=ptype, alpha=0.8)
        ax1.set_title("Velocity × Total Spin")
        ax1.set_xlabel("Velocity (km/h)")
        ax1.set_ylabel("Total Spin (rpm)")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Velocity/Total Spinの散布図を作成できるデータがありません", ha="center", va="center")
        ax1.axis("off")

    # HB x VB（±lim）
    if {"HB", "VB", "Pitch Type"}.issubset(d.columns) and d.dropna(subset=["HB", "VB"]).shape[0] > 0:
        dd = d.dropna(subset=["HB", "VB"]).copy()
        for ptype, sub in dd.groupby("Pitch Type"):
            ax2.scatter(sub["HB"], sub["VB"], label=ptype, alpha=0.8)
        ax2.set_title("HB × VB")
        ax2.set_xlabel("HB")
        ax2.set_ylabel("VB")
        ax2.set_xlim(-hbvb_lim, hbvb_lim)
        ax2.set_ylim(-hbvb_lim, hbvb_lim)
        ax2.axhline(0, linewidth=1, alpha=0.3)
        ax2.axvline(0, linewidth=1, alpha=0.3)
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "HB/VBの散布図を作成できるデータがありません", ha="center", va="center")
        ax2.axis("off")

    add_footer(fig, f"Session Date: {infer_session_date(df_session)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return fig_to_png_bytes(fig)

# =========================
# ③ サマリー表（球種別）
#   - 平均：Velocity / Total Spin / Spin Efficiency / VB / HB
#   - 最大：MAX球速 / MAX回転数
# =========================
def summary_table_page(pid: str, df_session: pd.DataFrame) -> bytes:
    fig = make_a4_figure()
    page_title(fig, f"サマリー（球種別） | {pid}")

    d = df_session.copy()
    if "Pitch Type" not in d.columns or d.empty:
        ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
        ax.text(0.5, 0.5, "サマリー表を作成できるデータがありません", ha="center", va="center")
        ax.axis("off")
        return fig_to_png_bytes(fig)

    mean_cols = [c for c in ["Velocity", "Total Spin", "Spin Efficiency", "VB", "HB"] if c in d.columns]
    max_cols = [c for c in ["Velocity", "Total Spin"] if c in d.columns]

    g_mean = d.groupby("Pitch Type")[mean_cols].mean(numeric_only=True) if mean_cols else pd.DataFrame()
    g_max  = d.groupby("Pitch Type")[max_cols].max(numeric_only=True) if max_cols else pd.DataFrame()

    out = pd.DataFrame(index=sorted(d["Pitch Type"].dropna().unique()))
    if "Velocity" in g_mean.columns:         out["平均球速"] = g_mean["Velocity"].round(2)
    if "Total Spin" in g_mean.columns:       out["平均回転数"] = g_mean["Total Spin"].round(0)
    if "Spin Efficiency" in g_mean.columns:  out["平均回転効率"] = g_mean["Spin Efficiency"].round(2)
    if "VB" in g_mean.columns:               out["平均VB"] = g_mean["VB"].round(2)
    if "HB" in g_mean.columns:               out["平均HB"] = g_mean["HB"].round(2)
    if "Velocity" in g_max.columns:          out["MAX球速"] = g_max["Velocity"].round(2)
    if "Total Spin" in g_max.columns:        out["MAX回転数"] = g_max["Total Spin"].round(0)

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
    ax.axis("off")
    tbl = ax.table(
        cellText=out.reset_index().values,
        colLabels=["球種"] + list(out.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    add_footer(fig, f"Session Date: {infer_session_date(df_session)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return fig_to_png_bytes(fig)

# =========================
# ④ 指標ページ（追加案）
#   - Fastball平均球速=100% として、他球種の平均球速が何%か
#   - 回転効率（Spin Efficiency）平均も併記
# =========================
def indicator_table_page(pid: str, df_session: pd.DataFrame) -> bytes:
    fig = make_a4_figure()
    page_title(fig, f"指標（Fastball=100%） | {pid}")

    d = df_session.copy()
    if "Pitch Type" not in d.columns or d.empty:
        ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
        ax.text(0.5, 0.5, "指標表を作成できるデータがありません", ha="center", va="center")
        ax.axis("off")
        return fig_to_png_bytes(fig)

    cols = [c for c in ["Velocity", "Spin Efficiency"] if c in d.columns]
    if not cols:
        ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
        ax.text(0.5, 0.5, "指標を計算できる列がありません", ha="center", va="center")
        ax.axis("off")
        return fig_to_png_bytes(fig)

    g = d.groupby("Pitch Type")[cols].mean(numeric_only=True).reset_index()

    # Fastball平均球速（基準）
    fb_velo = None
    if "Velocity" in g.columns:
        fb = g[g["Pitch Type"].apply(is_fastball)]
        if not fb.empty:
            fb_velo = float(fb["Velocity"].mean())

    out = pd.DataFrame()
    out["球種"] = g["Pitch Type"]

    if "Velocity" in g.columns:
        out["平均球速"] = g["Velocity"].round(2)
        if fb_velo and fb_velo > 0:
            out["球速(%) ※FB=100"] = (g["Velocity"] / fb_velo * 100).round(1)
        else:
            out["球速(%) ※FB=100"] = np.nan

    if "Spin Efficiency" in g.columns:
        out["平均回転効率"] = g["Spin Efficiency"].round(2)

    # 表示順：Fastballを先頭に
    out["__is_fb__"] = out["球種"].apply(is_fastball)
    out = pd.concat([out[out["__is_fb__"]], out[~out["__is_fb__"]]], ignore_index=True).drop(columns="__is_fb__")

    fig.text(0.05, 0.90, "※ Fastballが無い場合、球速(%)は空欄になります。", ha="left", va="top", fontsize=10)

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
    ax.axis("off")
    tbl = ax.table(
        cellText=out.values,
        colLabels=list(out.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    add_footer(fig, f"Session Date: {infer_session_date(df_session)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return fig_to_png_bytes(fig)

# =========================
# ⑤ 投手間比較（ストレートのみ）
#   - 各投手1点（平均）
#   - 名前表示
# =========================
def overall_compare_pages(all_pitchers: Dict[str, pd.DataFrame], hbvb_lim: float) -> List[bytes]:
    rows = []
    for pid, df in all_pitchers.items():
        if "Pitch Type" not in df.columns:
            continue
        d = df.copy()
        d = d[d["Pitch Type"].apply(is_fastball)]
        if d.empty:
            continue

        rec = {"pid": pid}
        for c in ["Velocity", "Total Spin", "HB", "VB"]:
            if c in d.columns:
                rec[c] = pd.to_numeric(d[c], errors="coerce").mean()
        rows.append(rec)

    comp = pd.DataFrame(rows)
    pages: List[bytes] = []

    if comp.empty:
        fig = make_a4_figure()
        page_title(fig, "投手間比較（ストレート）")
        ax = fig.add_axes([0.05, 0.10, 0.90, 0.80])
        ax.text(0.5, 0.5, "比較できるストレートデータがありません", ha="center", va="center")
        ax.axis("off")
        pages.append(fig_to_png_bytes(fig))
        return pages

    # Page 1: Velocity x Total Spin
    fig1 = make_a4_figure()
    page_title(fig1, "投手間比較（ストレート）：Velocity × Total Spin")
    ax1 = fig1.add_axes([0.10, 0.12, 0.85, 0.78])

    dd = comp.dropna(subset=[c for c in ["Velocity", "Total Spin"] if c in comp.columns])
    if {"Velocity", "Total Spin"}.issubset(dd.columns) and not dd.empty:
        ax1.scatter(dd["Velocity"], dd["Total Spin"], alpha=0.9)
        for _, r in dd.iterrows():
            ax1.text(r["Velocity"], r["Total Spin"], r["pid"], fontsize=8, alpha=0.9)
        ax1.set_xlabel("Velocity (km/h)")
        ax1.set_ylabel("Total Spin (rpm)")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Velocity/Total Spinの比較ができるデータがありません", ha="center", va="center")
        ax1.axis("off")

    add_footer(fig1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pages.append(fig_to_png_bytes(fig1))

    # Page 2: HB x VB
    fig2 = make_a4_figure()
    page_title(fig2, "投手間比較（ストレート）：HB × VB")
    ax2 = fig2.add_axes([0.10, 0.12, 0.85, 0.78])

    dd = comp.dropna(subset=[c for c in ["HB", "VB"] if c in comp.columns])
    if {"HB", "VB"}.issubset(dd.columns) and not dd.empty:
        ax2.scatter(dd["HB"], dd["VB"], alpha=0.9)
        for _, r in dd.iterrows():
            ax2.text(r["HB"], r["VB"], r["pid"], fontsize=8, alpha=0.9)
        ax2.set_xlabel("HB")
        ax2.set_ylabel("VB")
        ax2.set_xlim(-hbvb_lim, hbvb_lim)
        ax2.set_ylim(-hbvb_lim, hbvb_lim)
        ax2.axhline(0, linewidth=1, alpha=0.3)
        ax2.axvline(0, linewidth=1, alpha=0.3)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "HB/VBの比較ができるデータがありません", ha="center", va="center")
        ax2.axis("off")

    add_footer(fig2, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pages.append(fig_to_png_bytes(fig2))

    return pages

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Rapsodo CSV → 蓄積/可視化 → A4 PNG → Drive", layout="wide")
st.title("Rapsodo CSV 蓄積・可視化（A4 PNG出力 & Google Drive保存）")

with st.expander("Google Drive設定（PNGのみ保存）", expanded=True):
    st.write("Service Account を Streamlit Secrets に設定すると、PNGだけDriveに保存できる。")
    drive_folder_id = st.text_input("保存先フォルダID（空ならMyDrive直下）", value=st.secrets.get("DRIVE_FOLDER_ID", ""))

    drive_ready = False
    drive_service = None
    if "GCP_SERVICE_ACCOUNT" in st.secrets:
        try:
            drive_service = build_drive_service_from_info(st.secrets["GCP_SERVICE_ACCOUNT"])
            drive_ready = True
            st.success("Drive 接続OK（Service Account）")
        except Exception as e:
            st.error(f"Drive 接続NG: {e}")
    else:
        st.warning("st.secrets['GCP_SERVICE_ACCOUNT'] が未設定。Drive保存なしでも動く。")

st.divider()

uploaded = st.file_uploader("RapsodoのCSVをアップロード", type=["csv"])
if uploaded is None:
    st.stop()

# 投手識別
prefix7 = extract_prefix7(uploaded.name)
csv_bytes = uploaded.getvalue()
player = read_player_name_from_csv_bytes(csv_bytes)
pid = pitcher_id(prefix7, player)

st.subheader("識別結果")
st.write(f"- ファイル名先頭7桁: **{prefix7}**")
st.write(f"- Player Name（CSV 3行目）: **{player}**")
st.write(f"- 投手ID: **{pid}**")

# 読み込み（単回）
df_session_raw = load_rapsodo_csv(csv_bytes)
st.write("アップロードCSVの先頭（確認）")
st.dataframe(df_session_raw.head(30), use_container_width=True)

# 蓄積（累積）
df_all_raw = append_to_store(pid, df_session_raw)
st.success(f"蓄積完了：{pid}（累計行数: {len(df_all_raw)}）")

# 条件（フィルタ）
filters = build_filter_ui(df_all_raw)
hbvb_lim = filters["hbvb_lim"]

df_all = apply_filters(df_all_raw, filters)
df_session = apply_filters(df_session_raw, filters)

st.divider()
st.subheader("A4 PNG 出力（1人あたり複数ページ）")

session_date = infer_session_date(df_session_raw)  # ファイル名用：元のセッション日
# ※ページ内容はフィルタ後データを使用、ファイル名は「投手ID＋日付」を必ず含める

pages: List[Tuple[str, bytes]] = []
pages.append((f"{pid}_{session_date}_01_trend.png", plot_trend_page(pid, df_all)))
pages.append((f"{pid}_{session_date}_02_scatter.png", plot_scatter_page(pid, df_session, hbvb_lim)))
pages.append((f"{pid}_{session_date}_03_summary.png", summary_table_page(pid, df_session)))
pages.append((f"{pid}_{session_date}_04_indicator_FB100.png", indicator_table_page(pid, df_session)))

# 投手間比較（全投手・ストレートのみ）もフィルタ反映
all_pitchers_raw = load_all_pitchers()
all_pitchers = {k: apply_filters(v, filters) for k, v in all_pitchers_raw.items()}
comp_pages = overall_compare_pages(all_pitchers, hbvb_lim)
for i, b in enumerate(comp_pages, start=1):
    pages.append((f"ALL_fastball_{session_date}_compare_{i}.png", b))

# 表示＆DL
cols = st.columns(2)
for idx, (name, b) in enumerate(pages):
    with cols[idx % 2]:
        st.image(b, caption=name, use_container_width=True)
        st.download_button(
            label=f"PNGをダウンロード: {name}",
            data=b,
            file_name=name,
            mime="image/png",
        )

st.divider()
st.subheader("Google DriveへPNG保存（PNGのみ）")

if drive_ready:
    if st.button("この出力PNGをDriveに保存する"):
        ok, ng = 0, 0
        for name, b in pages:
            try:
                upload_png_bytes(
                    drive_service,
                    b,
                    filename=name,
                    folder_id=(drive_folder_id or None),
                )
                ok += 1
            except Exception as e:
                ng += 1
                st.error(f"失敗: {name} / {e}")
        st.success(f"Drive保存 完了：成功 {ok} / 失敗 {ng}")
else:
    st.info("Drive未設定：必要ならPNGをダウンロードして手動アップロードでOK。")
