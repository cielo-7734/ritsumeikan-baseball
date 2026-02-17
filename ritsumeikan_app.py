import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import urllib.request
import os
import io
import csv
from datetime import date, timedelta

# --- PDF生成 ---
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ==========================================================
# フォント設定（Matplotlib + PDF共通）
# ==========================================================

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
FONT_PATH = "NotoSansJP.ttf"

@st.cache_data
def load_font():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    return FONT_PATH

@st.cache_data
def setup_mpl_font():
    path = load_font()
    fm.fontManager.addfont(path)
    prop = fm.FontProperties(fname=path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return prop

try:
    font_prop = setup_mpl_font()
    has_font = True
except:
    font_prop = None
    has_font = False

st.set_page_config(page_title="Rapsodo Analyzer", layout="wide")


# ==========================================================
# CSV処理
# ==========================================================

def _decode_bytes(b: bytes):
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            return b.decode(enc), enc
        except:
            pass
    return b.decode("utf-8", errors="ignore"), "utf-8(ignore)"


def process_data(uploaded_file):
    raw = uploaded_file.getvalue()
    text, used_enc = _decode_bytes(raw)
    lines = text.splitlines()

    player_name = "Unknown"
    if len(lines) >= 3:
        reader = csv.reader([lines[2]])
        row3 = next(reader, [])
        if len(row3) >= 2:
            player_name = row3[1].strip()

    bio = io.BytesIO(raw)
    df = pd.read_csv(bio, skiprows=4, encoding=used_enc)
    df.columns = [c.strip().replace('"', "") for c in df.columns]

    rename_dict = {
        "Pitch Type": "球種",
        "Velocity": "球速",
        "Total Spin": "回転数",
        "True Spin (release)": "トゥルースピン",
        "Spin Efficiency (release)": "回転効率",
        "VB (trajectory)": "高さ変化",
        "HB (trajectory)": "横変化",
        "Date": "日付",
        "Is Strike": "判定",
    }
    df = df.rename(columns=rename_dict)
    df = df[~df["球種"].isin(["-", "Other"])]

    df["datetime"] = pd.to_datetime(df["日付"], errors="coerce")
    df["日付"] = df["datetime"].dt.date

    if "判定" in df.columns:
        df["ストライク数"] = df["判定"].map({"Y": 1, "N": 0}).fillna(0)
    else:
        df["ストライク数"] = 0

    for col in ["球速", "回転数", "トゥルースピン", "回転効率", "高さ変化", "横変化"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("-", pd.NA), errors="coerce")

    df = df.dropna(subset=["球速", "球種", "datetime"])
    return player_name, df


# ==========================================================
# サマリー
# ==========================================================

def create_summary(df):
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("球種").agg(
        投球数=("球速", "count"),
        球速平均=("球速", "mean"),
        回転数=("回転数", "mean"),
        回転効率=("回転効率", "mean"),
        変化量高さ=("高さ変化", "mean"),
        変化量横=("横変化", "mean"),
        ストライク率=("ストライク数", "mean"),
    ).reset_index()

    summary["ストライク率(%)"] = summary["ストライク率"] * 100
    summary = summary.drop(columns=["ストライク率"])
    return summary


def make_compare_table(sum_all, sum_this, sum_prev):
    def add_period(df, name):
        if df.empty:
            return df
        df = df.copy()
        df["期間"] = name
        return df

    a = add_period(sum_all, "全体")
    b = add_period(sum_this, "直近30日")
    c = add_period(sum_prev, "前月30日")

    out = pd.concat([a, b, c], ignore_index=True)
    if not out.empty:
        out = out.sort_values(["球種", "期間"])
    return out


# ==========================================================
# PDF生成
# ==========================================================

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf


def build_pdf(player_name, fig_avg, fig_mov, compare_df):
    path = load_font()
    pdfmetrics.registerFont(TTFont("NotoSansJP", path))

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buf, pagesize=A4)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=styles["Title"], fontName="NotoSansJP")
    normal_style = ParagraphStyle("n", parent=styles["Normal"], fontName="NotoSansJP", fontSize=9)

    story = []
    story.append(Paragraph(f"Rapsodo Report - {player_name}", title_style))
    story.append(Spacer(1, 8))

    # --- 横並びグラフ ---
    img_w = 250
    img_h = 160

    table_imgs = Table(
        [[
            RLImage(fig_to_png(fig_avg), width=img_w, height=img_h),
            RLImage(fig_to_png(fig_mov), width=img_w, height=img_h),
        ]]
    )
    story.append(table_imgs)
    story.append(Spacer(1, 10))

    # --- 比較表 ---
    if not compare_df.empty:
        data = [compare_df.columns.tolist()] + compare_df.round(1).values.tolist()
        tbl = Table(data, repeatRows=1)

        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "NotoSansJP"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.append(tbl)

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


# ==========================================================
# メイン
# ==========================================================

def main():
    st.title("⚾ ラプソード解析システム")

    file = st.file_uploader("CSVアップロード")

    if not file:
        return

    player_name, df = process_data(file)

    st.header(player_name)

    unique_pitches = sorted(df["球種"].unique())
    colors_map = dict(zip(unique_pitches, sns.color_palette("husl", len(unique_pitches))))

    # --- グラフ1 ---
    daily = df.groupby(["日付", "球種"])["球速"].mean().reset_index()
    fig_avg, ax1 = plt.subplots()
    sns.lineplot(data=daily, x="日付", y="球速", hue="球種", palette=colors_map, ax=ax1)
    st.pyplot(fig_avg)

    # --- グラフ2 ---
    fig_mov, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="横変化", y="高さ変化", hue="球種", palette=colors_map, ax=ax2)
    st.pyplot(fig_mov)

    # --- 期間分割 ---
    today = date.today()
    this_start = today - timedelta(days=29)
    prev_start = this_start - timedelta(days=30)
    prev_end = this_start - timedelta(days=1)

    sum_all = create_summary(df)
    sum_this = create_summary(df[df["日付"] >= this_start])
    sum_prev = create_summary(df[(df["日付"] >= prev_start) & (df["日付"] <= prev_end)])

    compare_df = make_compare_table(sum_all, sum_this, sum_prev)

    st.dataframe(compare_df)

    # --- PDF ---
    pdf_bytes = build_pdf(player_name, fig_avg, fig_mov, compare_df)

    st.download_button(
        "📄 PDFダウンロード",
        pdf_bytes,
        file_name=f"rapsodo_{player_name}.pdf",
        mime="application/pdf"
    )


if __name__ == "__main__":
    main()
