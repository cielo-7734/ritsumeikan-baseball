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

# --- PDF生成（ReportLab） ---
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# --- 完璧な日本語フォント設定（Matplotlib + PDF共通） ---
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
FONT_PATH = "NotoSansJP.ttf"

@st.cache_data
def load_font_file():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    return FONT_PATH

@st.cache_data
def setup_mpl_font():
    path = load_font_file()
    fm.fontManager.addfont(path)
    prop = fm.FontProperties(fname=path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return prop

try:
    font_prop = setup_mpl_font()
    has_font = True
except Exception:
    font_prop = None
    has_font = False

st.set_page_config(page_title="Rapsodo Analyzer", layout="wide")


def _decode_bytes(b: bytes):
    """utf-8 がダメなら cp932 / shift_jis で読む（Rapsodo CSV対策）"""
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            return b.decode(enc), enc
        except Exception:
            pass
    return b.decode("utf-8", errors="ignore"), "utf-8(ignore)"


def process_data(uploaded_file):
    file_id = uploaded_file.name[:7]
    try:
        raw = uploaded_file.getvalue()
        text, used_enc = _decode_bytes(raw)
        lines = text.splitlines()

        # 3行目の2列目を選手名として取得（ファイル形式に依存）
        player_name = "Unknown"
        if len(lines) >= 3:
            reader = csv.reader([lines[2]])
            row3 = next(reader, [])
            if len(row3) >= 2:
                player_name = row3[1].strip() or "Unknown"

        # pandas 用に BytesIO で読み直す
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

        # 必須列チェック
        required = ["球種", "球速", "日付"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"必要な列が見つかりません: {col}")

        # 「-」と「Other」を除外
        df = df[~df["球種"].isin(["-", "Other"])]

        # 日付処理
        df["datetime"] = pd.to_datetime(df["日付"], errors="coerce")
        df["日付"] = df["datetime"].dt.date

        # ストライク列
        if "判定" in df.columns:
            df["ストライク数"] = df["判定"].map({"Y": 1, "N": 0}).fillna(0)
        else:
            df["ストライク数"] = 0

        # 数値変換
        target_cols = ["球速", "回転数", "トゥルースピン", "回転効率", "高さ変化", "横変化"]
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace("-", pd.NA), errors="coerce")

        # 解析に必要な欠損を落とす
        df = df.dropna(subset=["球速", "球種", "datetime"])

        return player_name, file_id, df

    except Exception as e:
        st.error(f"解析エラー: {e}")
        return "Error", file_id, pd.DataFrame()


def create_summary(df):
    """球種別に平均などを集計してDataFrameで返す"""
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("球種").agg(
        球速平均=("球速", "mean"),
        球速最大=("球速", "max"),
        回転数=("回転数", "mean"),
        トゥルースピン=("トゥルースピン", "mean"),
        回転効率=("回転効率", "mean"),
        変化量高さ=("高さ変化", "mean"),
        変化量横=("横変化", "mean"),
        ストライク率=("ストライク数", "mean"),
        投球数=("球速", "count"),
    ).reset_index()

    summary["ストライク率(%)"] = summary["ストライク率"] * 100

    # 対FB比（Fastball がある場合）
    if (summary["球種"] == "Fastball").any():
        fb_v = summary.loc[summary["球種"] == "Fastball", "球速平均"].iloc[0]
        summary["球速比率(対FB %)"] = (summary["球速平均"] / fb_v) * 100

    show_cols = [c for c in summary.columns if c not in ["ストライク率"]]
    return summary[show_cols]


def fig_to_png_bytesio(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def df_for_pdf(df_summary: pd.DataFrame):
    """PDF用に丸め＆欠損処理したコピーを作る"""
    if df_summary.empty:
        return df_summary

    out = df_summary.copy()

    for col in ["球速平均", "球速最大", "回転効率", "変化量高さ", "変化量横", "ストライク率(%)", "球速比率(対FB %)"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)

    for col in ["回転数", "トゥルースピン"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0)

    if "投球数" in out.columns:
        out["投球数"] = pd.to_numeric(out["投球数"], errors="coerce").fillna(0).astype(int)

    out = out.fillna("")
    return out


def _add_table_to_story(story, base_font, title, df_summary):
    styles = getSampleStyleSheet()
    h_style = ParagraphStyle(
        "h_jp",
        parent=styles["Heading3"],
        fontName=base_font,
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
    )
    p_style = ParagraphStyle(
        "p_jp",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        leading=12,
    )

    story.append(Paragraph(title, h_style))

    if df_summary.empty:
        story.append(Paragraph("データなし（期間内の投球がありません）", p_style))
        story.append(Spacer(1, 6))
        return

    d = df_for_pdf(df_summary)
    header = list(d.columns)
    body = d.values.tolist()
    table_data = [header] + body

    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), base_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))


def build_one_pdf_for_all(players_payload):
    """
    players_payload: list of dict
      {
        "player_name": str,
        "file_id": str,
        "date_ranges": {"all": str, "this": str, "prev": str},
        "fig_avg": fig,
        "fig_mov": fig,
        "sum_all": df,
        "sum_this": df,
        "sum_prev": df
      }
    """
    # 日本語フォントをPDFに登録
    try:
        path = load_font_file()
        pdfmetrics.registerFont(TTFont("NotoSansJP", path))
        base_font = "NotoSansJP"
    except Exception:
        base_font = "Helvetica"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_jp",
        parent=styles["Title"],
        fontName=base_font,
        fontSize=16,
        leading=20,
    )
    h_style = ParagraphStyle(
        "h_jp",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    p_style = ParagraphStyle(
        "p_jp",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        leading=12,
    )

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=A4,
        leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24
    )

    story = []
    story.append(Paragraph("Rapsodo Analyzer レポート（まとめ）", title_style))
    story.append(Paragraph(f"作成日: {date.today()}", p_style))
    story.append(Spacer(1, 10))

    for i, item in enumerate(players_payload):
        if i > 0:
            story.append(PageBreak())

        player_name = item["player_name"]
        file_id = item["file_id"]
        dr = item["date_ranges"]

        story.append(Paragraph(f"選手: {player_name} / FileID: {file_id}", h_style))
        story.append(Paragraph(f"全体: {dr['all']}", p_style))
        story.append(Paragraph(f"直近30日: {dr['this']}", p_style))
        story.append(Paragraph(f"前月30日: {dr['prev']}", p_style))
        story.append(Spacer(1, 8))

        # グラフ（同じページから順に流し込み：ページは自動で増える）
        story.append(Paragraph("グラフ", h_style))
        avg_png = fig_to_png_bytesio(item["fig_avg"])
        mov_png = fig_to_png_bytesio(item["fig_mov"])

        story.append(RLImage(avg_png, width=520, height=260))
        story.append(Spacer(1, 8))
        story.append(RLImage(mov_png, width=520, height=360))
        story.append(Spacer(1, 10))

        # 表（この後に続けて入れる＝「グラフごと/表ごと」で分けない）
        story.append(Paragraph("球種別サマリー", h_style))
        _add_table_to_story(story, base_font, "全体", item["sum_all"])
        _add_table_to_story(story, base_font, "直近30日", item["sum_this"])
        _add_table_to_story(story, base_font, "前月30日", item["sum_prev"])

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


def main():
    st.title("⚾ ラプソード解析システム")
    files = st.file_uploader("CSVアップロード", accept_multiple_files=True)

    if not files:
        st.info("CSVをアップロードすると解析結果を表示します。")
        return

    players_payload = []

    for file in files:
        p_name, f_id, df = process_data(file)
        if df.empty:
            continue

        st.header(f"📊 {p_name} のラプソード資料")

        unique_pitches = sorted(df["球種"].unique())
        pitch_colors = dict(zip(unique_pitches, sns.color_palette("husl", len(unique_pitches))))

        daily_stats = df.groupby(["日付", "球種"])["球速"].agg(["mean", "max"]).reset_index()

        col_g1, col_g2 = st.columns(2)

        # --- fig: 平均球速推移 ---
        fig_avg, ax_avg = plt.subplots()
        sns.lineplot(
            data=daily_stats, x="日付", y="mean",
            hue="球種", marker="o", ax=ax_avg, palette=pitch_colors
        )
        title_txt, x_txt, y_txt = ("球速（平均値）", "日付", "球速") if has_font else ("Velocity (Avg)", "Date", "Velocity")
        ax_avg.set_title(title_txt)
        ax_avg.set_xlabel(x_txt)
        ax_avg.set_ylabel(y_txt)
        plt.xticks(rotation=45)
        ax_avg.legend(prop=font_prop) if has_font else ax_avg.legend()

        # --- fig: 変化量プロット ---
        fig_mov, ax_mov = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=df, x="横変化", y="高さ変化",
            hue="球種", s=100, ax=ax_mov, palette=pitch_colors
        )
        title_txt2, x_txt2, y_txt2 = ("変化量プロット", "横変化", "高さ変化") if has_font else ("Movement", "HB", "VB")
        ax_mov.set_title(title_txt2)
        ax_mov.set_xlabel(x_txt2)
        ax_mov.set_ylabel(y_txt2)
        ax_mov.axhline(0, linewidth=1)
        ax_mov.axvline(0, linewidth=1)
        ax_mov.legend(prop=font_prop) if has_font else ax_mov.legend()

        with col_g1:
            st.pyplot(fig_avg)
        with col_g2:
            st.pyplot(fig_mov)

        # --- サマリー（全体 / 直近30日 / 前月30日）---
        st.subheader("📌 球種別サマリー（全体 / 直近30日 / 前月30日）")

        today = date.today()
        this_start = today - timedelta(days=29)
        this_end = today
        prev_start = this_start - timedelta(days=30)
        prev_end = this_start - timedelta(days=1)

        df_all = df.copy()
        df_this = df[(df["日付"] >= this_start) & (df["日付"] <= this_end)].copy()
        df_prev = df[(df["日付"] >= prev_start) & (df["日付"] <= prev_end)].copy()

        sum_all = create_summary(df_all)
        sum_this = create_summary(df_this)
        sum_prev = create_summary(df_prev)

        fmt = {
            "球速平均": "{:.1f}",
            "球速最大": "{:.1f}",
            "回転数": "{:.0f}",
            "トゥルースピン": "{:.0f}",
            "回転効率": "{:.1f}",
            "変化量高さ": "{:.1f}",
            "変化量横": "{:.1f}",
            "ストライク率(%)": "{:.1f}",
            "球速比率(対FB %)": "{:.1f}",
        }

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 全体")
            st.dataframe(sum_all.style.format(fmt), use_container_width=True) if not sum_all.empty else st.info("データなし")
        with c2:
            st.markdown(f"### 直近30日（{this_start}〜{this_end}）")
            st.dataframe(sum_this.style.format(fmt), use_container_width=True) if not sum_this.empty else st.info("データなし（期間内の投球がありません）")
        with c3:
            st.markdown(f"### 前月30日（{prev_start}〜{prev_end}）")
            st.dataframe(sum_prev.style.format(fmt), use_container_width=True) if not sum_prev.empty else st.info("データなし（期間内の投球がありません）")

        # 全体期間文字列
        if df_all["日付"].notna().any():
            all_start = df_all["日付"].min()
            all_end = df_all["日付"].max()
            all_range_str = f"{all_start}〜{all_end}"
        else:
            all_range_str = "データなし"

        date_ranges = {
            "all": all_range_str,
            "this": f"{this_start}〜{this_end}",
            "prev": f"{prev_start}〜{prev_end}",
        }

        players_payload.append({
            "player_name": p_name,
            "file_id": f_id,
            "date_ranges": date_ranges,
            "fig_avg": fig_avg,
            "fig_mov": fig_mov,
            "sum_all": sum_all,
            "sum_this": sum_this,
            "sum_prev": sum_prev,
        })

    # まとめPDF（全ファイル分）
    st.divider()
    st.subheader("📄 PDF出力（アップロードした全員分を1つにまとめる）")

    if len(players_payload) == 0:
        st.warning("PDFにまとめるデータがありません。")
        return

    pdf_bytes = build_one_pdf_for_all(players_payload)

    st.download_button(
        label="📥 まとめPDFをダウンロード",
        data=pdf_bytes,
        file_name=f"rapsodo_report_all_{date.today()}.pdf",
        mime="application/pdf"
    )


if __name__ == "__main__":
    main()
