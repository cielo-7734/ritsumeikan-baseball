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


# ==========================================================
# フォント設定（Matplotlib + PDF共通）
# ==========================================================

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


# ==========================================================
# CSV処理
# ==========================================================

def _decode_bytes(b: bytes):
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

        # 3行目の2列目を選手名として取得（形式依存）
        player_name = "Unknown"
        if len(lines) >= 3:
            reader = csv.reader([lines[2]])
            row3 = next(reader, [])
            if len(row3) >= 2:
                player_name = row3[1].strip() or "Unknown"

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

        # 「-」「Other」除外
        df = df[~df["球種"].isin(["-", "Other"])]

        # 日付
        df["datetime"] = pd.to_datetime(df["日付"], errors="coerce")
        df["日付"] = df["datetime"].dt.date

        # ストライク
        if "判定" in df.columns:
            df["ストライク数"] = df["判定"].map({"Y": 1, "N": 0}).fillna(0)
        else:
            df["ストライク数"] = 0

        # 数値変換
        target_cols = ["球速", "回転数", "トゥルースピン", "回転効率", "高さ変化", "横変化"]
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace("-", pd.NA), errors="coerce")

        # 欠損落とし
        df = df.dropna(subset=["球速", "球種", "datetime"])

        return player_name, file_id, df

    except Exception as e:
        st.error(f"解析エラー: {e}")
        return "Error", file_id, pd.DataFrame()


# ==========================================================
# 集計
# ==========================================================

def create_summary(df):
    """球種別サマリー（画面表示・PDF両方で使う）"""
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("球種").agg(
        投球数=("球速", "count"),
        球速平均=("球速", "mean"),
        球速最大=("球速", "max"),
        回転数=("回転数", "mean"),
        トゥルースピン=("トゥルースピン", "mean"),
        回転効率=("回転効率", "mean"),
        変化量高さ=("高さ変化", "mean"),
        変化量横=("横変化", "mean"),
        ストライク率=("ストライク数", "mean"),
    ).reset_index()

    summary["ストライク率(%)"] = summary["ストライク率"] * 100
    summary = summary.drop(columns=["ストライク率"])
    return summary


# ==========================================================
# PDF生成ユーティリティ
# ==========================================================

def fig_to_png_bytesio(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf


def df_for_pdf(df_summary: pd.DataFrame):
    """PDF用：丸め & 欠損を空文字に"""
    if df_summary.empty:
        return df_summary

    out = df_summary.copy()

    # 丸め
    for c in ["球速平均", "球速最大", "回転効率", "変化量高さ", "変化量横", "ストライク率(%)"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    for c in ["回転数", "トゥルースピン"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0)

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
        spaceBefore=6,
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
        ("FONTSIZE", (0, 0), (-1, -1), 7),        # 1ページに寄せるため小さめ
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))


def build_one_pdf_for_all(players_payload):
    """
    players_payload: list of dict
      {
        "player_name": str,
        "file_id": str,
        "date_ranges": {"all": str, "this": str, "prev": str},
        "fig_avg": fig,
        "fig_mov": fig,
        "sum_all": df,      # 表1
        "sum_this": df      # 表2
      }
    PDFは「グラフ2つ＋表2つ（全体・直近30日）」のみ。
    """
    # 日本語フォント
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

    pdf_buf = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=A4,
        leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20
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
        story.append(Spacer(1, 6))

        # ---- グラフ（小さく横並び）----
        story.append(Paragraph("グラフ", h_style))

        avg_png = fig_to_png_bytesio(item["fig_avg"], dpi=170)
        mov_png = fig_to_png_bytesio(item["fig_mov"], dpi=170)

        img_w = 250
        img_h = 160
        img_tbl = Table(
            [[
                RLImage(avg_png, width=img_w, height=img_h),
                RLImage(mov_png, width=img_w, height=img_h),
            ]],
            colWidths=[img_w, img_w],
        )
        img_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(img_tbl)
        story.append(Spacer(1, 4))

        # ---- 表（2つだけ：全体・直近30日）----
        story.append(Paragraph("球種別サマリー", h_style))
        _add_table_to_story(story, base_font, "全体", item["sum_all"])
        _add_table_to_story(story, base_font, "直近30日", item["sum_this"])

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


# ==========================================================
# メイン
# ==========================================================

def main():
    st.title("⚾ ラプソード解析システム")

    files = st.file_uploader("CSVアップロード", accept_multiple_files=True)

    if not files:
        st.info("CSVをアップロードすると解析結果を表示します。")
        return

    players_payload = []

    # 画面表示用フォーマット
    fmt_ui = {
        "投球数": "{:.0f}",
        "球速平均": "{:.1f}",
        "球速最大": "{:.1f}",
        "回転数": "{:.0f}",
        "トゥルースピン": "{:.0f}",
        "回転効率": "{:.1f}",
        "変化量高さ": "{:.1f}",
        "変化量横": "{:.1f}",
        "ストライク率(%)": "{:.1f}",
    }

    for file in files:
        p_name, f_id, df = process_data(file)
        if df.empty:
            continue

        st.header(f"📊 {p_name} のラプソード資料")

        # 球種ごとの色固定
        unique_pitches = sorted(df["球種"].unique())
        pitch_colors = dict(zip(unique_pitches, sns.color_palette("husl", len(unique_pitches))))

        # 日別平均球速
        daily_stats = df.groupby(["日付", "球種"])["球速"].mean().reset_index()

        # --- グラフ作成（PDFにも使うので fig を保持） ---
        col_g1, col_g2 = st.columns(2)

        fig_avg, ax_avg = plt.subplots()
        sns.lineplot(
            data=daily_stats, x="日付", y="球速",
            hue="球種", marker="o", ax=ax_avg, palette=pitch_colors
        )
        title_txt, x_txt, y_txt = ("球速（平均値）", "日付", "球速") if has_font else ("Velocity (Avg)", "Date", "Velocity")
        ax_avg.set_title(title_txt)
        ax_avg.set_xlabel(x_txt)
        ax_avg.set_ylabel(y_txt)
        plt.xticks(rotation=45)
        ax_avg.legend(prop=font_prop) if has_font else ax_avg.legend()

        fig_mov, ax_mov = plt.subplots(figsize=(6, 6))
        sns.scatterplot(
            data=df, x="横変化", y="高さ変化",
            hue="球種", s=80, ax=ax_mov, palette=pitch_colors
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

        # --- 期間（直近30日 / 前月30日） ---
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

        # --- 画面表示：3表（元通り） ---
        st.subheader("📌 球種別サマリー（全体 / 直近30日 / 前月30日）")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 全体")
            if sum_all.empty:
                st.info("データなし")
            else:
                st.dataframe(sum_all.style.format(fmt_ui), use_container_width=True)

        with c2:
            st.markdown(f"### 直近30日（{this_start}〜{this_end}）")
            if sum_this.empty:
                st.info("データなし（期間内の投球がありません）")
            else:
                st.dataframe(sum_this.style.format(fmt_ui), use_container_width=True)

        with c3:
            st.markdown(f"### 前月30日（{prev_start}〜{prev_end}）")
            if sum_prev.empty:
                st.info("データなし（期間内の投球がありません）")
            else:
                st.dataframe(sum_prev.style.format(fmt_ui), use_container_width=True)

        # PDFに載せる期間表記
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

        # PDFは「表2つ」だけ：全体・直近30日
        players_payload.append({
            "player_name": p_name,
            "file_id": f_id,
            "date_ranges": date_ranges,
            "fig_avg": fig_avg,
            "fig_mov": fig_mov,
            "sum_all": sum_all,     # 表1
            "sum_this": sum_this,   # 表2
        })

    # ---- まとめPDF ----
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
