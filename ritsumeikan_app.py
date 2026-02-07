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

# --- 完璧な日本語フォント設定 ---
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
FONT_PATH = "NotoSansJP.ttf"

@st.cache_data
def load_font():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    fm.fontManager.addfont(FONT_PATH)
    prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return prop

try:
    font_prop = load_font()
    has_font = True
except Exception:
    font_prop = None
    has_font = False

st.set_page_config(page_title="Rapsodo Analyzer", layout="wide")


def _decode_bytes(b: bytes):
    """utf-8 がダメなら cp932 で読む（Rapsodo CSV対策）"""
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

    # 見せ方：内部列「ストライク率」は非表示
    show_cols = [c for c in summary.columns if c not in ["ストライク率"]]
    return summary[show_cols]


def main():
    st.title("⚾ ラプソード解析システム")
    files = st.file_uploader("CSVアップロード", accept_multiple_files=True)

    if not files:
        st.info("CSVをアップロードすると解析結果を表示します。")
        return

    for file in files:
        p_name, f_id, df = process_data(file)
        if df.empty:
            continue

        st.header(f"📊 {p_name} のラプソード資料")

        # 球種ごとの色を固定
        unique_pitches = sorted(df["球種"].unique())
        pitch_colors = dict(zip(unique_pitches, sns.color_palette("husl", len(unique_pitches))))

        # 日別（球種別）の球速平均・最大
        daily_stats = df.groupby(["日付", "球種"])["球速"].agg(["mean", "max"]).reset_index()

        # --- グラフ描画 ---
        col_g1, col_g2 = st.columns(2)

        with col_g1:
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
            st.pyplot(fig_avg)

        with col_g2:
            fig_mov, ax_mov = plt.subplots(figsize=(6, 6))
            sns.scatterplot(
                data=df, x="横変化", y="高さ変化",
                hue="球種", s=100, ax=ax_mov, palette=pitch_colors
            )
            title_txt, x_txt, y_txt = ("変化量プロット", "横変化", "高さ変化") if has_font else ("Movement", "HB", "VB")
            ax_mov.set_title(title_txt)
            ax_mov.set_xlabel(x_txt)
            ax_mov.set_ylabel(y_txt)
            ax_mov.axhline(0, linewidth=1)
            ax_mov.axvline(0, linewidth=1)
            ax_mov.legend(prop=font_prop) if has_font else ax_mov.legend()
            st.pyplot(fig_mov)

        # =========================
        # サマリー（全体 / 直近30日 / 前月30日 / 差分）
        # =========================
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
            if sum_all.empty:
                st.info("データなし")
            else:
                st.dataframe(sum_all.style.format(fmt), use_container_width=True)

        with c2:
            st.markdown(f"### 直近30日（{this_start}〜{this_end}）")
            if sum_this.empty:
                st.info("データなし（期間内の投球がありません）")
            else:
                st.dataframe(sum_this.style.format(fmt), use_container_width=True)

        with c3:
            st.markdown(f"### 前月30日（{prev_start}〜{prev_end}）")
            if sum_prev.empty:
                st.info("データなし（期間内の投球がありません）")
            else:
                st.dataframe(sum_prev.style.format(fmt), use_container_width=True)

        st.subheader("📈 差分（直近30日 − 前月30日）")
        if (not sum_this.empty) and (not sum_prev.empty):
            a = sum_this.set_index("球種")
            b = sum_prev.set_index("球種")
            common = a.index.intersection(b.index)

            diff_cols = ["球速平均", "回転数", "回転効率", "変化量高さ", "変化量横", "ストライク率(%)"]
            # 片方にしかない球種は比較できないので common のみ
            diff = (a.loc[common, diff_cols] - b.loc[common, diff_cols]).reset_index()

            st.dataframe(diff.style.format({c: "{:.1f}" for c in diff_cols}), use_container_width=True)
        else:
            st.info("直近30日または前月30日のデータが不足しているため差分を計算できません。")


if __name__ == "__main__":
    main()
