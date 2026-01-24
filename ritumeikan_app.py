import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import urllib.request
import os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# --- 日本語フォント設定 ---
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP%5Bwght%5D.ttf"
FONT_PATH = "NotoSansJP.ttf"
if not os.path.exists(FONT_PATH):
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)
prop = fm.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family'] = prop.get_name()

st.set_page_config(page_title="Rapsodo Analyzer", layout="wide")

def process_data(uploaded_file):
    file_id = uploaded_file.name[:7]
    try:
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        player_name = "Unknown"
        if len(content) >= 3:
            import csv
            reader = csv.reader([content[2]])
            row3 = next(reader)
            if len(row3) >= 2:
                player_name = row3[1].strip()
        
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=4)
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        
        rename_dict = {
            'Pitch Type': '球種', 'Velocity': '球速', 'Total Spin': '回転数',
            'True Spin (release)': 'トゥルースピン', 'Spin Efficiency (release)': '回転効率',
            'VB (trajectory)': '高さ変化', 'HB (trajectory)': '横変化',
            'Date': '日付', 'Is Strike': '判定'
        }
        df = df.rename(columns=rename_dict)
        df = df[~df['球種'].isin(['-', 'Other'])]
        
        df['datetime'] = pd.to_datetime(df['日付'], errors='coerce')
        df['日付'] = df['datetime'].dt.date
        
        if '判定' in df.columns:
            df['ストライク数'] = df['判定'].map({'Y': 1, 'N': 0}).fillna(0)
        
        target_cols = ['球速', '回転数', 'トゥルースピン', '回転効率', '高さ変化', '横変化']
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('-', pd.NA), errors='coerce')
        
        return player_name, file_id, df.dropna(subset=['球速', '球種', 'datetime'])
    except Exception as e:
        st.error(f"解析エラー: {e}")
        return "Error", file_id, pd.DataFrame()

def create_summary(df):
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby('球種').agg({
        '球速': ['mean', 'max'], '回転数': 'mean', 'トゥルースピン': 'mean',
        '回転効率': 'mean', '高さ変化': 'mean', '横変化': 'mean', 'ストライク数': 'mean'
    })
    summary.columns = [
        '球速(平均)', '球速(最大)', '回転数', 'トゥルースピン', 
        '回転効率(%)', '変化量(高さ)', '変化量(横)', 'ストライク率(%)'
    ]
    summary['ストライク率(%)'] = summary['ストライク率(%)'] * 100
    
    if 'Fastball' in summary.index:
        fb_v = summary.loc['Fastball', '球速(平均)']
        summary['球速比率(対FB %)'] = (summary['球速(平均)'] / fb_v) * 100
    
    return summary.style.format("{:.1f}")

def main():
    st.title("⚾ ラプソード解析システム")
    files = st.file_uploader("CSVアップロード", accept_multiple_files=True)
    
    if files:
        for file in files:
            p_name, f_id, df = process_data(file)
            if df.empty: continue
            
            st.header(f"📊 {p_name} のラプソード資料")

            # 色設定
            unique_pitches = sorted(df['球種'].unique())
            pitch_colors = dict(zip(unique_pitches, sns.color_palette("husl", len(unique_pitches))))

            # グラフ表示
            daily_stats = df.groupby(['日付', '球種'])['球速'].agg(['mean', 'max']).reset_index()
            st.subheader("📈 球速・変化量分析")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_avg, ax_avg = plt.subplots()
                sns.lineplot(data=daily_stats, x='日付', y='mean', hue='球種', marker='o', ax=ax_avg, palette=pitch_colors)
                ax_avg.set_title("球速（平均値）", fontproperties=prop)
                plt.xticks(rotation=45)
                st.pyplot(fig_avg)
            with col_g2:
                fig_mov, ax_mov = plt.subplots(figsize=(6, 6))
                sns.scatterplot(data=df, x='横変化', y='高さ変化', hue='球種', s=100, ax=ax_mov, palette=pitch_colors)
                ax_mov.axhline(0, color='black', lw=1); ax_mov.axvline(0, color='black', lw=1)
                ax_mov.set_xlim(-70, 70); ax_mov.set_ylim(-70, 70)
                ax_mov.set_title("変化量マップ", fontproperties=prop)
                st.pyplot(fig_mov)

            # --- データの分割（今月 vs 前3か月） ---
            latest_date = df['datetime'].max()
            this_month_start = latest_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # 前3か月の開始日を計算 (例: 今が8月なら、5, 6, 7月のデータを取得)
            three_months_ago_start = this_month_start - relativedelta(months=3)
            
            df_this_month = df[df['datetime'] >= this_month_start]
            df_last_3_months = df[(df['datetime'] >= three_months_ago_start) & (df['datetime'] < this_month_start)]

            # --- 表の表示 ---
            st.subheader(f"📋 今月のサマリー ({latest_date.strftime('%Y年%m月')})")
            if not df_this_month.empty:
                st.dataframe(create_summary(df_this_month))
            else:
                st.info("今月のデータはありません。")

            st.subheader(f"📋 直近3か月のサマリー ({three_months_ago_start.strftime('%Y/%m')} ～ { (this_month_start - timedelta(days=1)).strftime('%Y/%m') })")
            if not df_last_3_months.empty:
                st.dataframe(create_summary(df_last_3_months))
            else:
                st.info("指定期間（前3か月）のデータはありません。")

if __name__ == "__main__":
    main()