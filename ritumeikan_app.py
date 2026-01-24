import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import urllib.request
import os

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

        # --- 「-」と「Other」を除外 ---
        df = df[~df['球種'].isin(['-', 'Other'])]
        
        df['日付'] = pd.to_datetime(df['日付'], errors='coerce').dt.date
        
        if '判定' in df.columns:
            df['ストライク数'] = df['判定'].map({'Y': 1, 'N': 0}).fillna(0)
        
        target_cols = ['球速', '回転数', 'トゥルースピン', '回転効率', '高さ変化', '横変化']
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('-', pd.NA), errors='coerce')
        
        return player_name, file_id, df.dropna(subset=['球速', '球種'])
    except Exception as e:
        st.error(f"解析エラー: {e}")
        return "Error", file_id, pd.DataFrame()

def main():
    st.title("⚾ ラプソード解析システム")
    files = st.file_uploader("CSVアップロード", accept_multiple_files=True)
    
    if files:
        for file in files:
            p_name, f_id, df = process_data(file)
            if df.empty: continue
            
            st.header(f"📊 {p_name} のラプソード資料")

            # --- 日ごとの平均とMAXを計算 ---
            daily_stats = df.groupby(['日付', '球種'])['球速'].agg(['mean', 'max']).reset_index()

            # --- グラフ表示 ---
            st.subheader("📈 球速推移分析")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_avg, ax_avg = plt.subplots()
                sns.lineplot(data=daily_stats, x='日付', y='mean', hue='球種', marker='o', ax=ax_avg)
                ax_avg.set_title("球速（平均値）", fontproperties=prop)
                ax_avg.set_xlabel("日付", fontproperties=prop)
                ax_avg.set_ylabel("平均球速 (km/h)", fontproperties=prop)
                plt.xticks(rotation=45)
                st.pyplot(fig_avg)
            
            with col2:
                fig_max, ax_max = plt.subplots()
                sns.lineplot(data=daily_stats, x='日付', y='max', hue='球種', marker='o', ax=ax_max, palette="flare")
                ax_max.set_title("球速（MAX値）", fontproperties=prop)
                ax_max.set_xlabel("日付", fontproperties=prop)
                ax_max.set_ylabel("最高球速 (km/h)", fontproperties=prop)
                plt.xticks(rotation=45)
                st.pyplot(fig_max)

            # --- 変化量グラフ ---
            st.subheader("🎯 変化量分析")
            
            fig_mov, ax_mov = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=df, x='横変化', y='高さ変化', hue='球種', s=100, ax=ax_mov)
            ax_mov.axhline(0, color='black', lw=1); ax_mov.axvline(0, color='black', lw=1)
            ax_mov.set_xlim(-70, 70); ax_mov.set_ylim(-70, 70)
            ax_mov.set_title("変化量", fontproperties=prop)
            ax_mov.set_xlabel("横変化量", fontproperties=prop)
            ax_mov.set_ylabel("縦変化量", fontproperties=prop)
            st.pyplot(fig_mov)

            # --- 集計表 ---
            st.subheader("📋 球種別サマリー")
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
            
            st.dataframe(summary.style.format("{:.1f}"))

if __name__ == "__main__":
    main()