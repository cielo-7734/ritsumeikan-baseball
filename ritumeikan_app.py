import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # 日付変換（時刻を切り捨てて「日」単位にする）
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

            # --- グラフ表示 ---
            col1, col2 = st.columns(2)
            with col1:
                # 球速グラフ（日ごとにポイント、タイトル・軸名指定）
                fig1, ax1 = plt.subplots()
                sns.stripplot(data=df, x='日付', y='球速', hue='球種', dodge=True, ax=ax1)
                ax1.set_title("球速")
                ax1.set_xlabel("日付")
                ax1.set_ylabel("球速")
                plt.xticks(rotation=45)
                st.pyplot(fig1)
            
            with col2:
                # 変化量グラフ（タイトル・軸名指定）
                
                fig2, ax2 = plt.subplots()
                sns.scatterplot(data=df, x='横変化', y='高さ変化', hue='球種', s=100, ax=ax2)
                ax2.axhline(0, color='black', lw=1); ax2.axvline(0, color='black', lw=1)
                ax2.set_xlim(-70, 70); ax2.set_ylim(-70, 70)
                ax2.set_title("変化量")
                ax2.set_xlabel("横変化量")
                ax2.set_ylabel("縦変化量")
                st.pyplot(fig2)

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