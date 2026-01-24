import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import io
import os

# --- 設定 ---
st.set_page_config(page_title="Rapsodo Data Analyzer", layout="wide")
sns.set(font="MS Gothic") # 日本語フォント設定

def process_data(uploaded_file):
    # ファイル名から先頭7桁を取得 
    file_id = uploaded_file.name[:7]
    
    # 3行目からPlayer Nameを取得 
    temp_df = pd.read_csv(uploaded_file, skiprows=2, nrows=1)
    player_name = temp_df.iloc[0, 0] # 3行目の1列目と想定
    
    # 投手IDの作成 
    pitcher_id = f"{file_id}_{player_name}"
    
    # 5行目ヘッダー、6行目データ読み込み [cite: 18, 19]
    df = pd.read_csv(uploaded_file, skiprows=4)
    
    # 「-」を欠損値として削除 [cite: 20]
    df = df.replace('-', pd.NA).dropna(subset=['Velocity', 'Total Spin', 'HB', 'VB'])
    df['Velocity'] = df['Velocity'].astype(float)
    df['Total Spin'] = df['Total Spin'].astype(float)
    
    return pitcher_id, player_name, df

def create_summary_table(df):
    # 球種別の平均とMAX [cite: 44-54]
    summary = df.groupby('Pitch Type').agg({
        'Velocity': ['mean', 'max'],
        'Total Spin': ['mean', 'max'],
        'Spin Efficiency': 'mean',
        'VB': 'mean',
        'HB': 'mean'
    }).reset_index()
    
    # Fastballの平均球速を基準とした割合算出 
    fb_avg_v = summary.loc[summary['Pitch Type'] == 'Fastball', ('Velocity', 'mean')].values
    if len(fb_avg_v) > 0:
        base_v = fb_avg_v[0]
        summary[('Velocity', 'Relative %')] = (summary[('Velocity', 'mean')] / base_v) * 100
        
    return summary

def main():
    st.title("ラプソード データ解析システム")
    
    uploaded_files = st.file_uploader("CSVファイルをアップロードしてください", accept_multiple_files=True)
    
    if uploaded_files:
        all_data = {}
        
        for file in uploaded_files:
            p_id, p_name, df = process_data(file)
            if p_id not in all_data:
                all_data[p_id] = []
            all_data[p_id].append(df)
            
        for p_id, dfs in all_data.items():
            combined_df = pd.concat(dfs).drop_duplicates()
            st.header(f"投手: {p_id}")
            
            # 1. トレンドグラフ (球速・回転数) [cite: 27-35]
            fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
            sns.lineplot(data=combined_df, x='Date', y='Velocity', hue='Pitch Type', marker='o', ax=ax1[0])
            ax1[0].set_title("Velocity Trend")
            sns.lineplot(data=combined_df, x='Date', y='Total Spin', hue='Pitch Type', marker='o', ax=ax1[1])
            ax1[1].set_title("Total Spin Trend")
            st.pyplot(fig1)
            
            # 2. 変化量散布図 (HB x VB) [cite: 36-43]
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=combined_df, x='HB', y='VB', hue='Pitch Type', s=100)
            ax2.set_xlim(-70, 70)
            ax2.set_ylim(-70, 70)
            ax2.axhline(0, color='black', lw=1)
            ax2.axvline(0, color='black', lw=1)
            ax2.set_title("HB vs VB Trajectory")
            st.pyplot(fig2)
            
            # 3. 集計表 [cite: 44]
            summary = create_summary_table(combined_df)
            st.write("### 集計サマリー", summary)

            # PDF出力ボタン (ReportLabを使用) 
            if st.button(f"{p_id} のPDFを出力"):
                # ここにPDF生成ロジックを実装
                st.success(f"{p_id}.pdf を作成しました")

if __name__ == "__main__":
    main()