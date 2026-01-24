import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ページ設定
st.set_page_config(page_title="Rapsodo Data Analyzer", layout="wide")
sns.set(font="MS Gothic") # 日本語フォント設定

def process_data(uploaded_file):
    # ファイル名から先頭7桁を取得 [cite: 23]
    file_id = uploaded_file.name[:7]
    
    try:
        # CSVを一度文字列として読み込み、行ごとに分割
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        
        # 3行目（インデックス2）のB列（2項目目）から名前を取得 [cite: 24]
        # split(',')で分割し、[1]がB列に相当します
        player_name = "Unknown"
        if len(content) >= 3:
            row3_items = content[2].split(',')
            if len(row3_items) >= 2:
                player_name = row3_items[1].replace('"', '').strip()
        
        # 5行目（インデックス4）をヘッダー、6行目以降を数値データとして読み込む [cite: 18, 19]
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=4)
        
        # 「-」を欠損値として削除し、数値を変換 [cite: 20]
        df = df.replace('-', pd.NA)
        cols = ['Velocity', 'Total Spin', 'HB', 'VB']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Velocity', 'Total Spin'])
        
        return player_name, file_id, df

    except Exception as e:
        st.error(f"ファイル解析エラー ({uploaded_file.name}): {e}")
        return "Unknown", file_id, pd.DataFrame()

def main():
    st.title("ラプソード データ解析システム")
    
    uploaded_files = st.file_uploader("CSVファイルをアップロードしてください", accept_multiple_files=True)
    
    if uploaded_files:
        all_data = {}
        
        for file in uploaded_files:
            p_name, f_id, df = process_data(file)
            if not df.empty:
                # 名前をキーにしてデータをまとめる [cite: 25]
                p_key = f"{f_id}_{p_name}"
                if p_key not in all_data:
                    all_data[p_key] = {"name": p_name, "dfs": []}
                all_data[p_key]["dfs"].append(df)
        
        for p_key, data in all_data.items():
            combined_df = pd.concat(data["dfs"]).drop_duplicates()
            
            # --- ここでタイトルを指定の形式に変更 ---
            st.header(f"📊 {data['name']} のラプソード資料")
            st.subheader(f"投手ID: {p_key}")
            
            # 1. トレンドグラフ (球速・回転数) [cite: 27-30]
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                sns.lineplot(data=combined_df, x='Date', y='Velocity', hue='Pitch Type', marker='o')
                ax1.set_title("Velocity Trend")
                st.pyplot(fig1)
            
            with col2:
                fig2, ax2 = plt.subplots()
                sns.lineplot(data=combined_df, x='Date', y='Total Spin', hue='Pitch Type', marker='o')
                ax2.set_title("Total Spin Trend")
                st.pyplot(fig2)
            
            # 2. 変化量散布図 (HB x VB) [cite: 36-43]
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=combined_df, x='HB', y='VB', hue='Pitch Type', s=100)
            ax3.set_xlim(-70, 70)
            ax3.set_ylim(-70, 70)
            ax3.axhline(0, color='black', lw=1)
            ax3.axvline(0, color='black', lw=1)
            ax3.set_title("HB vs VB Trajectory")
            st.pyplot(fig3)

            # 3. 集計表 [cite: 44-54]
            st.write("### 球種別サマリー")
            summary = combined_df.groupby('Pitch Type').agg({
                'Velocity': ['mean', 'max'],
                'Total Spin': ['mean', 'max']
            })
            st.table(summary)

if __name__ == "__main__":
    main()