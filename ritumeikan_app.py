import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 日本語フォント設定（MS Gothicがインストールされている環境を想定）
sns.set(font="MS Gothic") 

def process_data(uploaded_file):
    # ファイル名から先頭7桁を取得 [cite: 23]
    file_id = uploaded_file.name[:7]
    
    try:
        # テキストとして読み込み、名前を取得 
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        player_name = "Unknown"
        if len(content) >= 3:
            # 3行目（インデックス2）のB列（2項目目）を抽出
            row3 = content[2].split(',')
            if len(row3) >= 2:
                player_name = row3[1].replace('"', '').strip()
        
        # 5行目ヘッダー、6行目以降を数値データとして読み込み [cite: 18, 19]
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=4)
        
        # 「-」表記を欠損値として削除 [cite: 20]
        df = df.replace('-', pd.NA)
        cols = ['Velocity', 'Total Spin', 'HB (trajectory)', 'VB (trajectory)', 'Spin Efficiency (release)']
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 球速・回転数がない行を削除
        df = df.dropna(subset=['Velocity', 'Total Spin'])
        
        # カラム名を扱いやすく変更
        df = df.rename(columns={'HB (trajectory)': 'HB', 'VB (trajectory)': 'VB', 'Spin Efficiency (release)': 'SpinEff'})
        
        return player_name, file_id, df

    except Exception as e:
        st.error(f"解析エラー: {e}")
        return "Unknown", file_id, pd.DataFrame()

def main():
    st.title("ラプソード データ解析システム")
    
    uploaded_files = st.file_uploader("CSVファイルをアップロードしてください", accept_multiple_files=True)
    
    if uploaded_files:
        all_data = {}
        for file in uploaded_files:
            p_name, f_id, df = process_data(file)
            if not df.empty:
                p_key = f"{f_id}_{p_name}"
                if p_key not in all_data:
                    all_data[p_key] = {"name": p_name, "dfs": []}
                all_data[p_key]["dfs"].append(df)
        
        for p_key, data in all_data.items():
            combined_df = pd.concat(data["dfs"]).drop_duplicates()
            
            # --- タイトルの表示 ---
            st.header(f"📊 {data['name']} のラプソード資料")
            st.info(f"投手ID: {p_key}")
            
            # 1. トレンドグラフ [cite: 27-35]
            st.subheader("① トレンド分析")
            fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
            sns.lineplot(data=combined_df, x='Date', y='Velocity', hue='Pitch Type', marker='o', ax=ax1[0])
            ax1[0].set_title("球速トレンド (Velocity)")
            sns.lineplot(data=combined_df, x='Date', y='Total Spin', hue='Pitch Type', marker='o', ax=ax1[1])
            ax1[1].set_title("回転数トレンド (Total Spin)")
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
            # 2. 変化量散布図 [cite: 36-43]
            
            st.subheader("② 変化量（HB/VB）マップ")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            sns.scatterplot(data=combined_df, x='HB', y='VB', hue='Pitch Type', s=100)
            ax2.set_xlim(-70, 70) # 変化量範囲指定 [cite: 42]
            ax2.set_ylim(-70, 70)
            ax2.axhline(0, color='black', lw=1)
            ax2.axvline(0, color='black', lw=1)
            ax2.set_xlabel("横変化量 (HB)")
            ax2.set_ylabel("縦変化量 (VB)")
            st.pyplot(fig2)

            # 3. 球種別サマリー表 [cite: 44-54]
            st.subheader("③ 球種別集計サマリー")
            summary = combined_df.groupby('Pitch Type').agg({
                'Velocity': ['mean', 'max'],
                'Total Spin': ['mean', 'max'],
                'SpinEff': 'mean',
                'VB': 'mean',
                'HB': 'mean'
            })
            
            # Fastball基準の相対球速計算 [cite: 1]
            if 'Fastball' in summary.index:
                fb_avg = summary.loc['Fastball', ('Velocity', 'mean')]
                summary['球速比率(対FB %)'] = (summary[('Velocity', 'mean')] / fb_avg) * 100
            
            st.dataframe(summary.style.format("{:.1f}"))

if __name__ == "__main__":
    main()