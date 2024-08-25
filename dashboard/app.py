import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch import mode
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go


# Konfigurasi halaman
st.set_page_config(
    page_title="Football Player Analysis Dashboard",
    page_icon="⚽",
    layout="wide",
)

# Fungsi untuk mengkonversi nilai mata uang ke numerik
def convert_currency(value):
    if pd.isna(value):
        return np.nan
    value = str(value).replace('€', '').replace(',', '.').strip()
    if 'M' in value:
        return float(value.replace('M', '')) * 1e6
    elif 'K' in value:
        return float(value.replace('K', '')) * 1e3
    return float(value)

# Fungsi untuk format mata uang untuk visualisasi
def format_euro(value):
    if pd.isna(value):
        return np.nan
    if value >= 1e6:
        return f"€{value / 1e6:.1f}M"
    elif value >= 1e3:
        return f"€{value / 1e3:.1f}K"
    return f"€{value:.0f}"

# Fungsi untuk load data
@st.cache_data
def load_data():
    df = pd.read_csv('data_cleaned.csv')
    df['Value_numeric'] = df['Value'].apply(convert_currency)
    df['Wage_numeric'] = df['Wage'].apply(convert_currency)
    df['Release_clause_numeric'] = df['Release clause'].apply(convert_currency)
    df['Value_numeric'].fillna(df['Value_numeric'].mean(), inplace=True)
    df['Wage_numeric'].fillna(df['Wage_numeric'].mean(), inplace=True)
    df['Release_clause_numeric'].fillna(df['Release_clause_numeric'].mean(), inplace=True)
    df['Value_formatted'] = df['Value_numeric'].apply(format_euro)
    df['Wage_formatted'] = df['Wage_numeric'].apply(format_euro)
    df['Release_clause_formatted'] = df['Release_clause_numeric'].apply(format_euro)
    df['Age_category'] = pd.cut(df['Age'], bins=[15, 20, 25, 30, 35, 40, 45], labels=[1, 2, 3, 4, 5, 6], right=False)
    Age_labels = {1: '16,17,18,19,20', 2: '21,22,23,24,25', 3: '26,27,28,29,30', 4: '31,32,33,34,35', 5: '36,37,38,39,40', 6: '41,42,43,44'}
    df['Age_label'] = df['Age_category'].map(Age_labels)
    return df

df = load_data()

# Tampilan Dashboard
st.sidebar.title("Football Player Dashboard ⚽")
st.sidebar.markdown("Navigasi:")
page = st.sidebar.selectbox("Pilih halaman", ["Overview", "Analysis", "Data","What's You Looking For","Comparison","Team Overview","Transfer Market"])

if page == "Overview":
    st.title('Football Player Analysis Overview ⚽')

    # Card Layout untuk Statistik Utama
    st.markdown("## 📊 Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", df.shape[0])   
    col2.metric("Average Value", f"€{df['Value_numeric'].mean():,.2f}")
    col3.metric("Average Wage", f"€{df['Wage_numeric'].mean():,.2f}")
    
    st.markdown("---")

    # Data Overview
    st.header('Data Overview')
    st.dataframe(df.head(10))



elif page == "Analysis":
    sns.set_theme(style="whitegrid")
    st.title('Football Player Analysis and Visualizations')

    # Menampilkan daftar pemain dengan nilai Value tertinggi dari seluruh pemain
    st.markdown("### 🏆 Top Players Overall by Value")
    
    top_players = df.sort_values(by='Value_numeric', ascending=False).head(50)
    top_players.reset_index(drop=True, inplace=True)
    top_players.index += 1  # Mulai index dari 1
    
    st.dataframe(top_players[['name', 'team', 'Value_formatted', 'Wage_formatted','Release_clause_formatted', 'Best position']])
    st.markdown("---")

    # Menampilkan daftar pemain dengan nilai Value tertinggi berdasarkan setiap posisi di kolom "Best position"
    st.markdown("### 🏆 Top Players by Position")

    positions = df['Best position'].unique()

    for position in positions:
        st.markdown(f"#### {position} - Top Players")
        
        top_position_players = df[df['Best position'] == position].sort_values(by='Value_numeric', ascending=False).head(10)
        top_position_players.reset_index(drop=True, inplace=True)
        top_position_players.index += 1  # Mulai index dari 1
        
        st.dataframe(top_position_players[['name', 'team', 'Value_formatted', 'Wage_formatted','Release_clause_formatted', 'Best position']])
        st.markdown("---")

    # Visualisasi pengaruh Age_category terhadap Value menggunakan Plotly
    st.markdown("### 🎨 Influence of Age Category on Player Value")

    fig = px.bar(df, x='Age_category', y='Value_numeric', color='Age_category', 
                labels={'Value_numeric':'Average Player Value'},
                title='Average Player Value by Age Category',
                template='plotly_dark')
    fig.update_layout(title_font_size=20, title_x=0.5)
    st.plotly_chart(fig)

    # Distribusi nilai pemain menggunakan Plotly
    st.markdown("### 📊 Distribution of Player Values")

    fig = px.histogram(df, x='Value_numeric', nbins=30, marginal="box", 
                    title='Distribution of Player Values', 
                    template='plotly_dark', color_discrete_sequence=['#008080'])
    fig.update_layout(title_font_size=20, title_x=0.5)
    st.plotly_chart(fig)

    
    # Rata-rata nilai pemain berdasarkan posisi terbaik menggunakan Plotly
    st.markdown("### 📊 Average Player Value by Best Position")

    fig = px.bar(df, x='Best position', y='Value_numeric', color='Best position', 
                labels={'Value_numeric':'Average Player Value'},
                title='Average Player Value by Best Position',
                template='plotly_dark')
    fig.update_layout(title_font_size=20, title_x=0.5, xaxis_tickangle=-45)
    st.plotly_chart(fig)

    # Distribution of Player Value by Dominant Foot
    
    
    st.markdown("### Distribution of Player Value by Dominant Foot")
    # Hitung jumlah pemain berdasarkan kaki dominan
    foot_counts = df['foot'].value_counts()
    
    # Buat pie chart untuk distribusi pemain berdasarkan kaki dominan
    fig, ax = plt.subplots()
    ax.pie(foot_counts, labels=foot_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    ax.set_title('Distribution of Players by Dominant Foot')
    st.pyplot(fig)

    # Analisis pengaruh kaki dominan terhadap nilai pemain
    st.markdown("### Average Player Value by Dominant Foot")
    
   # Rata-rata nilai pemain berdasarkan kaki dominan menggunakan Plotly
    st.markdown("### 📊 Average Player Value by Dominant Foot")

    avg_value_by_foot = df.groupby('foot')['Value_numeric'].mean().reset_index()

    fig = px.bar(avg_value_by_foot, x='foot', y='Value_numeric', color='foot',
                labels={'Value_numeric':'Average Player Value'},
                title='Average Player Value by Dominant Foot',
                template='plotly_dark', color_discrete_sequence=['#ff9999','#66b3ff'])
    fig.update_layout(title_font_size=20, title_x=0.5)
    st.plotly_chart(fig)



    # Visualisasi Heatmap Korelasi Fitur Terhadap Value
    st.markdown("### 🔥 Correlation Heatmap of Features Affecting Player Value")
    # Pilih kolom numerik
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop kolom yang tidak diinginkan
    exclude_columns = ['name', 'Best position', 'team']  # Kolom yang tidak akan dimasukkan dalam korelasi
    relevant_columns = [col for col in numeric_columns if col not in exclude_columns]

    # Kalkulasi korelasi hanya terhadap 'Value'
    corr_with_value = df[relevant_columns].corrwith(df['Value'])

    # Sortir berdasarkan korelasi
    corr_with_value = corr_with_value.sort_values(ascending=False)

    # Visualisasi sebagai bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_with_value.to_frame(), annot=True, cmap='RdBu_r', center=0, cbar=True, ax=ax)
    ax.set_title('Correlation of Numerical Features with Value')

    st.pyplot(fig)

elif page == "Data":
    st.title('Explore the Data')

    # Show raw data
    st.dataframe(df)

    # Save cleaned data
    if st.button('Save Cleaned Data'):
        df.to_csv('data_cleaned.csv', index=False)
        st.success("Data telah disimpan sebagai 'data_cleaned.csv'")

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Use the sidebar to navigate through different sections of the dashboard.")

elif page == "What's You Looking For":
    
    # Tombol search
    st.title("Find Your Desired Players ⚽")
    
    # Input preferensi user untuk rentang usia dan posisi
    age_range = st.selectbox(
        "Select Age Range", 
        options=df['Age_label'].unique(),
        index=0
    )
    
    position = st.selectbox(
        "Select Position",
        options=df['Best position'].unique(),
        index=0
    )

    Dominant_foot = st.selectbox(
        "Select Dominant Foot",
        options=df['foot'].unique(),
    )
    
    # Input rentang nilai dalam Euro menggunakan slider
    min_value, max_value = st.slider(
        "Select Value Range (in Euros)", 
        min_value=int(df['Value_numeric'].min()), 
        max_value=int(df['Value_numeric'].max()), 
        value=(int(df['Value_numeric'].min()), int(df['Value_numeric'].max())), 
        step=1000000,
        key="value_range_slider"
    )
    
    # Tombol search
    if st.button('Search'):
        # Filter data berdasarkan preferensi yang dipilih
        filtered_players = df[
            (df['Age_label'] == age_range) &
            (df['Best position'] == position) &
            (df['Value_numeric'] >= min_value) &
            (df['Value_numeric'] <= max_value)
        ]
        
        # Tampilkan hasil pencarian
        if not filtered_players.empty:
            st.markdown(f"### Players with Age {age_range}, Position {position}, and Value between €{min_value:,} and €{max_value:,}")
            st.dataframe(filtered_players[['name', 'team', 'Value_formatted', 'Wage_formatted', 'Release_clause_formatted', 'Best position']])
        else:
            st.warning("No players found with the selected criteria.")
    

if page == "Comparison":
    st.title('Compare Football Players ⚽')

     # Pilih posisi terlebih dahulu
    position = st.selectbox("Select Position", options=df['Best position'].unique())

    # Pastikan bahwa posisi telah dipilih
    if position:
        # Filter nama pemain berdasarkan posisi yang dipilih
        available_players = df[df['Best position'] == position]['name'].unique()
        
        # Pilih pemain yang akan dibandingkan, hanya menampilkan pemain berdasarkan posisi yang dipilih
        players = st.multiselect("Select Players to Compare", options=available_players)
        
        if len(players) > 1:
            # Filter data berdasarkan pemain yang dipilih
            comparison_data = df[df['name'].isin(players)]

            st.markdown(f"### Comparing Players: {', '.join(players)}")
            
            # Tampilkan tabel perbandingan
            st.dataframe(comparison_data[['name', 'team', 'Value_formatted', 'Wage_formatted', 'Release_clause_formatted', 'Best position', 'Age', 'foot']])

            # Visualisasi perbandingan nilai pemain
            st.markdown("### Player Value Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x='name', y='Value_numeric', data=comparison_data, ax=ax)
            ax.set_title('Player Value Comparison')
            ax.set_xlabel('Player')
            ax.set_ylabel('Value (Numeric)')
            st.pyplot(fig)

            # Visualisasi perbandingan usia pemain
            st.markdown("### Player Age Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x='name', y='Age', data=comparison_data, ax=ax)
            ax.set_title('Player Age Comparison')
            ax.set_xlabel('Player')
            ax.set_ylabel('Age')
            st.pyplot(fig)

            # Visualisasi perbandingan gaji pemain
            st.markdown("### Player Wage Comparison")
            fig, ax = plt.subplots()
            sns.barplot(x='name', y='Wage_numeric', data=comparison_data, ax=ax)
            ax.set_title('Player Wage Comparison')
            ax.set_xlabel('Player')
            ax.set_ylabel('Wage (Numeric)')
            st.pyplot(fig)
        else:
            st.warning("Please select at least two players to compare.")
    else:
        st.warning("Please select a position first.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar to navigate through different sections of the dashboard.")


if page == "Team Overview":
    st.title('Team Overview ⚽')

    # Pilih tim dengan daftar yang diurutkan secara abjad
    team = st.selectbox("Select Team", options=sorted(df['team'].unique()))

    if team:
        # Filter data berdasarkan tim yang dipilih
        team_data = df[df['team'] == team]
        
        st.markdown(f"### Overview for {team}")

        # Tampilkan statistik dasar untuk tim
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Players", team_data.shape[0])
        col2.metric("Average Value", f"€{team_data['Value_numeric'].mean():,.2f}")
        col3.metric("Average Wage", f"€{team_data['Wage_numeric'].mean():,.2f}")

        st.markdown("---")

        # Tampilkan tabel pemain dari tim yang dipilih dengan tampilan yang lebih baik
        st.markdown(f"### Players in {team}")
        st.dataframe(team_data[['name', 'Value_formatted', 'Wage_formatted', 'Release_clause_formatted', 'Best position', 'Age', 'foot']], use_container_width=True)

        st.markdown("---")

        # Visualisasi distribusi nilai pemain dalam tim menggunakan Plotly
        st.markdown(f"### Distribution of Player Values in {team}")
        fig_value = px.histogram(team_data, x='Value_numeric', nbins=10, title=f'Distribution of Player Values in {team}', labels={'Value_numeric': 'Value (Numeric)'})
        fig_value.update_layout(bargap=0.2)
        st.plotly_chart(fig_value, use_container_width=True)

        # Visualisasi distribusi usia pemain dalam tim menggunakan Plotly
        st.markdown(f"### Age Distribution of Players in {team}")
        fig_age = px.histogram(team_data, x='Age', nbins=10, title=f'Age Distribution of Players in {team}', labels={'Age': 'Age'})
        fig_age.update_layout(bargap=0.2)
        st.plotly_chart(fig_age, use_container_width=True)

        # Visualisasi jumlah pemain berdasarkan posisi terbaik menggunakan Plotly
        st.markdown(f"### Players by Best Position in {team}")
        position_counts = team_data['Best position'].value_counts().reset_index()
        position_counts.columns = ['Best Position', 'Count']
        fig_position = px.bar(
            position_counts,
            x='Best Position',
            y='Count',
            title=f'Players by Best Position in {team}',
            labels={'Best Position': 'Best Position', 'Count': 'Count'}
        )
        fig_position.update_layout(xaxis_title='Best Position', yaxis_title='Count', xaxis_tickangle=-45)
        st.plotly_chart(fig_position, use_container_width=True)

    else:
        st.warning("Please select a team to view its overview.")


# Transfer Market Page
elif page == "Transfer Market":
    st.title("Transfer Market Overview ⚽")

    # Transfer Window Tracker
    st.markdown("## ⏳ Transfer Window Tracker")
    st.info("The current transfer window closes in: **15 days, 4 hours, 32 minutes**")
    
    st.markdown("---")

    # Top Transfers
    st.markdown("## 💸 Top Transfers")
    top_transfers = df.sort_values(by='Value_numeric', ascending=False).head(10)
    
    # Gunakan plotly untuk visualisasi yang lebih interaktif
    fig_top_transfers = px.bar(
        top_transfers,
        x='name',
        y='Value_numeric',
        color='Value_numeric',
        labels={'Value_numeric': 'Value (in €)', 'name': 'Player'},
        title='Top 10 Transfers by Value',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_top_transfers.update_layout(xaxis_title='Player', yaxis_title='Value (Numeric)')
    st.plotly_chart(fig_top_transfers)

    st.markdown("---")

    # Transfer Activity per Team
    st.markdown("## 🔄 Transfer Activity per Team")
    team = st.selectbox("Select a team to view transfer activity", sorted(df['team'].unique()))
    
    if team:
        # Filter transfer data
        transfer_in = df[df['team'] == team]
        transfer_out = df[df['Previous_team'] == team] if 'Previous_team' in df.columns else pd.DataFrame()

        st.markdown(f"### Incoming Transfers to {team}")
        if not transfer_in.empty:
            st.dataframe(transfer_in[['name', 'Value_formatted', 'Best position', 'Age']])
            
            # Visualisasi Incoming Transfers
            fig_incoming_transfers = px.pie(
                transfer_in,
                names='Best position',
                values='Value_numeric',
                title=f'Incoming Transfers by Position for {team}',
                color='Best position',
                color_discrete_map={
                    'Goalkeeper': 'blue',
                    'Defender': 'green',
                    'Midfielder': 'red',
                    'Forward': 'purple'
                }
            )
            st.plotly_chart(fig_incoming_transfers)
        else:
            st.write("No incoming transfers found.")

        st.markdown("---")

        st.markdown(f"### Outgoing Transfers from {team}")
        if not transfer_out.empty:
            st.dataframe(transfer_out[['name', 'Value_formatted', 'Best position', 'Age']])
            
            # Visualisasi Outgoing Transfers
            fig_outgoing_transfers = px.bar(
                transfer_out,
                x='name',
                y='Value_numeric',
                color='Value_numeric',
                labels={'Value_numeric': 'Value (in €)', 'name': 'Player'},
                title=f'Outgoing Transfers by Value for {team}',
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig_outgoing_transfers.update_layout(xaxis_title='Player', yaxis_title='Value (Numeric)')
            st.plotly_chart(fig_outgoing_transfers)
        else:
            st.write("No outgoing transfers found.")

    st.markdown("---")
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Stay tuned for the latest transfer updates!")


# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar to navigate through different sections of the dashboard.")