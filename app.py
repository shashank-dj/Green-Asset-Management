
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    compute_sustainability_kpis,
    get_daily_insights,
    generate_recommendations,
)

# File paths in GitHub repo
DATA_PATH_MAIN = "GreenAssetMonitor/low_carbon_park_energy_data.csv"
DATA_PATH_CAT = "GreenAssetMonitor/low_carbon_park_energy_data_categorical.csv"

# Streamlit page settings
st.set_page_config(page_title="Green Asset Monitor", layout="wide")
st.title("🌱 Green Asset Monitor – Industrial Sustainability Dashboard")

st.markdown(
    """
This dashboard provides insights into emissions, renewable energy usage, 
energy mix, and reliability for a low-carbon industrial park.
"""
)

# Sidebar
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DATA_PATH_MAIN)

# Preprocessing
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Renewable_Gen_kWh"] = df["Solar_Gen_kWh"] + df["Wind_Gen_kWh"]
df["Renewable_Penetration"] = df["Renewable_Gen_kWh"] / df["Load_Demand_kWh"]
df["CO2_Intensity"] = df["Carbon_Emission_kg"] / df["Load_Demand_kWh"]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 KPIs", "📈 Visual Insights", "🔎 Daily Metrics", "💡 Recommendations"])

# ===================== TAB 1 – KPI SUMMARY =====================
with tab1:
    st.subheader("Key Performance Indicators")
    kpis = compute_sustainability_kpis(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total CO₂ Emissions", f"{kpis['total_co2']:.2f} kg")
    col2.metric("Avg CO₂ Intensity", f"{kpis['avg_co2_intensity']:.3f} kg/kWh")
    col3.metric("Renewable Penetration", f"{kpis['renewable_penetration']*100:.2f}%")

    col4, col5 = st.columns(2)
    col4.metric("Total Energy Demand", f"{kpis['total_energy_demand']:.2f} kWh")
    col5.metric("Total Grid Import", f"{kpis['total_grid_import']:.2f} kWh")

    st.write("---")
    st.write("### Raw Dataset Preview")
    st.dataframe(df.head())

# ===================== TAB 2 – VISUAL INSIGHTS =====================
with tab2:
    st.subheader("Energy Mix Over Time")
    fig = px.area(
        df,
        x="Timestamp",
        y=["Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh"],
        title="Energy Mix Over Time",
        labels={"value": "Energy (kWh)", "Timestamp": "Time"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Carbon Emissions")
    df_daily = get_daily_insights(df)
    fig2 = px.line(df_daily, x="Date", y="total_co2", title="Daily Carbon Emissions (kg)")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Daily Renewable Penetration")
    fig3 = px.line(df_daily, x="Date", y="avg_renewable_penetration", title="Daily Renewable Penetration")
    st.plotly_chart(fig3, use_container_width=True)

# ===================== TAB 3 – DAILY METRICS =====================
with tab3:
    st.subheader("Daily Summary Table")
    st.dataframe(df_daily)

    csv = df_daily.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Daily Metrics CSV", csv, "daily_metrics.csv")

# ===================== TAB 4 – RECOMMENDATIONS =====================
with tab4:
    st.subheader("Automated Sustainability Recommendations")

    recommendations = generate_recommendations(df, df_daily)

    if len(recommendations) == 0:
        st.success("No anomalies detected. Operations are stable and efficient. 🌱")
    else:
        for r in recommendations:
            st.warning(r)
