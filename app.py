import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    compute_sustainability_kpis,
    get_daily_insights,
    generate_recommendations,
)

# Optional ML imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -------------------------
# Config & file paths
# -------------------------
DATA_PATH_MAIN = "GreenAssetMonitor/low_carbon_park_energy_data.csv"
DATA_PATH_CAT = "GreenAssetMonitor/low_carbon_park_energy_data_categorical.csv"

st.set_page_config(page_title="Green Asset Monitor", layout="wide")
st.title("🌱 Green Asset Monitor – Industrial Sustainability Dashboard")

st.markdown(
    """
This dashboard provides insights into emissions, renewable energy usage, 
energy mix, and reliability for a low-carbon industrial park.
"""
)


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
forecast_periods = st.sidebar.number_input("Forecast periods (days)", min_value=1, max_value=30, value=7)
rf_estimators = st.sidebar.slider("RandomForest estimators", min_value=10, max_value=500, value=150, step=10)


# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_df(path_or_buffer):
    return pd.read_csv(path_or_buffer)

try:
    if uploaded_file:
        df = load_df(uploaded_file)
    else:
        df = load_df(DATA_PATH_MAIN)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# -------------------------
# Preprocessing
# -------------------------
timestamp_cols = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower() or "date" in c.lower()]
if not timestamp_cols:
    st.error("No timestamp-like column found.")
    st.stop()

ts_col = "Timestamp" if "Timestamp" in df.columns else timestamp_cols[0]
df["Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")

if df["Timestamp"].isna().all():
    st.error("Timestamp parsing failed.")
    st.stop()

needed_cols = [
    "Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh",
    "Load_Demand_kWh", "Storage_Input_kWh", "Storage_Output_kWh",
    "EV_Charging_kWh", "Carbon_Emission_kg", "Storage_Level_%", "Reliability_Score"
]

for c in needed_cols:
    if c not in df.columns:
        df[c] = np.nan if c == "Storage_Level_%" else 0.0

df["Renewable_Gen_kWh"] = df["Solar_Gen_kWh"].fillna(0) + df["Wind_Gen_kWh"].fillna(0)
df["Load_Demand_kWh"] = df["Load_Demand_kWh"].replace(0, np.nan)
df["Renewable_Penetration"] = (df["Renewable_Gen_kWh"] / df["Load_Demand_kWh"]).fillna(0)
df["CO2_Intensity"] = (df["Carbon_Emission_kg"] / df["Load_Demand_kWh"]).fillna(0)
df["Storage_Net_kWh"] = df["Storage_Output_kWh"] - df["Storage_Input_kWh"]

df = df.sort_values("Timestamp").reset_index(drop=True)

df_daily = get_daily_insights(df)


# -------------------------
# Forecasting helper
# -------------------------
def forecast_co2(df_in, periods_days=7):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed.")

    df_prophet = df_in.groupby(df_in["Timestamp"].dt.date).agg(
        daily_co2=("Carbon_Emission_kg", "sum")
    ).reset_index()

    df_prophet.columns = ["ds", "y"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods_days)
    forecast = model.predict(future)
    return forecast


# -------------------------
# Reliability classifier helper
# -------------------------
def train_reliability_classifier(df_in, n_estimators=150):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not installed.")

    df_local = df_in.copy()
    if "Reliability_Score" not in df_local.columns:
        raise RuntimeError("Reliability_Score missing.")

    df_local["Reliability_Class"] = df_local["Reliability_Score"].apply(
        lambda x: 2 if x > 70 else 1 if x > 40 else 0
    )

    features = [
        "Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh",
        "Load_Demand_kWh", "Storage_Input_kWh", "Storage_Output_kWh",
        "Carbon_Emission_kg",
    ]

    X = df_local[features].fillna(0)
    y = df_local["Reliability_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    feat_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, report, feat_df


# -------------------------
# Advanced charts
# -------------------------
def plot_correlation_mat(df_in):
    corr = df_in.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="YlGnBu", ax=ax)
    return fig


def energy_sankey(df_in):
    solar = df_in["Solar_Gen_kWh"].sum()
    wind = df_in["Wind_Gen_kWh"].sum()
    grid = df_in["Grid_Import_kWh"].sum()
    storage_in = df_in["Storage_Input_kWh"].sum()
    storage_out = df_in["Storage_Output_kWh"].sum()
    load = df_in["Load_Demand_kWh"].sum()
    emissions = df_in["Carbon_Emission_kg"].sum()

    labels = ["Solar", "Wind", "Grid", "Storage_In", "Storage_Out", "Load", "Emissions"]
    li = {l: i for i, l in enumerate(labels)}

    sources = [
        li["Solar"], li["Wind"], li["Grid"], li["Storage_Out"], li["Grid"], li["Load"]
    ]
    targets = [
        li["Load"], li["Load"], li["Load"], li["Load"], li["Storage_In"], li["Emissions"]
    ]
    values = [solar, wind, grid, storage_out, storage_in, emissions]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values),
    )])
    return fig


def co2_intensity_heatmap(df_in):
    df2 = df_in.copy()
    df2["Hour"] = df2["Timestamp"].dt.hour
    df2["Day"] = df2["Timestamp"].dt.date
    pivot = df2.pivot_table(index="Day", columns="Hour", values="CO2_Intensity").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="Reds", ax=ax)
    return fig


# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab_ml, tab_adv = st.tabs([
    "📊 KPIs", "📈 Visual Insights", "🔎 Daily Metrics",
    "💡 Recommendations", "🤖 ML Models", "📊 Advanced Analytics"
])


# -------------------------------------
# TAB 1 – KPIs
# -------------------------------------
with tab1:
    st.subheader("Key Performance Indicators")
    kpis = compute_sustainability_kpis(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total CO₂ Emissions", f"{kpis['total_co2']:.2f} kg")
    col2.metric("Avg CO₂ Intensity", f"{kpis['avg_co2_intensity']:.3f}")
    col3.metric("Renewable Penetration", f"{kpis['renewable_penetration']*100:.2f}%")

    col4, col5 = st.columns(2)
    col4.metric("Total Demand", f"{kpis['total_energy_demand']:.2f} kWh")
    col5.metric("Grid Import", f"{kpis['total_grid_import']:.2f} kWh")

    st.write("### Dataset Preview")
    st.dataframe(df.head())


# -------------------------------------
# TAB 2 – VISUAL INSIGHTS
# -------------------------------------
with tab2:
    st.subheader("Energy Mix Over Time")
    fig = px.area(
        df, x="Timestamp",
        y=["Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh"],
        title="Energy Mix Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Carbon Emissions")
    fig2 = px.line(df_daily, x="Date", y="total_co2")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Renewable Penetration")
    fig3 = px.line(df_daily, x="Date", y="avg_renewable_penetration")
    st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------
# TAB 3 – DAILY METRICS
# -------------------------------------
with tab3:
    st.subheader("Daily Summary Table")
    st.dataframe(df_daily)

    csv = df_daily.to_csv(index=False).encode()
    st.download_button("Download Daily Metrics CSV", csv, "daily_metrics.csv")


# -------------------------------------
# TAB 4 – RECOMMENDATIONS
# -------------------------------------
with tab4:
    st.subheader("Automated Sustainability Recommendations")
    recs = generate_recommendations(df, df_daily)

    if not recs:
        st.success("No anomalies detected. Operations are stable and efficient.")
    else:
        for r in recs:
            st.warning(r)


# -------------------------------------
# TAB 5 – ML MODELS
# -------------------------------------
with tab_ml:
    st.subheader("ML: Forecasting & Reliability Classification")

    # Forecast
    st.markdown("### 🔮 CO₂ Emission Forecast")
    if PROPHET_AVAILABLE:
        try:
            fc = forecast_co2(df, periods_days=int(forecast_periods))
            fig_fc = px.line(fc, x="ds", y=["y", "yhat"], title="Forecasted CO₂")
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Prophet not installed.")

    st.write("---")

    # RF classifier
    st.markdown("### ⚙️ Reliability Classification")
    if SKLEARN_AVAILABLE:
        try:
            model, report, feat = train_reliability_classifier(df, rf_estimators)
            st.json(report)

            fig_feat = px.bar(feat, x="importance", y="feature",
                              orientation="h", title="Feature Importance")
            st.plotly_chart(fig_feat, use_container_width=True)
        except Exception as e:
            st.error(str(e))
    else:
        st.info("scikit-learn not installed.")


# -------------------------------------
# TAB 6 – ADVANCED ANALYTICS
# -------------------------------------
with tab_adv:
    st.subheader("Correlation Heatmap")
    try:
        st.pyplot(plot_correlation_mat(df))
    except:
        st.warning("Correlation plot failed.")

    st.subheader("Energy Flow Sankey")
    try:
        st.plotly_chart(energy_sankey(df), use_container_width=True)
    except:
        st.warning("Sankey failed.")

    st.subheader("CO₂ Intensity Heatmap")
    try:
        st.pyplot(co2_intensity_heatmap(df))
    except:
        st.warning("Heatmap failed.")

    st.subheader("Renewable vs Load Density")
    try:
        fig_den = px.density_heatmap(df, x="Renewable_Gen_kWh", y="Load_Demand_kWh")
        st.plotly_chart(fig_den, use_container_width=True)
    except:
        st.warning("Density plot failed.")



# -------------------------
# End of App
# -------------------------
st.markdown("---")
st.caption("Green Asset Monitor – Industrial Sustainability Dashboard 🌱")
