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

# Optional ML imports (handle missing packages gracefully)
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
# Config / file paths
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
# Sidebar: upload / options
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
    df_local = pd.read_csv(path_or_buffer)
    return df_local

try:
    if uploaded_file:
        df = load_df(uploaded_file)
    else:
        df = load_df(DATA_PATH_MAIN)
except FileNotFoundError as e:
    st.error(f"Could not find the dataset at path: {DATA_PATH_MAIN}. Please upload a CSV in the sidebar or add the file to the repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -------------------------
# Preprocessing
# -------------------------
# Ensure timestamp exists and cast
timestamp_cols = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower() or "date" in c.lower()]
if len(timestamp_cols) == 0:
    st.error("No timestamp-like column found in dataset. Ensure your CSV contains a timestamp column.")
    st.stop()

# Prefer exact 'Timestamp' if present, else use first timestamp-like column
ts_col = "Timestamp" if "Timestamp" in df.columns else timestamp_cols[0]
df["Timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
if df["Timestamp"].isna().all():
    st.error("Failed to parse any timestamps. Check the timestamp format in your CSV.")
    st.stop()

# Normalize / fill columns used by the dashboard
# Provide defaults or add missing derived columns where possible
for col in [
    "Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh",
    "Load_Demand_kWh", "Storage_Input_kWh", "Storage_Output_kWh",
    "EV_Charging_kWh", "Carbon_Emission_kg", "Storage_Level_%", "Reliability_Score"
]:
    if col not in df.columns:
        # create zero columns for missing numeric fields
        if col == "Storage_Level_%":
            df[col] = np.nan
        else:
            df[col] = 0.0

# Derived columns (safe math - avoid division by zero)
df["Renewable_Gen_kWh"] = df["Solar_Gen_kWh"].fillna(0.0) + df["Wind_Gen_kWh"].fillna(0.0)
df["Load_Demand_kWh"] = df["Load_Demand_kWh"].replace(0, np.nan)  # to avoid div-by-zero -> will fill later
df["Renewable_Penetration"] = (df["Renewable_Gen_kWh"] / df["Load_Demand_kWh"]).fillna(0.0)
df["CO2_Intensity"] = (df["Carbon_Emission_kg"] / df["Load_Demand_kWh"]).fillna(0.0)
df["Storage_Net_kWh"] = df["Storage_Output_kWh"].fillna(0.0) - df["Storage_Input_kWh"].fillna(0.0)

# Ensure sorting
df = df.sort_values("Timestamp").reset_index(drop=True)

# Compute daily insights once and reuse
df_daily = get_daily_insights(df)

# -------------------------
# Helper: Forecasting
# -------------------------
def forecast_co2(df_in, periods_days=7):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed. Install with `pip install prophet` to enable forecasting.")
    # Use daily aggregated CO2
    df_prophet = df_in.groupby(df_in['Timestamp'].dt.date).agg(daily_co2=('Carbon_Emission_kg', 'sum')).reset_index()
    df_prophet.columns = ['ds', 'y']
    # Prophet expects ds as datetime
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods_days)
    forecast = m.predict(future)
    return forecast

# -------------------------
# Helper: Reliability classifier
# -------------------------
def train_reliability_classifier(df_in, n_estimators=150):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("Scikit-learn is not installed. Install with `pip install scikit-learn` to enable ML classifier.")
    df_train = df_in.copy()
    # Create categorical classes from Reliability_Score if available
    if "Reliability_Score" not in df_train.columns:
        raise RuntimeError("Reliability_Score column not present in data.")
    df_train['Reliability_Class'] = df_train['Reliability_Score'].apply(lambda x: 2 if x > 70 else 1 if x > 40 else 0)
    features = [
        'Solar_Gen_kWh', 'Wind_Gen_kWh', 'Grid_Import_kWh',
        'Load_Demand_kWh', 'Storage_Input_kWh', 'Storage_Output_kWh',
        'Carbon_Emission_kg'
    ]
    # Replace any NaN with 0 for training features
    X = df_train[features].fillna(0.0)
    y = df_train['Reliability_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    feature_importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return model, report, feature_importance_df

# -------------------------
# Helper: Advanced charts
# -------------------------
def plot_correlation_mat(df_in):
    numeric = df_in.select_dtypes(include=[np.number])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, ax=ax, cmap="YlGnBu", annot=False)
    plt.tight_layout()
    return fig

def energy_sankey(df_in):
    # create nodes and aggregated values for links
    solar = df_in["Solar_Gen_kWh"].sum()
    wind = df_in["Wind_Gen_kWh"].sum()
    grid = df_in["Grid_Import_kWh"].sum()
    storage_in = df_in["Storage_Input_kWh"].sum()
    storage_out = df_in["Storage_Output_kWh"].sum()
    load = df_in["Load_Demand_kWh"].fillna(0.0).sum()
    emissions = df_in["Carbon_Emission_kg"].sum()

    labels = ["Solar", "Wind", "Grid", "Storage_In", "Storage_Out", "Load", "Emissions"]
    # We'll map sources -> targets (simple illustrative mapping)
    label_idx = {l: i for i, l in enumerate(labels)}

    # links (source indices, target indices, values)
    sources = []
    targets = []
    values = []

    # generation -> load
    sources += [label_idx["Solar"], label_idx["Wind"], label_idx["Grid"], label_idx["Storage_Out"]]
    targets += [label_idx["Load"]] * 4
    values += [solar, wind, grid, storage_out]

    # storage charge goes from grid/solar/wind -> Storage_In (illustrative)
    # for sankey simplicity, connect grid -> Storage_In (value = storage_in)
    sources += [label_idx["Grid"]]
    targets += [label_idx["Storage_In"]]
    values += [storage_in]

    # load -> emissions (illustrative)
    sources += [label_idx["Load"]]
    targets += [label_idx["Emissions"]]
    values += [emissions]

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text="Energy Flow Sankey Diagram (aggregated)", font_size=10)
    return fig

def co2_intensity_heatmap(df_in):
    df_tmp = df_in.copy()
    df_tmp['Hour'] = df_tmp['Timestamp'].dt.hour
    df_tmp['Day'] = df_tmp['Timestamp'].dt.date
    pivot = df_tmp.pivot_table(index='Day', columns='Hour', values='CO2_Intensity', aggfunc='mean').fillna(0)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="Reds", ax=ax)
    ax.set_title("CO2 Intensity (kg/kWh) — Hour x Day")
    plt.tight_layout()
    return fig

# -------------------------
# Tabs: extended layout
# -------------------------
tab1, tab2, tab3, tab4, tab_ml, tab_adv = st.tabs([
    "📊 KPIs",
    "📈 Visual Insights",
    "🔎 Daily Metrics",
    "💡 Recommendations",
    "🤖 ML Models",
    "📊 Advanced Analytics"
])

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
    st.write("### Raw Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10))

# ===================== TAB 2 – VISUAL INSIGHTS =====================
with tab2:
    st.subheader("Energy Mix Over Time")
    fig_area = px.area(
        df,
        x="Timestamp",
        y=["Solar_Gen_kWh", "Wind_Gen_kWh", "Grid_Import_kWh"],
        title="Energy Mix Over Time",
        labels={"value": "Energy (kWh)", "Timestamp": "Time"},
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("Daily Carbon Emissions")
    fig_co2 = px.line(df_daily, x="Date", y="total_co2", title="Daily Carbon Emissions (kg)")
    st.plotly_chart(fig_co2, use_container_width=True)

    st.subheader("Daily Renewable Penetration")
    fig_pen = px.line(df_daily, x="Date", y="avg_renewable_penetration", title="Daily Renewable Penetration")
    st.plotly_chart(fig_pen, use_container_width=True)

    st.subheader("CO₂ vs Reliability Score")
    # If Reliability_Score present, show scatter
    if "Reliability_Score" in df.columns:
        # create a categorical class for coloring
        df['Reliability_Class'] = df['Reliability_Score'].apply(lambda x: "High" if x > 70 else "Medium" if x > 40 else "Low")
        fig_scatter = px.scatter(df, x="Carbon_Emission_kg", y="Reliability_Score", color="Reliability_Class",
                                 title="Emissions vs Reliability")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Reliability_Score not present in dataset; scatter plot is skipped.")

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

# ===================== TAB 5 – ML MODELS =====================
with tab_ml:
    st.subheader("ML: Forecasting & Reliability Classification")

    # Forecasting
    st.markdown("### 🔮 CO₂ Emissions Forecast")
    if PROPHET_AVAILABLE:
        try:
            forecast = forecast_co2(df, periods_days=int(forecast_periods))
            # show the forecasted yhat and historical y
            fig_fore = px.line(forecast, x='ds', y=['y', 'yhat', 'yhat_lower', 'yhat_upper'],
                               labels={'ds': 'Date', 'value': 'CO₂ (kg)'},
                               title=f"Forecasted Daily CO₂ (next {forecast_periods} days)")
            st.plotly_chart(fig_fore, use_container_width=True)
            st.write("### Forecast (tail)")
            st.dataframe(forecast[['ds', 'y', 'yhat']].tail(10))
        except Exception as e:
            st.error(f"Error during forecasting: {e}")
    else:
        st.info("Prophet not installed. Install with `pip install prophet` to enable forecasting.")

    st.markdown("---")

    # Reliability classifier
    st.markdown("### ⚙️ Reliability Classification (RandomForest)")
    if SKLEARN_AVAILABLE:
        try:
            model, report, feat_imp = train_reliability_classifier(df, n_estimators=rf_estimators)
            st.write("### Model performance (classification report)")
            st.json(report)

            st.write("### Feature importance")
            fig_imp = px.bar(feat_imp, x="importance", y="feature", orientation="h", title="Feature importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Error training classifier: {e}")
    else:
        st.info("Scikit-learn not installed. Install with `pip install scikit-learn` to enable classifier training.")

# ===================== TAB 6 – ADVANCED ANALYTICS =====================
with tab_adv:
    st.subheader("Advanced Analytics & Charts")

    # Correlation heatmap
    st.markdown("### 📊 Correlation Heatmap")
    try:
        fig_corr = plot_correlation_mat(df)
        st.pyplot(fig_corr)
    except Exception as e:
        st.error(f"Error plotting correlation heatmap: {e}")

    st.markdown("---")
    # Sankey
    st.markdown("### 🔀 Energy Flow (Sankey)")
    try:
        fig_sankey = energy_sankey(df)
        st.plotly_chart(fig_sankey, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating sankey: {e}")

    st.markdown("---")
    # CO2 intensity heatmap
    st.markdown("### 🔥 CO₂ Intensity Heatmap (Hour × Day)")
    try:
        fig_heat = co2_intensity_heatmap(df)
        st.pyplot(fig_heat)
    except Exception as e:
        st.error(f"Error plotting CO2 intensity heatmap: {e}")

    st.markdown("---")
    # Renewable vs Load density heatmap
    st.markdown("### 💠 Renewable vs Load Density")
    try:
        fig_density = px.density_heatmap(
            df,
            x="Renewable_Gen_kWh",
            y="Load_Demand_kWh",
            title="Renewable Generation vs Load Demand Density",
            nbinsx=50, nbinsy=50
        )
        st.plotly_chart(fig_density, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting renewable vs load density: {e}")

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.caption(
    "Tip: If ML sections are empty, install optional dependencies: "
    "`pip install prophet scikit-learn` and restart the app."
)

with st.expander("📥 Export processed data & utilities", expanded=False):
    st.write("You can download the cleaned & enriched dataset used by the dashboard.")
    try:
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed full dataset (CSV)", csv_all, "green_asset_processed.csv")
    except Exception as e:
        st.error(f"Could not prepare download: {e}")

    try:
        csv_daily = df_daily.to_csv(index=False).encode("utf-8")
        st.download_button("Download daily metrics (CSV)", csv_daily, "green_asset_daily_metrics.csv")
    except Exception as e:
        st.error(f"Could not prepare daily metrics download: {e}")

with st.expander("ℹ️ App Info & Next Steps", expanded=False):
    st.markdown(
        """
**What this app includes**
- KPI computation, daily aggregation, and data preprocessing.
- Visual insights: energy mix, CO₂ trends, renewable penetration, heatmaps.
- Rule-based sustainability recommendations.
- Optional ML modules:
  - CO₂ forecasting (Prophet)
  - Reliability classification (RandomForest)
- Advanced analytics:
  - Correlation heatmap
  - Energy flow Sankey diagram
  - CO₂ intensity heatmap (hour × day)
  - Renewable vs Load density map

---

### **To enable ML features**
Install optional dependencies in your environment:

```bash
pip install prophet scikit-learn







