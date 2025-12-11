import pandas as pd
import numpy as np

# ==========================================
# 1. KPI CALCULATION
# ==========================================
def compute_sustainability_kpis(df):
    """Compute global sustainability KPIs used in the dashboard."""

    total_energy = df["Load_Demand_kWh"].sum()
    total_renew = (df["Solar_Gen_kWh"] + df["Wind_Gen_kWh"]).sum()
    total_grid = df["Grid_Import_kWh"].sum()
    total_storage = (df["Storage_Output_kWh"] - df["Storage_Input_kWh"]).sum()
    total_co2 = df["Carbon_Emission_kg"].sum()

    avg_co2_intensity = total_co2 / total_energy if total_energy > 0 else 0
    renewable_penetration = total_renew / total_energy if total_energy > 0 else 0

    return {
        "total_energy_demand": total_energy,
        "total_renewable_gen": total_renew,
        "total_grid_import": total_grid,
        "total_storage_net": total_storage,
        "total_co2": total_co2,
        "avg_co2_intensity": avg_co2_intensity,
        "renewable_penetration": renewable_penetration,
    }


# ==========================================
# 2. DAILY KPI AGGREGATION
# ==========================================
def get_daily_insights(df):
    """Aggregate sustainability metrics by day."""

    df_daily = (
        df.groupby(df["Timestamp"].dt.date)
        .agg(
            total_renewable=("Renewable_Gen_kWh", "sum"),
            total_grid=("Grid_Import_kWh", "sum"),
            total_demand=("Load_Demand_kWh", "sum"),
            total_co2=("Carbon_Emission_kg", "sum"),
            avg_renewable_penetration=("Renewable_Penetration", "mean"),
            avg_co2_intensity=("CO2_Intensity", "mean"),
        )
        .reset_index()
        .rename(columns={"Timestamp": "Date"})
    )

    return df_daily


# ==========================================
# 3. RULE-BASED RECOMMENDATIONS ENGINE
# ==========================================
def generate_recommendations(df, df_daily):
    """Generate sustainability recommendations using rule-based logic."""

    recs = []

    # Rule 1 — High emissions
    threshold_high_co2 = df_daily["total_co2"].mean() * 1.2
    high_emissions_days = df_daily[df_daily["total_co2"] > threshold_high_co2]

    for _, row in high_emissions_days.iterrows():
        recs.append(
            f"⚠️ High emissions on {row['Date']}. Consider shifting load to hours of high renewable generation."
        )

    # Rule 2 — Low renewable penetration
    low_renewable_days = df_daily[df_daily["avg_renewable_penetration"] < 0.40]

    for _, row in low_renewable_days.iteritems():
        recs.append(
            f"🌥 Low renewable penetration on {row['Date']}. Improve storage discharge strategy or shift flexible loads."
        )

    # Rule 3 — CO₂ intensity spikes
    co2_intensity_limit = df_daily["avg_co2_intensity"].mean() * 1.25
    high_intensity_days = df_daily[df_daily["avg_co2_intensity"] > co2_intensity_limit]

    for _, row in high_intensity_days.iterrows():
        recs.append(
            f"🔥 High CO₂ intensity on {row['Date']}. Grid import may be from high-emission sources."
        )

    return recs


# ==========================================
# 4. OPTIONAL — CO₂ FORECASTING (Prophet)
# ==========================================
def forecast_co2_prophet(df, periods_days=7):
    """
    Uses Prophet to forecast CO₂ emissions.
    This function is intentionally import-lazy to avoid failures on Streamlit Cloud.
    """
    try:
        from prophet import Prophet
    except Exception:
        raise RuntimeError("Prophet not installed. Install with: pip install prophet")

    # Prepare daily aggregated CO₂ data
    df_prophet = (
        df.groupby(df["Timestamp"].dt.date)
        .agg(daily_co2=("Carbon_Emission_kg", "sum"))
        .reset_index()
    )

    df_prophet.columns = ["ds", "y"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods_days)
    forecast = model.predict(future)

    return forecast


# ==========================================
# 5. OPTIONAL — Reliability Classification (RandomForest)
# ==========================================
def train_reliability_classifier(df, n_estimators=150):
    """
    Train a RandomForest classifier to predict High/Medium/Low reliability.
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
    except Exception:
        raise RuntimeError("scikit-learn not installed. Install with: pip install scikit-learn")

    df_local = df.copy()
    if "Reliability_Score" not in df_local.columns:
        raise RuntimeError("Dataset does not contain 'Reliability_Score' column.")

    df_local["Reliability_Class"] = df_local["Reliability_Score"].apply(
        lambda x: 2 if x > 70 else 1 if x > 40 else 0
    )

    features = [
        "Solar_Gen_kWh",
        "Wind_Gen_kWh",
        "Grid_Import_kWh",
        "Load_Demand_kWh",
        "Storage_Input_kWh",
        "Storage_Output_kWh",
        "Carbon_Emission_kg",
    ]

    X = df_local[features].fillna(0.0)
    y = df_local["Reliability_Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    feature_importance_df = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return model, report, feature_importance_df


# ==========================================
# 6. OPTIONAL — Advanced Analytics Helpers
# ==========================================
def safe_minmax(value):
    """Utility function to handle safe min/max scaling."""
    if value is None or value == 0:
        return 0
    return value

