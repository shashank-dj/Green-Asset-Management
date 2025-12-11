import pandas as pd

def compute_sustainability_kpis(df):
    kpis = {
        "total_energy_demand": df["Load_Demand_kWh"].sum(),
        "total_renewable_gen": (df["Solar_Gen_kWh"] + df["Wind_Gen_kWh"]).sum(),
        "total_grid_import": df["Grid_Import_kWh"].sum(),
        "total_storage_net": (df["Storage_Output_kWh"] - df["Storage_Input_kWh"]).sum(),
        "total_co2": df["Carbon_Emission_kg"].sum(),
    }

    kpis["avg_co2_intensity"] = kpis["total_co2"] / kpis["total_energy_demand"]
    kpis["renewable_penetration"] = kpis["total_renewable_gen"] / kpis["total_energy_demand"]

    return kpis


def get_daily_insights(df):
    df_daily = df.groupby(df["Timestamp"].dt.date).agg(
        total_renewable=("Renewable_Gen_kWh", "sum"),
        total_grid=("Grid_Import_kWh", "sum"),
        total_demand=("Load_Demand_kWh", "sum"),
        total_co2=("Carbon_Emission_kg", "sum"),
        avg_renewable_penetration=("Renewable_Penetration", "mean"),
        avg_co2_intensity=("CO2_Intensity", "mean"),
    ).reset_index().rename(columns={"Timestamp": "Date"})

    return df_daily


def generate_recommendations(df, df_daily):
    recs = []

    # Rule 1: High CO2
    high = df_daily[df_daily["total_co2"] > df_daily["total_co2"].mean() * 1.2]
    for _, row in high.iterrows():
        recs.append(
            f"⚠️ High emissions on {row['Date']}. Consider shifting load to renewable generation hours."
        )

    # Rule 2: Low renewable penetration
    low_ren = df_daily[df_daily["avg_renewable_penetration"] < 0.40]
    for _, row in low_ren.iterrows():
        recs.append(
            f"🌥 Low renewable penetration on {row['Date']}. Improve storage discharge strategies."
        )

    # Rule 3: High CO₂ intensity
    high_int = df_daily[df_daily["avg_co2_intensity"] > df_daily["avg_co2_intensity"].mean() * 1.25]
    for _, row in high_int.iterrows():
        recs.append(
            f"🔥 High CO₂ intensity on {row['Date']}. Grid import may be from high-emission sources."
        )

    return recs
