# 🌱 Green Asset Monitor  
### *Industrial Carbon & Energy Sustainability Analytics Dashboard*

Green Asset Monitor is a data-driven sustainability analytics system built using **Python**, **Pandas**, **Plotly**, and **Streamlit**.  
It analyzes energy generation, demand, storage behavior, and carbon emissions in a **low-carbon industrial park**, providing actionable insights for sustainability teams and operations engineers.

This project uses the publicly available **Low-Carbon Industrial Park Energy Dataset** from Kaggle and demonstrates how industrial facilities can measure, monitor, and optimize their carbon footprint.

---

## 🚀 Project Overview

The goal of Green Asset Monitor is to:

- Compute **carbon emissions and CO₂ intensity**  
- Analyze **renewable energy penetration** (solar + wind)  
- Study **grid dependency** and **energy storage performance**  
- Visualize **energy flows, emissions, and sustainability KPIs**  
- Generate **automated recommendations** to improve operational efficiency  
- Provide a clean, interactive **Streamlit dashboard** for decision-making  

This project can be used as:
- A sustainability analytics prototype  
- A reference architecture for industrial carbon reporting  
- A portfolio project demonstrating data + cloud + sustainability expertise  

---

## 📦 Features

### 🔹 **1. Carbon Emission Insights**
- Total daily CO₂ emissions  
- Hourly & daily CO₂ intensity (kg/kWh)  
- Identification of high-emission periods  

### 🔹 **2. Renewable Energy Analysis**
- Solar + wind generation  
- Daily renewable penetration (%)  
- Comparison of renewable vs grid supply  

### 🔹 **3. Energy Mix Monitoring**
- Time-series breakdown:
  - Solar generation  
  - Wind generation  
  - Grid import  
  - Load demand  
  - Energy storage activity  

### 🔹 **4. Automated Sustainability Recommendations**
Rule-based insights such as:
- High-emission days  
- Low renewable penetration  
- High CO₂ intensity warnings  

### 🔹 **5. Interactive Streamlit Dashboard**
- KPI cards  
- Advanced charts (Plotly)  
- File upload support  
- Downloadable metrics table  

---

## 📊 Dataset Information

The dataset includes **hourly measurements** of:

| Feature | Description |
|--------|-------------|
| Solar_Gen_kWh | Solar generation (kWh) |
| Wind_Gen_kWh | Wind generation (kWh) |
| Grid_Import_kWh | Grid electricity imported |
| Load_Demand_kWh | Total energy demand |
| EV_Charging_kWh | Electric vehicle charging load |
| Storage_Input_kWh | Charging into battery storage |
| Storage_Output_kWh | Discharging from battery storage |
| Storage_Level_% | State of charge (%) |
| Carbon_Emission_kg | CO₂ emitted in that hour |
| Reliability_Score | System reliability (0–100) |

The dataset simulates a 30-day industrial park with mixed renewable supply, grid import, and storage systems.

---

## 🧮 Key Metrics Computed

- **Total CO₂ Emissions**  
- **Average CO₂ Intensity (kg/kWh)**  
- **Renewable Penetration (%)**  
- **Total Grid Import vs Renewable Generation**  
- **Storage Net Output (discharge–charge)**  

Daily summary includes:
- Total Renewables  
- Total Demand  
- Total Grid Import  
- Daily CO₂  
- Daily Renewable Penetration  
- Daily CO₂ Intensity  

---

## 📈 Visual Insights Provided

- Energy Mix Over Time (Stacked Area Chart)  
- Daily Carbon Emissions Trend  
- Daily Renewable Penetration Trend  
- CO₂ Intensity Distribution  

---

## 🧠 Automated Recommendations

Examples of generated insights:

- ⚠️ *High emissions on 2024-05-23. Consider shifting load to renewable peak hours.*  
- 🌥 *Low renewable penetration detected. Improve storage discharge strategy.*  
- 🔥 *High CO₂ intensity. Grid import may be from high-carbon sources.*  

These rules can be expanded with ML models in future versions.

---

