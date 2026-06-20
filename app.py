import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="CholeraGuard AI",
    page_icon="🦠",
    layout="wide"
)

st.title("🇬🇭 CholeraGuard AI: Ghana 16-Region Early Warning System")
st.subheader("AI-Powered Cholera Surveillance and Outbreak Prediction Platform for all 16 Regions of Ghana")

st.write("""
This platform uses Artificial Intelligence and Machine Learning to predict cholera outbreak risk using rainfall, water access, sanitation, and population density indicators.
""")

# ---------------------------
# DATA
# ---------------------------

monthly_data = pd.DataFrame({
    "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "Rainfall_mm": [30.27,31.73,152.81,216.25,191.41,221.95,257.42,68.67,142.59,166.92,174.68,22.69],
    "Cholera_Cases": [12,15,22,41,58,73,85,49,38,27,20,14],
    "Water_Access": [85,85,84,84,83,83,82,82,82,81,81,81],
    "Sanitation": [25,25,26,26,27,27,28,28,29,29,30,30],
    "Population_Density": [1300,1300,1350,1400,1450,1500,1550,1550,1500,1450,1400,1350]
})

regional_data = pd.DataFrame({
    "Region": [
        "Greater Accra", "Ashanti", "Central", "Eastern",
        "Western", "Western North", "Volta", "Oti",
        "Northern", "Savannah", "North East", "Upper East",
        "Upper West", "Bono", "Bono East", "Ahafo"
    ],
    "Rainfall_mm": [
        220, 210, 165, 150, 140, 135, 130, 125,
        90, 88, 86, 85, 80, 78, 82, 95
    ],
    "Water_Access": [
        82, 80, 78, 76, 77, 74, 75, 73,
        70, 69, 68, 68, 67, 69, 70, 72
    ],
    "Sanitation": [
        35, 34, 30, 29, 28, 26, 27, 25,
        22, 21, 20, 21, 20, 21, 22, 24
    ],
    "Population_Density": [
        5000, 4200, 2600, 2400, 2100, 1500, 1800, 1300,
        1200, 900, 850, 900, 800, 850, 950, 1000
    ],
    "Latitude": [
5.6037,
6.6885,
5.1053,
6.0910,
4.8966,
10.0600,
6.6000,
8.5500,
9.4000,
9.8250,
10.7080,
10.7850,
7.7340,
7.9500,
6.8000,
5.9500
],

"Longitude": [
-0.1870,
-1.6244,
-1.2795,
-0.2591,
-1.7831,
-2.5000,
0.0000,
0.2500,
-1.0000,
-0.3000,
-0.9820,
-2.4850,
-2.1040,
-1.6000,
-2.5000,
-0.9000
]
})

X = monthly_data[["Rainfall_mm","Water_Access","Sanitation","Population_Density"]]
y = monthly_data["Cholera_Cases"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

regional_data["Predicted_Cases"] = model.predict(
    regional_data[["Rainfall_mm","Water_Access","Sanitation","Population_Density"]]
)

def risk_level(cases):
    if cases >= 60:
        return "🔴 OUTBREAK ALERT"
    elif cases >= 30:
        return "🟡 WATCHLIST"
    else:
        return "🟢 STABLE"

regional_data["Risk_Level"] = regional_data["Predicted_Cases"].apply(risk_level)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.header("Run a Regional Prediction")

region = st.sidebar.selectbox("Select Region", regional_data["Region"])

selected = regional_data[regional_data["Region"] == region].iloc[0]

rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, int(selected["Rainfall_mm"]))
water_access = st.sidebar.slider("Water Access (%)", 0, 100, int(selected["Water_Access"]))
sanitation = st.sidebar.slider("Sanitation Coverage (%)", 0, 100, int(selected["Sanitation"]))
population_density = st.sidebar.slider("Population Density", 100, 6000, int(selected["Population_Density"]))

input_df = pd.DataFrame({
    "Rainfall_mm": [rainfall],
    "Water_Access": [water_access],
    "Sanitation": [sanitation],
    "Population_Density": [population_density]
})

prediction = model.predict(input_df)[0]
risk = risk_level(prediction)

# ---------------------------
# DASHBOARD
# ---------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected Region", region)
col2.metric("Predicted Cases", round(prediction, 1))
col3.metric("Risk Level", risk)
col4.metric("Model Type", "Random Forest")

st.divider()

st.header("📊 National Cholera Surveillance Dashboard")

top_regions = regional_data.sort_values("Predicted_Cases", ascending=False)

st.subheader("Regional Risk Ranking")
st.dataframe(top_regions)

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(top_regions["Region"], top_regions["Predicted_Cases"])
ax1.axhline(60, linestyle="--")
ax1.set_title("Predicted Cholera Risk by Region")
ax1.set_ylabel("Predicted Cases")
ax1.set_xticklabels(top_regions["Region"], rotation=45, ha="right")
st.pyplot(fig1)

st.divider()

st.header("📈 Monthly Cholera and Rainfall Trends")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(monthly_data["Month"], monthly_data["Cholera_Cases"], marker="o", label="Cholera Cases")
ax2.plot(monthly_data["Month"], monthly_data["Rainfall_mm"] / 3, marker="s", label="Rainfall Scaled")
ax2.set_title("Monthly Cholera Cases and Rainfall Pattern")
ax2.set_ylabel("Cases / Scaled Rainfall")
ax2.legend()
st.pyplot(fig2)

st.divider()

st.header("🧠 Factors Driving Cholera Risk")

importance = pd.DataFrame({
    "Factor": ["Rainfall", "Water Access", "Sanitation", "Population Density"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.dataframe(importance)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.bar(importance["Factor"], importance["Importance"])
ax3.set_title("Feature Importance Analysis")
ax3.set_ylabel("Importance Score")
st.pyplot(fig3)

st.divider()

st.header("🚨 Public Health Recommendation")

if prediction >= 60:
    st.error("""
    HIGH OUTBREAK RISK DETECTED.

    Recommended actions:
    - Activate emergency cholera surveillance.
    - Pre-position oral rehydration salts and IV fluids.
    - Intensify water quality testing.
    - Launch community WASH education.
    - Notify regional public health authorities.
    """)
elif prediction >= 30:
    st.warning("""
    MODERATE RISK DETECTED.

    Recommended actions:
    - Increase community surveillance.
    - Monitor rainfall and water contamination.
    - Prepare response teams.
    - Strengthen WASH messaging.
    """)
else:
    st.success("""
    LOW CURRENT RISK.

    Recommended actions:
    - Continue routine surveillance.
    - Maintain WASH education.
    - Monitor rainfall trends.
    """)

st.divider()

st.caption("CholeraGuard AI v2 | Developed by Lovelace Osei Bonsu | Prototype for public health surveillance and hackathon demonstration.")
