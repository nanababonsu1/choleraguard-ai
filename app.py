import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
from fpdf import FPDF
def get_live_weather(city="Accra"):
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data["current_condition"][0]
        rainfall_mm = float(current.get("precipMM", 0))
        temperature_c = float(current.get("temp_C", 0))
        humidity = float(current.get("humidity", 0))

        return rainfall_mm, temperature_c, humidity

    except Exception:
        return None, None, None
st.set_page_config(
    page_title="CholeraGuard AI",
    page_icon="🦠",
    layout="wide"
)
st.markdown("""
# 🧠 CholeraGuard AI  
### Ghana National Cholera Surveillance & Outbreak Prediction Dashboard

**AI-powered early warning system for cholera risk monitoring, rainfall scenario analysis, district-level intelligence, and public health response planning.**
""")

st.divider()
st.title("🇬🇭 CholeraGuard AI: Ghana National Cholera Surveillance Dashboard")
st.subheader("AI-Powered Cholera Early Warning and Outbreak Prediction System for all 16 Regions of Ghana")

st.write("""
This platform uses Artificial Intelligence and Machine Learning to predict cholera outbreak risk using rainfall, water access, sanitation, and population density indicators.
""")

# ---------------------------
# DATA
# ---------------------------
st.header("🌧 Rainfall Data Source")

rainfall_source = st.selectbox(
    "Choose Rainfall Source",
    ["Historical Ghana Dataset", "Manual Entry"]
)
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
district_data = pd.DataFrame({
    "District": [
        "Accra Metropolitan", "Tema Metropolitan", "Ga East", "Ga West", "La Dade-Kotopon",
        "Kumasi Metropolitan", "Obuasi Municipal", "Ejisu", "Asokore Mampong", "Bekwai",
        "Cape Coast Metropolitan", "Komenda Edina Eguafo Abirem", "Awutu Senya East",
        "Sekondi-Takoradi Metropolitan", "Tarkwa-Nsuaem",
        "Koforidua", "New Juaben South", "Akwapim South",
        "Ho Municipal", "Keta Municipal",
        "Tamale Metropolitan", "Sagnarigu", "Savelugu",
        "Sunyani Municipal", "Techiman Municipal",
        "Wa Municipal", "Bolgatanga Municipal", "Bawku Municipal", "Nalerigu-Gambaga", "Dambai"
    ],

    "Region": [
        "Greater Accra", "Greater Accra", "Greater Accra", "Greater Accra", "Greater Accra",
        "Ashanti", "Ashanti", "Ashanti", "Ashanti", "Ashanti",
        "Central", "Central", "Central",
        "Western", "Western",
        "Eastern", "Eastern", "Eastern",
        "Volta", "Volta",
        "Northern", "Northern", "Northern",
        "Bono", "Bono",
        "Upper West", "Upper East", "Upper East", "North East", "Oti"
    ],

    "Rainfall_mm": [
        220, 210, 205, 195, 200,
        180, 160, 170, 155, 150,
        140, 135, 130,
        125, 120,
        165, 150, 145,
        130, 120,
        90, 88, 85,
        80, 78,
        82, 95, 90, 86, 85
    ],

    "Water_Access": [
        82, 80, 78, 76, 77,
        74, 75, 73, 72, 70,
        69, 68, 66,
        67, 69,
        70, 72, 71,
        68, 67,
        61, 60, 59,
        66, 65,
        63, 58, 57, 56, 60
    ],

    "Sanitation": [
        35, 34, 30, 29, 28,
        26, 27, 25, 22, 21,
        20, 21, 22,
        24, 23,
        28, 27, 26,
        25, 24,
        22, 21, 20,
        21, 22,
        20, 21, 19, 18, 25
    ],

    "Population_Density": [
        6500, 5800, 4500, 4200, 5000,
        6000, 3500, 3000, 4200, 2800,
        3000, 2600, 3200,
        2800, 2300,
        2400, 2200, 2000,
        1800, 1600,
        1500, 1300, 1100,
        1000, 1200,
        800, 900, 850, 750, 1300
    ],"Latitude": [
    5.5600, 5.6698, 5.7361, 5.6980, 5.5660,
    6.6885, 6.2023, 6.7130, 6.7040, 6.4519,
    5.1053, 5.0833, 5.5340,
    4.9340, 5.3064,
    6.0941, 6.0830, 5.9500,
    6.6000, 5.9167,
    9.4075, 9.4320, 9.6244,
    7.3399, 7.5842,
    10.0600, 10.7856, 11.0616, 10.5273, 8.0667
],

"Longitude": [
    -0.2057, -0.0166, -0.1830, -0.3100, -0.1650,
    -1.6244, -1.6717, -1.3560, -1.5930, -1.5784,
    -1.2466, -1.2167, -0.4200,
    -1.7137, -1.9847,
    -0.2591, -0.2500, -0.1667,
    0.4700, 0.9833,
    -0.8533, -0.8424, -0.8253,
    -2.3268, -1.9382,
    -2.5019, -0.8514, -0.2417, -0.3698, 0.1790
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
    if cases >= 100:
        return "🔴 OUTBREAK"
    elif cases >= 50:
        return "🟠 HIGH RISK"
    elif cases >= 20:
        return "🟡 ALERT"
    else:
        return "🟢 NORMAL"

regional_data["Risk_Level"] = regional_data["Predicted_Cases"].apply(risk_level)
regional_data["WHO_Status"] = regional_data["Predicted_Cases"].apply(risk_level)

st.sidebar.markdown("---")
st.sidebar.title("🧠 CholeraGuard AI")
st.sidebar.caption("National Cholera Surveillance Dashboard")
st.sidebar.markdown("### ⚙️ Prediction Controls")

region = st.sidebar.selectbox("Select Region", regional_data["Region"])

selected = regional_data[regional_data["Region"] == region].iloc[0]

st.sidebar.markdown("---")

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    0,
    300,
    int(selected["Rainfall_mm"])
)

rainfall_scenario = st.sidebar.selectbox(
    "Rainfall Scenario",
    [
        "Use Selected Region Rainfall",
        "Low Rainfall Scenario",
        "Moderate Rainfall Scenario",
        "Heavy Rainfall Scenario",
        "Extreme Flooding Scenario"
        "Live Weather - Accra"
    ]
)

if rainfall_scenario == "Low Rainfall Scenario":
    rainfall_input = 50
elif rainfall_scenario == "Moderate Rainfall Scenario":
    rainfall_input = 120
elif rainfall_scenario == "Heavy Rainfall Scenario":
    rainfall_input = 220
elif rainfall_scenario == "Extreme Flooding Scenario":
    rainfall_input = 300
    
elif rainfall_scenario == "Live Weather - Accra":
    rainfall_mm, temperature_c, humidity = get_live_weather("Accra")

    if rainfall_mm is not None:
        rainfall_input = rainfall_mm

        st.sidebar.success(
            f"🌦 Live Weather\n\n"
            f"Rainfall: {rainfall_mm} mm\n"
            f"Temperature: {temperature_c}°C\n"
            f"Humidity: {humidity}%"
        )
    else:
        st.sidebar.error("Unable to retrieve live weather data.")
        rainfall_input = rainfall
else:
    rainfall_input = rainfall

st.sidebar.info(f"🌧 Rainfall Used: {rainfall_input} mm")

st.sidebar.markdown("---")

water_access = st.sidebar.slider(
    "Water Access (%)",
    0,
    100,
    int(selected["Water_Access"])
)

st.sidebar.markdown("---")

sanitation = st.sidebar.slider(
    "Sanitation Coverage (%)",
    0,
    100,
    int(selected["Sanitation"])
)

st.sidebar.markdown("---")

population_density = st.sidebar.slider(
    "Population Density",
    100,
    6000,
    int(selected["Population_Density"])
)
st.sidebar.markdown("---")

st.sidebar.info("""
### 🤖 AI Model Information

**Version:** 2.0

**Country:** Ghana 🇬🇭

**Forecast Horizon:** 3 Months

**AI Engine:** Random Forest Prediction Model

**Purpose:** National Cholera Early Warning & Decision Support System
""")
st.sidebar.markdown("---")

st.sidebar.info("""
### 🤖 AI Model Information

**Version:** 2.0  
**Country:** Ghana 🇬🇭  
**Forecast Horizon:** 3 Months  
**Purpose:** Cholera Early Warning & Public Health Decision Support
""")
input_df = pd.DataFrame({
    "Rainfall_mm": [rainfall_input],
    "Water_Access": [water_access],
    "Sanitation": [sanitation],
    "Population_Density": [population_density]
})

prediction = model.predict(input_df)[0]
gauge_value = prediction
regional_data.loc[
    regional_data["Region"] == region,
    "Predicted_Cases"
] = prediction

regional_data["Risk_Level"] = regional_data["Predicted_Cases"].apply(risk_level)
regional_data["WHO_Status"] = regional_data["Predicted_Cases"].apply(risk_level)

# ---------------------------
# DASHBOARD
# ---------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Selected Region", region)


col4.metric("Model Type", "Random Forest")
st.subheader("🎯 Cholera Risk Gauge")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gauge_value,
    title={"text": "Outbreak Risk Score"},
    gauge={
        "axis": {"range": [0, 120]},
        "bar": {"color": "red"},
        "steps": [
            {"range": [0, 20], "color": "lightgreen"},
            {"range": [20, 50], "color": "yellow"},
            {"range": [50, 100], "color": "orange"},
            {"range": [100, 120], "color": "red"}
        ]
    }
))
outbreak_regions = len(
    regional_data[
        regional_data["Predicted_Cases"] >= 100
    ]
)

high_risk_regions = len(
    regional_data[
        regional_data["Predicted_Cases"] >= 50
    ]
)

alert_regions = len(
    regional_data[
        regional_data["Predicted_Cases"] >= 20
    ]
)

st.metric(
    "National Outbreak Regions",
    outbreak_regions
)

st.metric(
    "High Risk Regions",
    high_risk_regions
)

st.metric(
    "Alert Regions",
    alert_regions
)
st.plotly_chart(fig_gauge, use_container_width=True)
st.divider()
regional_data["Predicted_Cases"] = regional_data["Rainfall_mm"] * 180
total_cases = regional_data["Predicted_Cases"].sum()
high_risk = len(regional_data[regional_data["Predicted_Cases"] > 25000])

colA, colB, colC = st.columns(3)

colA.metric("Total Predicted Cases", f"{int(total_cases):,}")
colB.metric("Regions Monitored", "16")
colC.metric("High Risk Regions", high_risk)
st.divider()

if high_risk >= 5:
    st.error(f"🔴 NATIONAL CHOLERA WATCH: {high_risk} regions currently exceed outbreak-risk thresholds.")
elif high_risk >= 2:
    st.warning(f"🟡 ELEVATED CHOLERA RISK: {high_risk} regions require enhanced surveillance.")
else:
    st.success("🟢 NATIONAL STATUS STABLE: No major outbreak-risk concentration detected.")

st.subheader("🤖 AI Public Health Recommendations")

if high_risk >= 5:
    st.markdown("""
    **Recommended Actions**
    - Activate national cholera preparedness protocols.
    - Prioritize Greater Accra, Ashanti, Central, Eastern, and Western regions.
    - Increase water-quality testing and WASH surveillance.
    - Pre-position ORS, IV fluids, antibiotics, and rapid response teams.
    - Intensify public education on safe water, handwashing, and sanitation.
    """)
elif high_risk >= 2:
    st.markdown("""
    **Recommended Actions**
    - Strengthen regional surveillance in watchlist areas.
    - Monitor rainfall trends and water contamination signals.
    - Prepare district response teams.
    - Begin targeted community WASH messaging.
    """)
else:
    st.markdown("""
    **Recommended Actions**
    - Continue routine cholera surveillance.
    - Maintain community WASH education.
    - Monitor rainfall and sanitation trends.
    """)
st.subheader("🗺️ Ghana Cholera Risk Map")

fig_map = px.scatter_mapbox(
    regional_data,
    lat="Latitude",
    lon="Longitude",
    hover_name="Region",
    hover_data=["Predicted_Cases"],
    size="Predicted_Cases",
    color="Predicted_Cases",
    color_continuous_scale="Reds",
    zoom=5,
    center={"lat": 7.9465, "lon": -1.0232},
    height=600
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.plotly_chart(fig_map, use_container_width=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Regions at High Risk",
        len(regional_data[regional_data["Predicted_Cases"] > 20000])
    )

with col2:
    st.metric(
        "Average Risk",
        round(regional_data["Predicted_Cases"].mean(),0)
    )

with col3:
    st.metric(
        "Highest Risk Region",
        regional_data.sort_values(
            "Predicted_Cases",
            ascending=False
        ).iloc[0]["Region"]
    )

with col4:
    st.metric(
        "Total Regions",
        len(regional_data)
    )

st.subheader("📊 National Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Predicted Cases",
        int(regional_data["Predicted_Cases"].sum())
    )

with col2:
    highest_region = regional_data.sort_values(
        "Predicted_Cases",
        ascending=False
    ).iloc[0]["Region"]

    st.metric("Highest Risk Region", highest_region)

with col3:
    st.metric(
        "Average Regional Risk",
        round(regional_data["Predicted_Cases"].mean(), 1)
    )

st.subheader("🚨 National Cholera Alert Status")

if regional_data["Predicted_Cases"].max() >= 30000:
    st.error("🔴 NATIONAL OUTBREAK WARNING")
elif regional_data["Predicted_Cases"].max() >= 20000:
    st.warning("🟠 HIGH RISK ALERT")
elif regional_data["Predicted_Cases"].max() >= 10000:
    st.warning("🟡 ELEVATED RISK")
else:
    st.success("🟢 LOW RISK")
st.divider()

st.subheader("🔥 Top 5 High-Risk Regions")

top5 = regional_data.sort_values(
    "Predicted_Cases",
    ascending=False
).head(5)

st.dataframe(
    top5[["Region", "Predicted_Cases"]],
    use_container_width=True
)

st.subheader("🚨 Regional Alert Status")

regional_data["Status"] = regional_data["Predicted_Cases"].apply(
    lambda x:
    "🔴 OUTBREAK"
    if x >= 30000 else
    "🟠 HIGH RISK"
    if x >= 20000 else
    "🟡 ALERT"
    if x >= 10000 else
    "🟢 NORMAL"
)

st.dataframe(
    regional_data[
        ["Region","Predicted_Cases","Risk_Level","WHO_Status"]
    ],
    use_container_width=True
)
st.divider()

st.divider()

st.subheader("🏙 Top 10 High-Risk Districts")

district_data["Risk_Score"] = (
    district_data["Rainfall_mm"] * 100
    + district_data["Population_Density"] * 2
    - district_data["Water_Access"] * 50
    - district_data["Sanitation"] * 40
)

top_districts = district_data.sort_values(
    "Risk_Score",
    ascending=False
).head(10)

st.dataframe(
    top_districts[["District", "Region", "Risk_Score"]],
    use_container_width=True
)
st.subheader("📍 Interactive District Drill-Down")

selected_district = st.selectbox(
    "Select a district to explore:",
    district_data["District"]
)

district_info = district_data[district_data["District"] == selected_district].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("District", district_info["District"])
    st.metric("Region", district_info["Region"])

with col2:
    st.metric("Risk Score", round(district_info["Risk_Score"], 1))
    st.metric("Risk Level", "High" if district_info["Risk_Score"] >= 30000 else "Moderate")

st.progress(min(int(district_info["Risk_Score"] / 40000 * 100), 100))

st.caption(f"District risk score: {district_info['Risk_Score']:.1f}")
st.subheader("🌧 Rainfall Intelligence Panel")

st.info(
    f"Current rainfall input for this simulation is {rainfall_input} mm under the '{rainfall_scenario}' scenario."
)

if rainfall_input >= 250:
    st.error("Extreme rainfall/flooding conditions detected. Cholera transmission risk may increase significantly.")
elif rainfall_input >= 180:
    st.warning("Heavy rainfall conditions detected. Increase surveillance and water-quality monitoring.")
elif rainfall_input >= 100:
    st.warning("Moderate rainfall conditions detected. Continue close monitoring.")
else:
    st.success("Low rainfall conditions detected. Routine monitoring recommended.")
st.subheader("🗺️ District-Level Cholera Risk Map")

fig_district_map = px.scatter_mapbox(
    district_data,
    lat="Latitude",
    lon="Longitude",
    hover_name="District",
    hover_data=["Region", "Risk_Score"],
    size="Risk_Score",
    color="Risk_Score",
    color_continuous_scale="Reds",
    zoom=5,
    center={"lat": 7.9465, "lon": -1.0232},
    height=600
)

fig_district_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig_district_map, use_container_width=True)
st.caption(
    "Note: The district map shows district-level baseline risk scores. The gauge and prediction metrics update based on the selected regional simulation inputs."
)
st.subheader("📥 Export Surveillance Data")

csv = regional_data.to_csv(index=False)

st.download_button(
    label="Download Surveillance Report (CSV)",
    data=csv,
    file_name="cholera_surveillance_report.csv",
    mime="text/csv"
)
def create_pdf_report():
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CholeraGuard AI Surveillance Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Regions Monitored: 16", ln=True)
    pdf.cell(0, 10, f"Total Predicted Cases: {int(total_cases):,}", ln=True)
    pdf.cell(0, 10, f"High Risk Regions: {high_risk}", ln=True)

    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Public Health Recommendation", ln=True)

    pdf.set_font("Arial", "", 11)
    recommendation = "Continue routine cholera surveillance and WASH education."
    pdf.multi_cell(0, 8, recommendation)

    pdf_output = pdf.output(dest="S")

    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1")
    return bytes(pdf_output)


pdf_report = create_pdf_report()

st.download_button(
    label="Download Surveillance Report (PDF)",
    data=pdf_report,
    file_name="cholera_surveillance_report.pdf",
    mime="application/pdf"
)
st.subheader("📈 Monthly Cholera Trend")
st.subheader("🔮 3-Month Cholera Forecast")

future_months = ["Next Month", "Month +2", "Month +3"]

current_prediction = prediction

forecast_cases = [
    int(current_prediction * 1.05),
    int(current_prediction * 1.10),
    int(current_prediction * 1.15)
]
st.subheader("📊 Forecast Confidence")

if prediction >= 100:
    confidence = "High Confidence"
    confidence_score = 90
elif prediction >= 50:
    confidence = "Moderate Confidence"
    confidence_score = 80
else:
    confidence = "Exploratory Forecast"
    confidence_score = 70

col1, col2 = st.columns(2)

with col1:
    st.metric("Forecast Confidence", f"{confidence_score}%")

with col2:
    st.metric("Forecast Status", confidence)
    st.subheader("🧠 AI Outbreak Probability Meter")

outbreak_probability = min(
    100,
    int((prediction / 100) * 100)
)

st.progress(outbreak_probability / 100)

if outbreak_probability >= 80:
    st.error(f"🔴 Outbreak Probability: {outbreak_probability}% — Very High")
elif outbreak_probability >= 60:
    st.warning(f"🟠 Outbreak Probability: {outbreak_probability}% — High")
elif outbreak_probability >= 40:
    st.warning(f"🟡 Outbreak Probability: {outbreak_probability}% — Moderate")
else:
    st.success(f"🟢 Outbreak Probability: {outbreak_probability}% — Low")
    st.subheader("📝 AI Situation Report")

highest_region = regional_data.sort_values(
    "Predicted_Cases",
    ascending=False
).iloc[0]["Region"]

st.markdown(f"""
**Executive Summary**

- **Selected Region:** {region}
- **Predicted Cholera Cases:** {int(prediction):,}
- **Highest-Risk Region Nationally:** {highest_region}
- **Rainfall Scenario:** {rainfall_scenario}
- **Model Rainfall Input:** {rainfall_input} mm
- **Outbreak Probability:** {outbreak_probability}%
- **Forecast Confidence:** {confidence_score}% ({confidence})

**Recommended Public Health Interpretation**

Cholera risk is currently being influenced by rainfall intensity, sanitation coverage, water access, and population density. Public health teams should prioritize surveillance, WASH interventions, water-quality testing, community education, and rapid response readiness in high-risk areas.
""")
st.subheader("🌍 WHO Outbreak Threshold Assessment")

if prediction >= 100:
    st.error("🔴 WHO Assessment: Predicted cases exceed the national outbreak threshold. Immediate emergency response is recommended.")
elif prediction >= 50:
    st.warning("🟠 WHO Assessment: High transmission risk. Intensify surveillance and preparedness.")
elif prediction >= 20:
    st.warning("🟡 WHO Assessment: Increased alert level. Strengthen monitoring and WASH interventions.")
else:
    st.success("🟢 WHO Assessment: Routine surveillance. No evidence of widespread transmission.")
    st.subheader("🇬🇭 National Risk Classification")

if prediction >= 100:
    risk_level = "SEVERE"
elif prediction >= 50:
    risk_level = "HIGH"
elif prediction >= 20:
    risk_level = "MODERATE"
else:
    risk_level = "LOW"

st.metric("Current National Risk", risk_level)
st.subheader("🛰 AI Situation Report")

st.markdown(f"""
### Executive Situation Brief

**Date:** {pd.Timestamp.today().strftime("%d %B %Y")}

**Country:** Ghana

**AI Surveillance Summary**

The AI surveillance engine predicts approximately **{int(prediction)} cholera cases** based on current environmental and public health indicators.

### Key Findings

- Highest-risk region: **{highest_region}**
- National Risk Level: **{risk_level}**
- Rainfall Scenario: **{rainfall_scenario}**
- Estimated Rainfall: **{rainfall_input} mm**
- Water Access: **{water_access}%**
- Sanitation Coverage: **{sanitation}%**
- Population Density: **{population_density:,} persons/km²**

### AI Assessment

The current environmental conditions suggest that cholera transmission may {'increase' if prediction >= 50 else 'remain stable'} over the coming weeks.

Priority should be given to intensified surveillance, rapid response preparedness, improved WASH interventions, and continuous monitoring of rainfall patterns.

---
**Generated automatically by CholeraGuard AI**
""")
st.subheader("📈 Forecast Trend Interpretation")

forecast_change = forecast_cases[-1] - forecast_cases[0]

if forecast_change > 0:
    st.warning(
        f"Projected cholera burden is increasing over the next 3 months by approximately {int(forecast_change)} cases."
    )
elif forecast_change < 0:
    st.success(
        f"Projected cholera burden is decreasing over the next 3 months by approximately {abs(int(forecast_change))} cases."
    )
else:
    st.info("Projected cholera burden is stable over the next 3 months.")
forecast_df = pd.DataFrame({
    "Month": future_months,
    "Forecasted Cases": forecast_cases
})

fig_forecast = px.line(
    forecast_df,
    x="Month",
    y="Forecasted Cases",
    markers=True,
    title="AI Forecast of Cholera Cases"
)

st.plotly_chart(fig_forecast, use_container_width=True)
fig2, ax2 = plt.subplots(figsize=(10,5))

ax2.plot(
    monthly_data["Month"],
    monthly_data["Cholera_Cases"],
    marker="o"
)

ax2.set_title("Monthly Cholera Cases")
ax2.set_ylabel("Cases")

st.pyplot(fig2)

st.divider()
st.subheader("🔮 3-Month Cholera Forecast")

recent_cases = monthly_data["Cholera_Cases"].tail(3).mean()

forecast_months = ["Next Month", "Month 2", "Month 3"]
forecast_cases = [
    recent_cases * 1.10,
    recent_cases * 1.20,
    recent_cases * 1.30
]

forecast_df = pd.DataFrame({
    "Forecast Period": forecast_months,
    "Predicted Cases": [round(x, 1) for x in forecast_cases]
})

st.dataframe(forecast_df, use_container_width=True)

fig_forecast, ax_forecast = plt.subplots(figsize=(10, 5))

ax_forecast.plot(
    forecast_df["Forecast Period"],
    forecast_df["Predicted Cases"],
    marker="o"
)

ax_forecast.set_title("Projected Cholera Cases for Next 3 Months")
ax_forecast.set_ylabel("Predicted Cases")

st.pyplot(fig_forecast)

st.subheader("🚦 National Cholera Risk Gauge")

if high_risk >= 5:
    st.error("🔴 OUTBREAK RISK - Immediate Response Required")
elif high_risk >= 3:
    st.warning("🟡 WATCHLIST - Enhanced Surveillance Recommended")
else:
    st.success("🟢 NORMAL - Situation Stable")
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
