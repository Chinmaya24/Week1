# ============================================
# EV Price Prediction using Random Forest
# ============================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import joblib


# ============================================
# Step 2: Load Dataset
# ============================================
df = pd.read_excel("FEV-data-Excel.xlsx")  # ✅ Replace with your actual file path

print("\n📄 First 5 Rows of the Dataset:")
print(df.head())
print("\n📊 Dataset Info:")
print(df.info())
print("\n🔍 Missing Values Count:")
print(df.isnull().sum())

# ============================================
# Step 3: Handle Missing Values
# ============================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

print("\n✅ Missing values cleaned successfully!")

# ============================================
# Step 4: Encode Categorical Columns
# ============================================
for col in df.select_dtypes(include="object"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("✅ Text columns converted to numbers!")

# ============================================
# Step 5: Prepare Features (X) and Target (y)
# ============================================
df = df.drop(columns=["Car full name", "Model"], errors='ignore')
X = df.drop(columns=["Minimal price (gross) [PLN]"])
y = df["Minimal price (gross) [PLN]"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into training and testing sets!")

# ============================================
# Step 6: Train the Random Forest Model
# ============================================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Model trained successfully!")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 RMSE: {rmse:.2f}")

# ============================================
# Step 7: Feature Importance
# ============================================
importances = model.feature_importances_
features = np.array(X.columns)
indices = np.argsort(importances)[::-1]

print("\n🔍 Top Features Influencing EV Price:")
for i in range(min(10, len(features))):
    print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.3f}")

plt.figure(figsize=(10,6))
plt.barh(features[indices][:10][::-1], importances[indices][:10][::-1])
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting EV Price")
plt.tight_layout()
plt.show()

# ============================================
# Step 8: Save the Model & Feature Columns
# ============================================
joblib.dump(model, "ev_price_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")

print("\n✅ Model and feature names saved successfully as 'ev_price_model.pkl' and 'feature_names.pkl'!")

# ============================================
# Step 9: Test the Model with New Input
# ============================================
print("\n📋 Expected feature order:")
print(list(X.columns))

# Example input (22 features)
new_data = [[
    1,      # Make (encoded)
    150,    # Engine power [KM]
    300,    # Maximum torque [Nm]
    1,      # Type of brakes
    0,      # Drive type
    60,     # Battery capacity [kWh]
    420,    # Range (WLTP) [km]
    270,    # Wheelbase [cm]
    450,    # Length [cm]
    180,    # Width [cm]
    160,    # Height [cm]
    1650,   # Minimal empty weight [kg]
    2100,   # Permissible gross weight [kg]
    450,    # Maximum load capacity [kg]
    5,      # Number of seats
    4,      # Number of doors
    18,     # Tire size [in]
    160,    # Maximum speed [kph]
    500,    # Boot capacity (VDA) [l]
    7.2,    # Acceleration 0–100 kph [s]
    120,    # Maximum DC charging power [kW]
    15.5    # Energy consumption [kWh/100 km]
]]

# Convert list to DataFrame to avoid warning
new_df = pd.DataFrame(new_data, columns=X.columns)

# Predict and display result
predicted_price = model.predict(new_df)
print(f"\n💰 Predicted EV Price: {predicted_price[0]:,.2f} PLN")

# ============================================
# 🚘 Electric Vehicle Price Predictor (₹ INR)
# ============================================

import streamlit as st
import joblib

# Load model and feature names
model = joblib.load("ev_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Conversion: PLN → INR
PLN_TO_INR = 21.0  # Update when needed

# ============================================
# Streamlit Page Setup
# ============================================
st.set_page_config(
    page_title="EV Price Predictor",
    page_icon="⚡",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
    }
    .stButton>button {
        background-color: #0078ff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005ce6;
        color: #fff;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Header
# ============================================
st.title("⚡ Electric Vehicle Price Predictor (₹)")
st.markdown("### Predict your EV price instantly with AI 🔍")
st.write("Provide your EV specifications below to get the estimated **market price in Indian Rupees (₹)**.")

st.divider()

# ============================================
# Input Sections in Cards
# ============================================

with st.container():
    st.markdown("### ⚙️ Performance Details")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            engine_power = st.slider("Engine power [KM]", 50, 500, 150)
            torque = st.slider("Max torque [Nm]", 100, 1000, 300)
        with col2:
            acceleration = st.slider("Acceleration 0–100 kph [s]", 3.0, 15.0, 7.5)
            max_speed = st.slider("Max speed [kph]", 100, 300, 160)
        with col3:
            charging_power = st.slider("DC charging power [kW]", 20, 350, 120)

st.markdown("---")

with st.container():
    st.markdown("### 🔋 Battery & Efficiency")
    col1, col2, col3 = st.columns(3)
    with col1:
        battery_capacity = st.slider("Battery capacity [kWh]", 20.0, 200.0, 60.0)
        range_km = st.slider("Range (WLTP) [km]", 100, 800, 420)
    with col2:
        energy_consumption = st.slider("Energy consumption [kWh/100 km]", 10.0, 30.0, 15.0)
        make = st.number_input("Make (encoded)", 0, 10, 1)
    with col3:
        brakes = st.number_input("Type of brakes (encoded)", 0, 5, 1)
        drive_type = st.number_input("Drive type (encoded)", 0, 5, 0)

st.markdown("---")

with st.container():
    st.markdown("### 📏 Dimensions & Weight")
    col1, col2, col3 = st.columns(3)
    with col1:
        wheelbase = st.slider("Wheelbase [cm]", 200, 400, 270)
        length = st.slider("Length [cm]", 350, 600, 450)
        width = st.slider("Width [cm]", 150, 250, 180)
    with col2:
        height = st.slider("Height [cm]", 130, 200, 160)
        min_weight = st.slider("Empty weight [kg]", 800, 3000, 1650)
        gross_weight = st.slider("Gross weight [kg]", 1200, 3500, 2100)
    with col3:
        load_capacity = st.slider("Load capacity [kg]", 200, 1000, 450)
        boot_capacity = st.slider("Boot capacity [l]", 100, 1000, 500)

st.markdown("---")

with st.container():
    st.markdown("### 🛞 Design & Structure")
    col1, col2, col3 = st.columns(3)
    with col1:
        seats = st.slider("Seats", 2, 9, 5)
        doors = st.slider("Doors", 2, 6, 4)
    with col2:
        tire_size = st.slider("Tire size [in]", 13, 22, 18)
    with col3:
        st.write("✅ All features ready!")

# ============================================
# Combine Input Data
# ============================================
input_data = [
    make, engine_power, torque, brakes, drive_type, battery_capacity, range_km,
    wheelbase, length, width, height, min_weight, gross_weight, load_capacity,
    seats, doors, tire_size, max_speed, boot_capacity, acceleration,
    charging_power, energy_consumption
]

# ============================================
# Prediction Button
# ============================================
st.markdown("---")
st.subheader("🧮 Price Estimation")

if st.button("💰 Predict EV Price (in ₹)"):
    predicted_pln = model.predict([input_data])[0]
    predicted_inr = predicted_pln * PLN_TO_INR

    st.success(f"**Estimated Price:** ₹ {predicted_inr:,.2f}")
    st.caption(f"(Converted from {predicted_pln:,.2f} PLN at ₹{PLN_TO_INR}/PLN)")
    st.balloons()

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown(
    "<center>🚗 Developed with ❤️ using Streamlit | EV Price Predictor (INR)</center>",
    unsafe_allow_html=True
)
