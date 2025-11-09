# ⚡ Electric Vehicle Price Prediction (EV-Price-Predictor)

An interactive **Machine Learning web app** that predicts **Electric Vehicle (EV)** prices based on their technical specifications.  
Built using **Python**, **Random Forest**, and **Streamlit**.

---

## 🚀 Features

- ✅ Predict EV prices instantly (converted to ₹ INR)
- ✅ Clean Streamlit UI with sliders and inputs
- ✅ Random Forest model trained on real EV dataset
- ✅ Automatic data preprocessing & feature encoding
- ✅ Model persistence using Joblib
- ✅ Visual insights (feature importance, charts)

---

## 🧠 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python |
| ML Model | RandomForestRegressor |
| Libraries | pandas, numpy, matplotlib, scikit-learn, joblib |
| Web App | Streamlit |
| Dataset | FEV-data-Excel.xlsx |

---

## 🧩 Dataset

The dataset includes detailed EV specifications such as:
- Battery capacity
- Engine power
- Range (WLTP)
- Weight and dimensions
- Torque, acceleration, top speed
- Energy consumption, boot capacity, etc.

---

## 🧮 Model Workflow

1. **Data Preprocessing**
   - Handles missing values and encodes categorical data.
2. **Model Training**
   - Trains a Random Forest model to predict EV price.
3. **Evaluation**
   - Computes R² score and RMSE.
4. **Model Saving**
   - Saves model (`ev_price_model.pkl`) and feature names.
5. **Prediction**
   - Predicts EV price based on new user input in the web app.

---

## 🌐 Streamlit App Usage

### 🧾 Step 1: Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib joblib streamlit openpyxl

🧾 Step 2: Run the App
streamlit run ev_price_app.py

🧾 Step 3: Use the App

Adjust EV parameters using sliders.

Click "💰 Predict EV Price".

See estimated price in ₹ INR.