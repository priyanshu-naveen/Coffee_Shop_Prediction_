import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ===================== BASIC PAGE SETUP ===================== #
st.set_page_config(
    page_title="Coffee Shop Daily Revenue Predictor",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Coffee Shop Daily Revenue Predictor")
st.write(
    "Predict your coffee shop's **daily revenue** based on key business factors. "
    "Use the sidebar to adjust inputs and click the button to get a prediction."
)

# ===================== MODEL LOADING ===================== #

MODEL_PATH = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_model():
    """Load the trained model from disk (cached across reruns)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at: {MODEL_PATH}. "
            "Please make sure the file is in the same folder as this app."
        )
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

# ===================== SIDEBAR INPUTS ===================== #

st.sidebar.header("🧮 Input Features")

customers_per_day = st.sidebar.slider(
    "Number of Customers Per Day",
    min_value=0,
    max_value=1000,
    value=200,
    help="Average number of customers visiting the coffee shop per day."
)

avg_order_value = st.sidebar.slider(
    "Average Order Value ($)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
    help="Average amount spent by each customer per visit."
)

operating_hours = st.sidebar.slider(
    "Operating Hours Per Day",
    min_value=0,
    max_value=24,
    value=10,
    help="Number of hours the coffee shop is open each day."
)

num_employees = st.sidebar.slider(
    "Number of Employees",
    min_value=1,
    max_value=50,
    value=5,
    help="Number of employees working at the coffee shop."
)

marketing_spend = st.sidebar.slider(
    "Marketing Spend Per Day ($)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=1.0,
    help="Amount spent on marketing per day."
)

foot_traffic = st.sidebar.slider(
    "Location Foot Traffic",
    min_value=0,
    max_value=2000,
    value=500,
    help="Estimated foot traffic at the location per day."
)

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About This App")
st.sidebar.write(
    "This app uses a **Random Forest Regressor** trained on historical coffee shop data "
    "to estimate daily revenue from:\n"
    "- Number of customers per day\n"
    "- Average order value\n"
    "- Operating hours per day\n"
    "- Number of employees\n"
    "- Marketing spend per day\n"
    "- Location foot traffic\n\n"
    "The model achieved an **R² score of ~0.95** on the test set."
)

# ===================== PREDICTION BUTTON ===================== #

st.markdown("### 🔍 Get Prediction")
st.write("Set your inputs on the left, then click the button below to predict daily revenue.")

if st.button("Predict Daily Revenue"):
    try:
        # Build input only when needed
        input_data = pd.DataFrame({
            "Number_of_Customers_Per_Day": [customers_per_day],
            "Average_Order_Value": [avg_order_value],
            "Operating_Hours_Per_Day": [operating_hours],
            "Number_of_Employees": [num_employees],
            "Marketing_Spend_Per_Day": [marketing_spend],
            "Location_Foot_Traffic": [foot_traffic],
        })

        with st.spinner("Loading model and predicting..."):
            model = load_model()          # cached: loads once per session
            prediction = model.predict(input_data)
            revenue = float(prediction[0])

        st.header("✅ Prediction Result")
        st.success(f"Estimated Daily Revenue: **${revenue:,.2f}**")

        st.header("📊 Revenue Insights")
        st.write(
            f"With **{customers_per_day} customers per day** and an "
            f"**average order value of ${avg_order_value:.2f}**, "
            f"your estimated daily revenue is **${revenue:,.2f}**."
        )

        st.info(
            "⚠️ **Note:** This is an estimate from a machine learning model trained on historical data. "
            "Actual revenue may vary due to competition, seasonality, local events, and customer loyalty."
        )

    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(
            "❌ An unexpected error occurred while loading the model or making a prediction.\n\n"
            f"Details: `{e}`"
        )
else:
    st.caption("👆 Adjust the values in the sidebar and click **Predict Daily Revenue** to see the result.")
