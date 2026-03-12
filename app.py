import streamlit as st
import sklearn
import pandas as pd
import pickle
import requests
from utils.preprocess import transform_features
from utils.email_generator import generate_email
from utils.marketing import strategy_map
from utils.segmentations import segment_map
from pathlib import Path

# 1. Initialize session state at the VERY top
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
    st.session_state.segment = None
    st.session_state.strategy = None
    st.session_state.email_text = None

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "model" / "kmeans_model.pkl"
scaler_path = BASE_DIR / "model" / "scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Segmentation Demo")
st.write("This tool predicts customer segments and automates marketing campaigns.")

st.header("Enter Customer Behavior Data")
customer_id = st.text_input("Customer ID")
email = st.text_input("Customer Email")

# (Input fields remain the same...)
recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
monetary = st.number_input("Total Spending", min_value=1.0, value=200.0)
total_quantity = st.number_input("Total Quantity Purchased", min_value=1, value=10)
avg_order_value = st.number_input("Average Order Value", min_value=1.0, value=40.0)
promo_ratio = st.slider("Promo Usage Ratio", 0.0, 1.0, 0.3)
avg_promo_amount = st.number_input("Average Promo Amount", min_value=0.0, value=10.0)

# --- PREDICTION BUTTON ---
if st.button("Predict Customer Segment"):
    data = pd.DataFrame({
        "recency":[recency], "frequency":[frequency], "monetary":[monetary],
        "total_quantity":[total_quantity], "avg_order_value":[avg_order_value],
        "promo_ratio":[promo_ratio], "avg_promo_amount":[avg_promo_amount]
    })

    data = transform_features(data)
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    # Save to session state
    st.session_state.segment = segment_map.get(cluster, "Unknown")
    st.session_state.strategy = strategy_map.get(st.session_state.segment, "Unknown")
    st.session_state.email_text = generate_email(st.session_state.segment)
    st.session_state.predicted = True

# --- RESULTS DISPLAY & TRIGGER (Outside the first button) ---
if st.session_state.predicted:
    st.markdown("---")
    st.success(f"Customer Segment: {st.session_state.segment}")
    
    st.subheader("Recommended Marketing Strategy")
    st.info(st.session_state.strategy)
    
    st.subheader("Generated Marketing Email")
    st.text_area("Email Content", st.session_state.email_text, height=200)

    # Use a unique key for the second button
    if st.button("🚀 Trigger Marketing Campaign", key="trigger_btn"):
        webhook_url = "https://emdad.app.n8n.cloud/webhook/customer-segmentation"
        
        payload = {
            "customer_id": customer_id,
            "email": email,
            "segment": st.session_state.segment,
            "strategy": st.session_state.strategy,
            "email_text": st.session_state.email_text
        }
        
        with st.spinner("Sending data to n8n..."):
            try:
                res = requests.post(webhook_url, json=payload, timeout=10)
                if res.status_code == 200:
                    st.balloons()
                    st.success("✅ Success! n8n has received the data.")
                else:
                    st.error(f"❌ n8n error: {res.status_code}")
            except Exception as e:
                st.error(f"📡 Connection Failed: {e}")
