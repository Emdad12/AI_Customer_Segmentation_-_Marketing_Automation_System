import streamlit as st
import pandas as pd
import pickle
import requests
from utils.preprocess import transform_features
from utils.email_generator import generate_email
from utils.marketing import strategy_map
from utils.segmentations import segment_map
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "model" / "kmeans_model.pkl"
scaler_path = BASE_DIR / "model" / "scaler.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Segmentation Demo")
st.write(
"""
This tool predicts customer segments using an unsupervised machine learning model.
Businesses can use this to identify high-value customers and automate marketing campaigns.
"""
)
st.header("Enter Customer Behavior Data")

customer_id = st.text_input("Customer ID")
email = st.text_input("Customer Email")

recency = st.number_input(
"Recency (days since last purchase)",
min_value=0,
max_value=1000,
value=30,
help="How many days since the customer's last order"
)

frequency = st.number_input(
"Frequency (number of orders)",
min_value=1,
value=5
)

monetary = st.number_input(
"Total Spending",
min_value=1.0,
value=200.0
)

total_quantity = st.number_input(
"Total Quantity Purchased",
min_value=1,
value=10
)

avg_order_value = st.number_input(
"Average Order Value",
min_value=1.0,
value=40.0
)

promo_ratio = st.slider(
"Promo Usage Ratio",
0.0,
1.0,
0.3
)

avg_promo_amount = st.number_input(
"Average Promo Amount",
min_value=0.0,
value=10.0
)

if st.button("Predict Customer Segment"):

    data = pd.DataFrame({

    "recency":[recency],
    "frequency":[frequency],
    "monetary":[monetary],
    "total_quantity":[total_quantity],
    "avg_order_value":[avg_order_value],
    "promo_ratio":[promo_ratio],
    "avg_promo_amount":[avg_promo_amount]

    })


    # preprocessing
    data = transform_features(data)

    # scaling
    data_scaled = scaler.transform(data)

    # prediction
    cluster = model.predict(data_scaled)[0]

    segment = segment_map.get(cluster,"Unknown")


    # marketing strategy
    strategy = strategy_map.get(segment,"Unknown")

    # email generation
    email_text = generate_email(segment)


    # -------------------------
    # Display Results
    # -------------------------

    st.success(f"Customer Segment: {segment}")

    st.subheader("Recommended Marketing Strategy")

    st.info(strategy)

    st.subheader("Generated Marketing Email")

    st.text(email_text)

    if st.button("🚀 Trigger Marketing Campaign"):
        webhook_url = "https://emdad.app.n8n.cloud/webhook/customer-segmentation"
        payload = {
        "customer_id": customer_id,
        "email": email,
        "segment": segment,
        "strategy": strategy,
        "email_text": email_text
    }
    
    with st.spinner("Sending to n8n..."):
        try:
            res = requests.post(webhook_url, json=payload)
            if res.status_code == 200:
                st.success("Sent to n8n successfully!")
            else:
                st.error("n8n received the data but returned an error.")
        except Exception as e:
            st.error(f"Could not connect to n8n: {e}")