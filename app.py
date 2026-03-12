import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn
from utils.preprocess import transform_features
from utils.email_generator import generate_email
from utils.marketing import strategy_map
from utils.segmentations import segment_map
from pathlib import Path

# --- 1. SESSION STATE INITIALIZATION ---
# This must be at the very top to keep the app "smart"
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
    st.session_state.segment = None
    st.session_state.strategy = None
    st.session_state.email_text = None

# Function to hide results as soon as user changes any input
def clear_old_results():
    st.session_state.predicted = False

# --- 2. LOAD MODELS ---
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model" / "kmeans_model.pkl"
scaler_path = BASE_DIR / "model" / "scaler.pkl"

@st.cache_resource # Keeps the app fast by not reloading the model every time
def load_assets():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- 3. UI HEADER ---
st.set_page_config(page_title="AI Marketing Automation", page_icon="🤖")
st.title("Customer Segmentation Demo")
st.write(
    """
    Predict customer segments using Machine Learning and trigger automated marketing via n8n.
    Changing any input will automatically reset the prediction.
    """
)

# --- 4. INPUT SECTION ---
st.header("Enter Customer Behavior Data")

# We add 'on_change=clear_old_results' to every input field
col_id, col_email = st.columns(2)
with col_id:
    customer_id = st.text_input("Customer ID", placeholder="C-1001", on_change=clear_old_results)
with col_email:
    email = st.text_input("Customer Email", placeholder="customer@example.com", on_change=clear_old_results)

st.divider()

col1, col2 = st.columns(2)
with col1:
    recency = st.number_input("Recency (days since last purchase)", value=30, on_change=clear_old_results)
    frequency = st.number_input("Frequency (number of orders)", value=5, on_change=clear_old_results)
    monetary = st.number_input("Total Spending ($)", value=200.0, on_change=clear_old_results)

with col2:
    total_quantity = st.number_input("Total Quantity Purchased", value=10, on_change=clear_old_results)
    avg_order_value = st.number_input("Average Order Value ($)", value=40.0, on_change=clear_old_results)
    promo_ratio = st.slider("Promo Usage Ratio", 0.0, 1.0, 0.3, on_change=clear_old_results)

avg_promo_amount = st.number_input("Average Promo Amount ($)", value=10.0, on_change=clear_old_results)

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Customer Segment", type="primary"):
    data = pd.DataFrame({
        "recency": [recency],
        "frequency": [frequency],
        "monetary": [monetary],
        "total_quantity": [total_quantity],
        "avg_order_value": [avg_order_value],
        "promo_ratio": [promo_ratio],
        "avg_promo_amount": [avg_promo_amount]
    })

    # Preprocessing & Prediction
    data = transform_features(data)
    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    # Storing results in Session State
    st.session_state.segment = segment_map.get(cluster, "Unknown")
    st.session_state.strategy = strategy_map.get(st.session_state.segment, "Unknown")
    st.session_state.email_text = generate_email(st.session_state.segment)
    st.session_state.predicted = True

# --- 6. DISPLAY RESULTS & AUTOMATION ---
# This section only appears when st.session_state.predicted is True
if st.session_state.predicted:
    st.divider()
    st.success(f"**Target Segment:** {st.session_state.segment}")
    
    with st.expander("View Marketing Strategy", expanded=True):
        st.info(st.session_state.strategy)
    
    st.subheader("Generated Marketing Email")
    # Using text_area so user can edit if they want before sending
    final_email = st.text_area("Email Draft", st.session_state.email_text, height=250)

    # TRIGGER BUTTON
    if st.button("🚀 Trigger Marketing Campaign"):
        webhook_url = "https://emdad.app.n8n.cloud/webhook/customer-segmentation"
        
        payload = {
            "customer_id": customer_id,
            "email": email,
            "segment": st.session_state.segment,
            "strategy": st.session_state.strategy,
            "email_text": final_email # Sends the potentially edited text
        }
        
        with st.spinner("Broadcasting to n8n operations hub..."):
            try:
                res = requests.post(webhook_url, json=payload, timeout=10)
                if res.status_code == 200:
                    st.balloons()
                    st.success("✅ Automation Triggered! Check your n8n executions.")
                else:
                    st.error(f"❌ n8n Error: Received status code {res.status_code}")
            except Exception as e:
                st.error(f"📡 Connection Failed: Ensure your n8n instance is online. Error: {e}")

    # Add a small manual reset if they want to clear everything without typing
    st.button("Reset All Fields", on_click=clear_old_results)
