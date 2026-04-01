# 🚀AI Customer Segmentation & Marketing Automation System

An end-to-end machine learning system that segments customers using behavioral data and automates personalized marketing campaigns through real-time prediction and workflow automation.

### 📌 Project Overview

This project builds a complete customer analytics and marketing automation pipeline by combining:

Customer segmentation using machine learning
Business-driven marketing strategy generation
Automated email campaign delivery

Users can input customer behavior data, instantly receive a predicted segment, and trigger automated personalized marketing emails.

### 🧠 Problem Statement

Businesses often struggle to:

Identify high-value customers
Personalize marketing campaigns
Automate engagement strategies

This project solves these problems using data-driven customer segmentation and automation.

### ⚙️ System Architecture
User Input (RFM Features)
        ↓
Data Preprocessing
        ↓
ML Model (Clustering)
        ↓
Customer Segment Prediction
        ↓
Marketing Strategy Engine
        ↓
Email Generator
        ↓
Webhook Trigger
        ↓
Automation (n8n)
        ↓
Email Sent
### 🛠️ Tech Stack
- Python
- scikit-learn
- Streamlit
- n8n
- Pandas, NumPy
- REST API (Webhook integration)
### 📊 Features
- 🔍 Customer Segmentation using RFM + behavioral features
- 🤖 Multiple Clustering Models (K-Means, DBSCAN, Hierarchical, GMM)
- ⚡ Real-Time Prediction via Streamlit web app
- 🎯 Marketing Strategy Engine based on customer segments
- ✉️ Automated Email Generation for personalized campaigns
- 🔗 Workflow Automation using n8n webhooks
- 🌐 Deployed Web Application for interactive demo
- 📥 Input Features

**The model uses the following customer features:**

- Recency
- Frequency
- Monetary
- Total Quantity
- Average Order Value
- Promo Ratio
- Average Promo Amount
### 📈 Model Development
- Performed feature engineering based on RFM analysis
- Applied transformations (e.g., log scaling for monetary features)
- Trained and evaluated multiple clustering algorithms
- Selected K-Means based on clustering performance metrics
### 🎯 Output
- Predicted customer segment
- Recommended marketing strategy
- Generated personalized email message
- Automatic email delivery via automation pipeline
### 🔗 Live Demo

***👉 Streamlit App:***
[(Click Here)](https://aicustomersegmentationmarketingautomationsystem-exaycfw3ned9js.streamlit.app/)

### 💻 How to Run Locally
git clone https://github.com/emdad-12/customer-segmentation-marketing-automation.git  
- cd customer-segmentation-marketing-automation
- pip install -r requirements.txt
- streamlit run app.py
### 🚀 Future Improvements
- Add customer segmentation visualization dashboard
- Store prediction history in database
- Integrate AI-based email generation (LLM)
- Connect with real-time business data sources
- Build full SaaS-style analytics interface
🎯 Skills Demonstrated
- Machine Learning (Clustering, Feature Engineering)
- Data Analytics (RFM Segmentation)
- Web App Development
- API Integration & Automation
- End-to-End ML System Design
- Business-Oriented Problem Solving
### 📌 Conclusion

This project demonstrates how machine learning can be combined with automation tools to create a real-world marketing intelligence system, enabling businesses to deliver personalized and scalable customer engagement strategies.
# Thank You
