# app.py

import streamlit as st
import pandas as pd, numpy as np, joblib, tensorflow as tf
from collections import deque
from sklearn.ensemble import RandomForestClassifier
# Import the new, simplified DFA
from simple_dfa import SimpleDFA
import time

st.set_page_config(page_title="IntrusionXpert", layout="wide")
st.title("IntrusionXpert — ML + CN + FLA (Simple DFA)")

# Load artifacts
scaler = joblib.load("scaler.pkl")
feat_cols = joblib.load("feature_columns.pkl")
rf: RandomForestClassifier = joblib.load("ids_rf.pkl")
cnn = tf.keras.models.load_model("ids_cnn.h5")

# Sidebar controls
use_ensemble = st.sidebar.checkbox("Use Ensemble (CNN + RF)", value=True)
base_thr = st.sidebar.slider("Malicious threshold", 0.5, 0.95, 0.7, 0.01)

uploaded = st.file_uploader("Upload NSL-KDD style CSV to score", type=["csv"])

# Define the Simple DFA and Streamlit placeholders
dfa = SimpleDFA(threshold=base_thr)
status_placeholder = st.empty()
state_placeholder = st.empty()

def align_and_scale(df):
    # one-hot to match training
    cat = ['protocol_type','service','flag']
    if 'label' not in df.columns:
        df['label'] = 'unknown'
    df_oh = pd.get_dummies(df, columns=[c for c in cat if c in df.columns], drop_first=False)
    for c in feat_cols + ['label']:
        if c not in df_oh.columns:
            df_oh[c] = 0
    df_oh = df_oh[feat_cols + ['label']]
    X = df_oh.drop(columns=['label']).values
    Xs = scaler.transform(X)
    Xc = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
    return Xs, Xc

if uploaded:
    df = pd.read_csv(uploaded)
    Xs, Xc = align_and_scale(df)

    # Probs
    p_rf = rf.predict_proba(Xs)[:,1]
    p_cnn = cnn.predict(Xc).ravel()
    probs = (p_rf + p_cnn)/2.0 if use_ensemble else p_cnn
    
    # Process each data point and update the dashboard
    for i, p in enumerate(probs):
        current_state = dfa.step(float(p))

        # --- NOTIFICATION LOGIC ---
        if current_state == "Intrusion":
            status_placeholder.error(f"🚨 INTRUSION DETECTED! Packet {i+1} is malicious.")
            st.balloons() # Visual effect for alert
        else:
            status_placeholder.success(f"✅ System is in a Normal state. Packet {i+1} is benign.")

        state_placeholder.write(f"Current DFA State: **{current_state}**")
        time.sleep(0.5) # Simulate real-time processing delay