# app.py  -> run:  streamlit run app.py

import streamlit as st
import pandas as pd, numpy as np, joblib, tensorflow as tf
from collections import deque
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import io

# --- Defining cat_cols directly to resolve the import error ---
cat_cols = ['protocol_type', 'service', 'flag']

# Define the full list of column names for the NSL-KDD dataset
NSL_COLS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

# Streamlit page configuration
st.set_page_config(page_title="IntrusionXpert", layout="wide")
st.title("IntrusionXpert — ML + CN + FLA (Adaptive FSM)")

# Load artifacts
scaler = joblib.load("scaler.pkl")
feat_cols = joblib.load("feature_columns.pkl")
rf: RandomForestClassifier = joblib.load("ids_rf.pkl")
cnn = tf.keras.models.load_model("ids_cnn.h5")

# Sidebar controls
use_ensemble = st.sidebar.checkbox("Use Ensemble (CNN + RF)", value=True)
base_thr = st.sidebar.slider("Base malicious threshold", 0.5, 0.95, 0.7, 0.01)
base_k = st.sidebar.slider("Base k_to_intrusion", 1, 5, 2)
base_m = st.sidebar.slider("Base m_to_normal", 1, 5, 2)
window = st.sidebar.slider("Adapt window size", 20, 300, 100, 10)

uploaded = st.file_uploader("Upload NSL-KDD style CSV to score", type=["csv"])

# Adaptive FSM (inline to avoid import path issues...)
class AdaptiveFSM:
    def __init__(self, base_threshold=0.7, base_k=2, base_m=2, window=100):
        self.state = "Normal"
        self.base_threshold = base_threshold
        self.base_k = base_k
        self.base_m = base_m
        self.window = window
        self.win_probs = deque(maxlen=window)
        self._mal_count = 0
        self._ben_count = 0

    def _current_params(self):
        if len(self.win_probs) == 0:
            return self.base_threshold, self.base_k, self.base_m, {}

        mean_p = float(np.mean(self.win_probs))
        thr = np.clip(self.base_threshold - 0.2*(mean_p-0.5), 0.55, 0.9)
        k = int(np.clip(round(self.base_k - 1.0*(mean_p-0.5)), 1, 4))
        m = int(np.clip(round(self.base_m - 1.0*(0.5-mean_p)), 1, 4))

        return thr, k, m, {"thr": thr, "k": k, "m": m}

    def step(self, mal_prob):
        self.win_probs.append(mal_prob)
        thr, k2i, m2n, prms = self._current_params()
        
        is_mal = mal_prob >= thr
        if is_mal:
            self._mal_count += 1; self._ben_count = 0
        else:
            self._ben_count += 1; self._mal_count = 0

        if self.state == "Normal":
            if is_mal:
                self.state = "Suspicious"

        elif self.state == "Suspicious":
            if self._mal_count >= k2i:
                self.state = "Intrusion"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Intrusion":
            if self._mal_count >= k2i:
                self.state = "Alert"
            elif self._ben_count >= m2n:
                self.state = "Normal"

        elif self.state == "Alert":
            if self._ben_count >= m2n:
                self.state = "Normal"
        
        return self.state, prms

def align_and_scale(df, feat_cols, cat_cols):
    df_oh = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    for c in feat_cols:
        if c not in df_oh.columns:
            df_oh[c] = 0
    df_oh = df_oh[feat_cols]
    X = df_oh.values
    Xs = scaler.transform(X)
    Xc = Xs.reshape(Xs.shape[0], Xs.shape[1], 1)
    return Xs, Xc

if uploaded:
    try:
        uploaded_string = io.StringIO(uploaded.getvalue().decode('utf-8'))
        df = pd.read_csv(uploaded_string)
        if not all(col in df.columns for col in cat_cols):
            raise ValueError("File does not have a proper header.")
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, header=None, names=NSL_COLS)

    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])

    if not all(col in df.columns for col in cat_cols):
        st.error(f"The uploaded CSV file is missing required columns. Please ensure it contains: {cat_cols}")
    else:
        if 'label' in df.columns:
            df = df.drop(columns=['label'])

        Xs, Xc = align_and_scale(df, feat_cols, cat_cols)
        
        p_rf = rf.predict_proba(Xs)[:,1]
        p_cnn = cnn.predict(Xc).ravel()
        probs = (p_rf + p_cnn)/2.0 if use_ensemble else p_cnn

        fsm = AdaptiveFSM(base_threshold=base_thr, base_k=base_k, base_m=base_m, window=window)
        states, thrs, ks, ms = [], [], [], []
        for p in probs:
            s, prm = fsm.step(float(p))
            states.append(s); thrs.append(prm["thr"]); ks.append(prm["k"]); ms.append(prm["m"])
        
        st.subheader("1. Malicious Probability Over Time")
        st.line_chart(pd.DataFrame({"probability": probs}), use_container_width=True)
        st.caption("This graph shows the predicted malicious probability for each packet. Values closer to 1 indicate a higher probability of being malicious.")

        # --- FSM State Over Time: Now with clearer transitions and a better look ---
        st.subheader("2. FSM State Over Time")
        state_order = ["Normal", "Suspicious", "Intrusion", "Alert"]
        state_data = pd.DataFrame({"FSM State": states})
        state_data["state_encoded"] = state_data["FSM State"].apply(lambda s: state_order.index(s))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(state_data.index, state_data["state_encoded"], marker='o', linestyle='-', color='b', label="FSM State")
        ax.set_yticks(range(len(state_order)))
        ax.set_yticklabels(state_order)
        ax.set_title("FSM State Transitions Over Time", fontsize=16)
        ax.set_xlabel("Time Step (Packets)", fontsize=12)
        ax.set_ylabel("FSM State", fontsize=12)
        ax.legend()
        
        # Add vertical lines at each state transition
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                ax.axvline(x=i, color='gray', linestyle='--', linewidth=1)

        st.pyplot(fig, use_container_width=True)
        st.caption("This graph clearly shows the FSM transitioning between different states based on the detected threats. The Y-axis now shows the state names directly for better readability, and vertical lines mark each state change.")
        plt.clf()

        st.subheader("3. Adaptive FSM Parameters")
        params = pd.DataFrame({
            "Dynamic Threshold": thrs,
            "Required malicious packets (k)": ks,
            "Required benign packets (m)": ms
        })
        st.line_chart(params, use_container_width=True)
        st.caption("This graph visualizes how the FSM's internal parameters adapt based on the recent history of malicious probabilities. A lower threshold and a smaller 'k' value mean the system becomes more sensitive to threats.")

        # --- Model Explainability (SHAP): Now using a waterfall plot with clearer explanation ---
        st.subheader("4. Model Explainability (SHAP)")
        st.markdown("""
        **How to read this plot:**
        This **waterfall plot** shows exactly how each feature contributed to the model's prediction for a single data point.
        - The **base value** is the average prediction across the entire dataset.
        - Each **colored bar** represents a feature.
        - **Red** bars indicate features that pushed the prediction **higher** (more malicious).
        - **Blue** bars indicate features that pushed the prediction **lower** (more normal).
        - The **final output value** is the model's prediction for this specific packet.
        """)
        
        if st.button("Explain a Random Malicious Prediction"):
            with st.spinner("Calculating SHAP values... This may take a moment."):
                malicious_indices = np.where(probs > 0.5)[0]

                if len(malicious_indices) > 0:
                    idx_to_explain = np.random.choice(malicious_indices)
                    
                    instance = Xc[idx_to_explain:idx_to_explain+1]
                    background = Xc[np.random.choice(Xc.shape[0], 100, replace=False)]
                    
                    explainer = shap.DeepExplainer(cnn, background)
                    shap_values = explainer.shap_values(instance)
                    
                    # FIX: Use np.squeeze() to flatten the TensorFlow tensors into the correct shape.
                    values_array = np.squeeze(shap_values[0])
                    base_val = np.squeeze(explainer.expected_value).item()
                    data_array = instance.flatten()
                    
                    fig = shap.waterfall_plot(
                        shap.Explanation(
                            values=values_array, 
                            base_values=base_val, 
                            data=data_array, 
                            feature_names=feat_cols
                        ),
                        show=False
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.clf()
                    
                    st.caption(f"SHAP explanation for packet at index **{idx_to_explain}**.")

                else:
                    st.info("No malicious predictions (probability > 0.5) found in the uploaded file to explain.")
        
        st.subheader("Timeline")
        timeline = pd.DataFrame({
            "index": np.arange(len(probs)),
            "prob_malicious": probs,
            "fsm_state": states
        }).set_index("index")
        st.dataframe(timeline)