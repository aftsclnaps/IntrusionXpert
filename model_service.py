import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import io
from collections import deque
import shap
import matplotlib.pyplot as plt
import base64
from sklearn.metrics import roc_auc_score # Needed for ROC calculation in analysis

# Set global TensorFlow seed for consistency
tf.random.set_seed(42)

# --- GLOBAL CONSTANTS ---

# Define the full list of column names for the NSL-KDD dataset (41 features + label + difficulty)
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
CAT_COLS = ['protocol_type', 'service', 'flag']

# --- MODEL LOADING (Done Once at startup) ---

try:
    SCALER = joblib.load("scaler.pkl")
    FEAT_COLS = joblib.load("feature_columns.pkl")
    RF_MODEL = joblib.load("ids_rf.pkl")
    CNN_MODEL = tf.keras.models.load_model("ids_cnn.h5")
    # Load test data for SHAP background (if X_test.joblib is available)
    try:
        X_TEST = joblib.load("X_test.joblib")
    except FileNotFoundError:
         X_TEST = None # Handle case where X_test might not be present
         print("Warning: X_test.joblib not found. SHAP background will use a subset of the analyzed data.")
         
except FileNotFoundError as e:
    print(f"FATAL MODEL ERROR: Missing required model artifact file: {e}. Ensure all .pkl, .h5, and .joblib files are in the same directory.")
    # Exiting here is usually better, but we let Flask handle it.

# --- Adaptive FSM Class (Copy of your 3_adaptive_fsm.py) ---
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
            return self.base_threshold, self.base_k, self.base_m

        mean_p = float(np.mean(self.win_probs))
        # Adapt threshold: higher mean malicious prob -> lower threshold (more sensitive)
        thr = np.clip(self.base_threshold - 0.2*(mean_p-0.5), 0.55, 0.9)

        # Adapt k: high mean malicious prob -> fewer confirmations needed
        k = int(np.clip(round(self.base_k - 1.0*(mean_p-0.5)), 1, 4))

        # Adapt m: if stream looks benign, require fewer benign confirmations to return
        m = int(np.clip(round(self.base_m - 1.0*(0.5-mean_p)), 1, 4))

        return thr, k, m

    def step(self, mal_prob):
        self.win_probs.append(mal_prob)
        thr, k2i, m2n = self._current_params()

        is_mal = mal_prob >= thr
        if is_mal:
            self._mal_count += 1; self._ben_count = 0
        else:
            self._ben_count += 1; self._mal_count = 0

        # FSM Transition Logic
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
        
        return self.state, thr, k2i, m2n

# --- Data Preprocessing Function (Crucial for alignment) ---

def align_and_scale_data(file_stream, feat_cols, scaler):
    """Reads the uploaded CSV, preprocesses it, and aligns columns to feat_cols."""
    
    # 1. Read CSV from stream, assign column names
    df = pd.read_csv(io.BytesIO(file_stream), header=None)
    
    if df.shape[1] == len(NSL_COLS):
         df.columns = NSL_COLS
    elif df.shape[1] == len(NSL_COLS) - 1:
        # If the file has 42 columns (features + label, but no difficulty)
        df.columns = NSL_COLS[:-1] # Exclude 'difficulty'
    else:
        raise ValueError(f"Uploaded file has {df.shape[1]} columns. Expected 42 or 43 (features + label + difficulty).")
        
    # Drop the 'label' and 'difficulty' columns before OHE/scaling
    if 'label' in df.columns: df = df.drop(columns=['label'])
    if 'difficulty' in df.columns: df = df.drop(columns=['difficulty'])
    
    # 2. One-Hot Encode Categorical Columns
    df_oh = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    
    # 3. Align Columns to Match Training Data (Crucial Step!)
    # Reindex adds missing columns (sets to 0) and drops extra ones.
    df_aligned = df_oh.reindex(columns=feat_cols, fill_value=0)
    
    # 4. Scale the Data
    X_scaled = scaler.transform(df_aligned.values)
    
    # 5. Reshape for CNN
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    return X_scaled, X_cnn, df_aligned.columns.tolist()

# --- Plotting Utilities ---

def plot_to_base64(fig):
    """Saves a matplotlib figure to a Base64 string for embedding in HTML."""
    if not fig:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_fsm_plots(probs, states, thresholds, k_values, m_values):
    """Generates the three main FSM plots."""
    plots = {}
    indices = np.arange(len(probs))
    
    # 1. Probability and Threshold Timeline
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(indices, probs, label='Malicious Probability', color='#0ff')
    ax1.plot(indices, thresholds, label='Adaptive Threshold', color='red', linestyle='--')
    ax1.set_title('Malicious Probability vs. Adaptive Threshold', color='white')
    ax1.set_xlabel('Packet Index', color='white')
    ax1.set_ylabel('Probability', color='white')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#0f172a')
    fig1.patch.set_facecolor('#0d0d0d')
    plots['probability'] = plot_to_base64(fig1)

    # 2. FSM State Transitions
    state_map = {"Normal": 0, "Suspicious": 1, "Intrusion": 2, "Alert": 3}
    state_vals = [state_map[s] for s in states]
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(indices, state_vals, drawstyle='steps-post', color='lime')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(state_map.keys())
    ax2.set_title('FSM State Transitions', color='white')
    ax2.set_xlabel('Packet Index', color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#0f172a')
    fig2.patch.set_facecolor('#0d0d0d')
    plots['fsm_state'] = plot_to_base64(fig2)

    # 3. Adaptive FSM Parameters
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(indices, k_values, label='k_to_intrusion', color='yellow')
    ax3.plot(indices, m_values, label='m_to_normal', color='cyan')
    ax3.set_title('Adaptive FSM Parameters (k & m)', color='white')
    ax3.set_xlabel('Packet Index', color='white')
    ax3.legend()
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#0f172a')
    fig3.patch.set_facecolor('#0d0d0d')
    plots['fsm_params'] = plot_to_base64(fig3)
    
    return plots

def generate_shap_waterfall(X_scaled, current_feat_cols, probs):
    """Generates a single SHAP waterfall plot for a random malicious packet."""
    
    malicious_indices = np.where(probs >= 0.5)[0]
    
    if len(malicious_indices) == 0:
        return None # No malicious packets found
    
    # Select a random malicious packet index
    idx_to_explain = np.random.choice(malicious_indices)
    
    # Prepare data for SHAP
    instance_scaled = X_scaled[idx_to_explain:idx_to_explain+1]
    instance_cnn = instance_scaled.reshape(1, instance_scaled.shape[1], 1)
    
    # Use a small background dataset (either X_TEST or the current scaled data)
    background_data = X_TEST if X_TEST is not None else X_scaled
    if background_data.shape[0] < 100:
        background_size = background_data.shape[0]
    else:
        background_size = 100
    
    background_indices = np.random.choice(background_data.shape[0], background_size, replace=False)
    background = background_data[background_indices].reshape(background_size, background_data.shape[1], 1)
    
    # SHAP DeepExplainer
    try:
        explainer = shap.DeepExplainer(CNN_MODEL, background)
        shap_values = explainer.shap_values(instance_cnn)
        
        # Prepare data for plotting
        values_array = np.squeeze(shap_values[0])
        base_val = np.squeeze(explainer.expected_value).item()
        data_array = instance_scaled.flatten()
        
        # Generate the waterfall plot
        fig_shap = plt.figure() # Create a new figure
        shap.waterfall_plot(
            shap.Explanation(
                values=values_array, 
                base_values=base_val, 
                data=data_array, 
                feature_names=current_feat_cols
            ),
            show=False
        )
        # Apply dark theme styles to SHAP plot
        plt.gca().tick_params(colors='white')
        plt.gca().set_facecolor('#0f172a')
        fig_shap.patch.set_facecolor('#0d0d0d')
        plt.title(f"SHAP for Packet Index {idx_to_explain}", color='white')
        
        return plot_to_base64(fig_shap)
    
    except Exception as e:
        print(f"SHAP calculation failed: {e}")
        return None

# --- Main Analysis Function ---

def analyze_data(file_stream, base_thr, base_k, base_m, window):
    
    # 1. Preprocess Data and Get Predictions
    X_scaled, X_cnn, current_feat_cols = align_and_scale_data(file_stream, FEAT_COLS, SCALER)
    
    p_rf = RF_MODEL.predict_proba(X_scaled)[:, 1]
    p_cnn = CNN_MODEL.predict(X_cnn).ravel()
    
    # Ensemble prediction (mean)
    probs = np.mean([p_rf, p_cnn], axis=0)

    # 2. Run Adaptive FSM
    fsm = AdaptiveFSM(base_threshold=base_thr, base_k=base_k, base_m=base_m, window=window)
    fsm_results = []
    
    for i, p in enumerate(probs):
        state, thr, k, m = fsm.step(p)
        fsm_results.append({
            "index": i,
            "probability": p,
            "fsm_state": state,
            "threshold": thr,
            "k_value": k,
            "m_value": m,
        })
    
    # 3. Generate Plots
    states = [r['fsm_state'] for r in fsm_results]
    thresholds = [r['threshold'] for r in fsm_results]
    k_values = [r['k_value'] for r in fsm_results]
    m_values = [r['m_value'] for r in fsm_results]
    
    main_plots = generate_fsm_plots(probs, states, thresholds, k_values, m_values)
    
    # 4. Generate SHAP Plot
    main_plots['waterfall_plot'] = generate_shap_waterfall(X_scaled, current_feat_cols, probs)
    
    # 5. Return Results
    return {
        "total_packets": len(probs),
        "fsm_results": fsm_results,
        "plots": main_plots,
    }