# 4_shap_explain.py

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved data and models from previous steps
X_test = joblib.load("X_test.joblib")
y_test = joblib.load("y_test.joblib")
feature_columns = joblib.load("feature_columns.pkl")
cnn_model = load_model("ids_cnn.h5")

# --- SHAP Explanation for CNN ---
# The CNN model expects 3D input, so we need to reshape the data
Xte_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Select a subset of benign and malicious samples for explanation
idx_ben = np.where(y_test == 0)[0][:100]
idx_mal = np.where(y_test == 1)[0][:100]
X_explain = np.vstack([X_test[idx_ben], X_test[idx_mal]])
X_explain_cnn = np.vstack([Xte_cnn[idx_ben], Xte_cnn[idx_mal]])

# Use a background dataset for the explainer
background = Xte_cnn[np.random.choice(Xte_cnn.shape[0], 100, replace=False)]

# Create the SHAP DeepExplainer
explainer = shap.DeepExplainer(cnn_model, background)

# --- CORRECTED CODE ---
# To fix the ValueError, we compute SHAP values for each instance individually
# This can be slow, but it's a reliable way to get the correct shape
shap_values_list = []
for i in range(X_explain_cnn.shape[0]):
    print(f"Calculating SHAP for instance {i+1}/{X_explain_cnn.shape[0]}...")
    shap_values = explainer.shap_values(X_explain_cnn[i:i+1])
    shap_values_list.append(shap_values[0].flatten())

# Stack the results to create the correct 2D array
shap_values_2d = np.vstack(shap_values_list)

# --- END OF CORRECTED CODE ---

# Display the summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values_2d, X_explain, feature_names=feature_columns, plot_type="bar")
plt.show()

# Display a force plot for a single malicious instance
print("\nGenerating SHAP force plot...")
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values_2d[100], X_explain[100], feature_names=feature_columns)