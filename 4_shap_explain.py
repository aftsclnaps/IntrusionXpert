# 4_shap_explain.py

import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load test data and model
X_test = joblib.load("X_test.joblib")
y_test = joblib.load("y_test.joblib")
feature_columns = joblib.load("feature_columns.pkl")
rf_model: RandomForestClassifier = joblib.load("ids_rf.pkl")

# Pick some malicious samples
mal_idx = np.where(y_test == 1)[0]
if len(mal_idx) == 0:
    raise ValueError("No malicious samples in test set")

instance_id = np.random.choice(mal_idx)
instance = X_test[instance_id:instance_id+1]

# SHAP with TreeExplainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(instance)

# Take malicious class contributions
sv_mal = shap_values[1][0]
base_val = explainer.expected_value[1]

print(f"Explaining instance {instance_id} (malicious)")

# Waterfall plot
shap.waterfall_plot(
    shap.Explanation(
        values=sv_mal,
        base_values=base_val,
        data=instance[0],
        feature_names=feature_columns
    )
)
plt.show()
