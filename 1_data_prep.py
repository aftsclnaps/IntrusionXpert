# 1_data_prep.py

import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

NSL_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    'label','difficulty'
]

def read_nsl(path):
    try:
        df = pd.read_csv(path)
        if 'label' not in df.columns:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == len(NSL_COLS):
            df.columns = NSL_COLS
        else:
            df.columns = NSL_COLS[:-1]  # no difficulty
    if 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
    return df

train = read_nsl("KDDTrain+.csv")
test  = read_nsl("KDDTest+.csv")

# Binary label
train['y'] = (train['label'] != 'normal').astype(int)
test['y']  = (test['label']  != 'normal').astype(int)

cat = ['protocol_type','service','flag']
both = pd.concat([train.drop(columns=['y']), test.drop(columns=['y'])], axis=0, ignore_index=True)
both_oh = pd.get_dummies(both, columns=cat, drop_first=False)

X_train_all = both_oh.iloc[:len(train)].drop(columns=['label'])
X_test_all  = both_oh.iloc[len(train):].drop(columns=['label'])
y_train = train['y'].values
y_test  = test['y'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_all.values)
X_test  = scaler.transform(X_test_all.values)

import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_train_all.columns), "feature_columns.pkl")

# --- MODIFIED CODE STARTS HERE ---
joblib.dump(X_train, "X_train.joblib")
joblib.dump(X_test, "X_test.joblib")
joblib.dump(y_train, "y_train.joblib")
joblib.dump(y_test, "y_test.joblib")
# --- MODIFIED CODE ENDS HERE ---

print("Shapes:", X_train.shape, X_test.shape, " Pos rate:", y_train.mean(), y_test.mean())