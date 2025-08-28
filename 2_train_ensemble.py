# 2_train_ensemble.py

import numpy as np, joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load arrays from the files saved by the data preprocessing script
X_train = joblib.load("X_train.joblib")
X_test = joblib.load("X_test.joblib")
y_train = joblib.load("y_train.joblib")
y_test = joblib.load("y_test.joblib")

tf.random.set_seed(42)

# ---- CNN ----
Xtr_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
Xte_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
cw = {0: cw[0], 1: cw[1]}

cnn = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(Xtr_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

cnn.fit(Xtr_cnn, y_train, validation_split=0.2, epochs=20, batch_size=256,
        class_weight=cw, callbacks=[es], verbose=1)

p_cnn = cnn.predict(Xte_cnn).ravel()

print("CNN AUC:", roc_auc_score(y_test, p_cnn))
print(classification_report(y_test, (p_cnn>=0.5).astype(int), digits=4))
print(confusion_matrix(y_test, (p_cnn>=0.5).astype(int)))

cnn.save("ids_cnn.h5")

# ---- RandomForest ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, n_jobs=-1, class_weight='balanced_subsample', random_state=42
)
rf.fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:,1]

print("RF AUC:", roc_auc_score(y_test, p_rf))

joblib.dump(rf, "ids_rf.pkl")

# ---- Ensemble (avg probs) ----
p_ens = (p_cnn + p_rf) / 2.0
print("Ensemble AUC:", roc_auc_score(y_test, p_ens))
print("Ensemble report:\n", classification_report(y_test, (p_ens>=0.5).astype(int), digits=4))