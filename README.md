🛡️ Adaptive Ensemble IDS: Intrusion Detection System
This project implements an Intrusion Detection System (IDS) using an ensemble of Machine Learning models (Random Forest and CNN) combined with a unique, stateful, and Adaptive Finite State Machine (FSM) for robust real-time anomaly detection. The system is deployed via a Flask web API, allowing users to upload network traffic data (in CSV format) for analysis and visualization of detection and state transitions.
🌟 Features
•	Ensemble Modeling: Combines predictions from a Random Forest Classifier and a 1D Convolutional Neural Network (CNN) for enhanced accuracy and stability.
•	Adaptive Finite State Machine (FSM): Processes the model's output probabilities sequentially, tracking the session state through four phases: Normal → Suspicious → Intrusion → Alert.
•	Dynamic Parameter Adjustment: The FSM adapts its detection sensitivity (threshold, k, and m parameters) based on the rolling mean of recent malicious probabilities in the stream, optimizing for detection rate and false alarm reduction.
•	Model Interpretability (SHAP): Uses SHAP (SHapley Additive exPlanations) to provide feature importance for individual malicious predictions, explaining why a particular packet was flagged.
•	Web API Deployment: A Flask API provides a user interface for file upload, parameter setting, and visualization of the analysis results.
🛠️ Project Structure
•	1_data_prep.py – Data loading, preprocessing (OHE, scaling), and saving of feature sets (X_train.joblib, scaler.pkl, etc.) based on the NSL-KDD dataset.
•	2_train_ensemble.py – Trains the CNN (ids_cnn.h5) and Random Forest (ids_rf.pkl) models and evaluates their ensemble performance.
•	adaptive_fsm.py – Defines the AdaptiveFSM class, which implements the state transition logic and dynamic parameter adaptation.
•	4_shap_explain.py – A standalone script to demonstrate the generation of SHAP waterfall plots for model interpretability.
•	model_service.py – The core backend service. It loads all models/scalers, preprocesses new data, runs the ensemble prediction, executes the FSM, and generates plots.
•	flask_api.py – Sets up the Flask web application and defines the /analyze endpoint for data upload and analysis.
•	templates/index.html – (Assumed) The front-end template for the web interface.
🚀 Setup and Run Instructions
You'll need the following Python libraries:
•	pip install numpy pandas scikit-learn joblib tensorflow keras flask matplotlib shap
You will also need the NSL-KDD dataset files (KDDTrain+.csv and KDDTest+.csv) in the root directory to run the training scripts.
1.	Step 1: Data Preparation
Run the data preprocessing script. This generates the necessary scaled data files and preprocessing objects.
Command: python 1_data_prep.py
2.	Step 2: Train Models
Train the CNN and Random Forest models.
Command: python 2_train_ensemble.py
3.	Step 3 (Optional): Model Interpretation Demo
Run the SHAP explanation script.
Command: python 4_shap_explain.py
4.	Step 4: Run the Web API
Start the Flask application.
Command: python flask_api.py
Access via http://127.0.0.1:5000/
⚙️ Adaptive FSM Parameters
•	base_threshold: The initial malicious probability threshold to flag a packet.
•	base_k: The initial number of consecutive malicious packets needed to transition from Suspicious to Intrusion/Alert.
•	base_m: The initial number of consecutive benign packets needed to reset the state back to Normal.
•	window: The size of the rolling window used to calculate the mean probability for adaptivity.
The FSM logic automatically adjusts dynamic parameters k and m:
- If the rolling mean probability is high, the system requires fewer malicious confirmations (decreases k) and more benign confirmations (increases m), making it more sensitive.
- If the rolling mean probability is low, the system requires fewer benign confirmations (decreases m) to return to Normal, making it less sensitive to false positives.
