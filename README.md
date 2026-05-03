# IntrusionXpert  
### Adaptive Ensemble-Based Network Intrusion Detection with Explainable AI and LLM-Assisted Security Recommendations

## Overview

IntrusionXpert is an intelligent intrusion detection system designed for LAN traffic classification.  
It combines Deep Learning, Machine Learning, Explainable AI, Adaptive Finite State Machines, and LLM-assisted recommendations for real-time cybersecurity decision support.

This project classifies network traffic into:

- Benign
- Malicious

and provides:

- Threat severity analysis
- Explainability using SHAP
- Adaptive state transition analysis
- AI-generated security recommendations

---

## SDG Alignment

This project supports:

- **SDG 9** — Industry, Innovation and Infrastructure  
- **SDG 16** — Peace, Justice and Strong Institutions  

by improving cybersecurity resilience using intelligent intrusion detection.

---

## Features

- CNN-based traffic classification
- Random Forest ensemble model
- Adaptive Finite State Machine (FSM)
- SHAP explainability
- Streamlit interactive dashboard
- LLM-based AI security assistant
- Flask API for model serving
- Docker containerization
- Git version control

---

## Project Architecture

Input Network Traffic  
↓  
Data Preprocessing  
↓  
CNN + Random Forest  
↓  
Ensemble Prediction  
↓  
Adaptive FSM  
↓  
SHAP Explainability  
↓  
LLM Security Assistant  
↓  
Streamlit Dashboard / Flask API  
↓  
Docker Deployment

---

## Project Structure

```text
IntrusionXpert/
│── 1_data_prep.py
│── 2_train_ensemble.py
│── adaptive_fsm.py
│── app.py
│── api.py
│── 5_llm_assistant.py
│── requirements.txt
│── Dockerfile
│── run.sh
│── README.md
│── ids_cnn.h5
│── ids_rf.pkl
│── scaler.pkl
│── feature_columns.pkl
