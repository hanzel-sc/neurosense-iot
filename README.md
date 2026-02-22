# NeuroSense AIoT  
## Personalized Multimodal Physiological State Inference System

---

## Overview

NeuroSense AIoT is a low-cost wearable computational physiology prototype that models autonomic nervous system dynamics using multimodal biosignals.

Instead of binary stress detection, the system focuses on:

- Personalized baseline modeling  
- Physiological deviation analysis  
- Temporal dynamics  
- Explainable inference  

---

## Core Principle

> Physiological inference is performed relative to subject-specific baselines rather than universal thresholds.

---

## System Architecture

Human Physiology → Sensors → ESP32 → ThingSpeak → Python Analytics Engine → Dashboard & Insights

---

## Hardware Components

- ESP32 Development Board  
- MAX30102 (PPG Sensor)  
- NTC Thermistor (Skin Temperature)  
- Moisture Sensor (EDA/Sweat Proxy)  
- Breadboard & Jumper Wires  

---

## Signals Acquired

- **Heart Rate (HR)**  
- **RMSSD (HRV)**  
- **Skin Temperature**  
- **Moisture (%)**

---

## Edge Processing (ESP32)

- Beat detection  
- HR computation  
- RMSSD calculation  
- Temperature estimation  
- Baseline deviation scoring  

---

## Cloud Layer (ThingSpeak)

Stores physiological features:

- Field 1 → HR  
- Field 2 → RMSSD  
- Field 3 → SkinTemp  
- Field 4 → Moisture  
- Field 5 → StressScore  

---

## Python Analytics Pipeline

- Data retrieval from ThingSpeak  
- Cleaning & validation  
- Feature engineering (Δ signals, Z-scores)  
- Statistical analysis  
- Anomaly detection (Isolation Forest)  
- Interactive visualizations (Plotly)  
- Explainable insights  

---

## Physiological State Model

Deviation-based interpretation:

- **Stable** → Within baseline manifold  
- **Mild Activation** → Moderate deviation  
- **Strong Activation** → Significant deviation  

---

## Real-Time Feedback

LED indicator:

- Green → Stable  
- Yellow → Mild deviation  
- Red → Strong deviation  

---

## Key Analytical Focus

- Physiological deviations  
- Signal dynamics  
- Rolling variance (instability)  
- Cross-signal correlations  
- Inter-individual variability  

---

## Limitations

- Not a medical device  
- Sensitive to motion artifacts  
- Requires calibration  

---