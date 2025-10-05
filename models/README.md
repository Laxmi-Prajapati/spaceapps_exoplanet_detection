# Exoplanet Classification Model - Deployment Guide

## Model Overview
- **Model Type**: Stacking Ensemble (RF + GB + SVM + LR)
- **Task**: Binary Classification (Confirmed vs Not Confirmed Exoplanets)
- **Performance**: 0.9573 ROC AUC, 0.8148 F1-Score
- **Training Date**: 2025-10-05 13:56:59

## Quick Start
```python
import pickle
import pandas as pd

# Load model components
with open('models/stacking_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('models/selected_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Make predictions
def predict_exoplanet(data):
    X_scaled = scaler.transform(data[features])
    X_selected = selector.transform(X_scaled)
    return model.predict_proba(X_selected)[:, 1]
```

## Required Features
- transit_epoch_bjd
- transit_depth_ppm
- equilibrium_temp_k
- impact_parameter
- stellar_teff_k
- stellar_mass_msun
- stellar_logg
- radius_ratio_est

## Performance Summary
- **Test Accuracy**: 0.8835
- **Test F1-Score**: 0.8148
- **Test Precision**: 0.8512
- **Test Recall**: 0.7814
- **Test ROC AUC**: 0.9573
- **Test Avg Precision**: 0.9162

## Model Configuration
- **Selected Features**: 8 out of 18
- **Feature Selection**: F-score based (SelectKBest)
- **Class Balancing**: Random Over-Sampling
- **Cross-Validation**: 3-fold CV with roc_auc
