# 🌟 Exoplanet Classification Project - SpaceApps 2025

A comprehensive machine learning pipeline for exoplanet confirmation using physics-based feature engineering and advanced ensemble methods.

## 🎯 Project Overview

This project implements a state-of-the-art binary classification system to distinguish between confirmed exoplanets and false positives using data from NASA's exoplanet archive. The pipeline features sophisticated data preprocessing, physics-first imputation strategies, and optimized ensemble learning.

### 🏆 Key Achievements
- **96.64% ROC AUC** on test data
- **84.45% F1-Score** with balanced precision-recall
- **Physics-first approach** for stellar parameter calculations
- **Production-ready** deployment package

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC AUC | 0.9664 |
| F1-Score | 0.8445 |
| Accuracy | 89.90% |
| Precision | 85.32% |
| Recall | 83.61% |

## 🔬 Technical Features

### Data Processing Pipeline (`unified.ipynb`)
- **Enhanced KNN Imputation**: 9 physics-motivated features with noise injection
- **Physics-first Calculations**: Stellar mass derived from fundamental physics
- **Comprehensive Error Detection**: Automatic correction of impossible values
- **Stratified Validation**: Robust imputation quality assessment

### Machine Learning Pipeline (`model.ipynb`)
- **F-Score Feature Selection**: Top 12 features from 18 candidates
- **Stacking Ensemble**: Random Forest + Gradient Boosting + SVM + Logistic Regression
- **Grid Search Optimization**: Hyperparameter tuning with 3-fold cross-validation
- **Class Imbalance Handling**: Random Over-Sampling (ROS)

### Key Features by Importance
1. **equilibrium_temp_k** (18.1%) - Planet equilibrium temperature
2. **impact_parameter** (13.4%) - Orbital geometry parameter  
3. **transit_depth_ppm** (11.9%) - Signal strength measure
4. **radius_ratio_est** (10.7%) - Planet-to-star radius ratio
5. **transit_epoch_bjd** (10.5%) - Transit timing

## 🚀 Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Data Processing
```bash
jupyter notebook unified.ipynb
```

### Model Training
```bash
jupyter notebook model.ipynb
```

### Production Inference
```python
import pickle
import pandas as pd

# Load trained model
with open('models/stacking_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Make predictions
def predict_exoplanet(data):
    X_scaled = scaler.transform(data)
    X_selected = selector.transform(X_scaled)
    return model.predict_proba(X_selected)[:, 1]
```

## 📁 Project Structure

```
├── unified.ipynb              # Data preprocessing pipeline
├── model.ipynb               # Machine learning pipeline
├── requirements.txt          # Python dependencies
├── data/                     # Processed datasets
│   └── unified_exoplanets_final_imputed.csv
├── models/                   # Trained model artifacts
│   ├── stacking_classifier.pkl
│   ├── scaler.pkl
│   ├── feature_selector.pkl
│   ├── metadata.json
│   └── README.md
└── artifacts/                # Preprocessing artifacts
    ├── derive_physics_columns.pkl
    ├── preprocess_config.pkl
    └── transit_knn_scaler.pkl
```

## 🔧 Technical Implementation

### Enhanced Imputation Strategy
- **KNN with Physics Features**: 9 engineered features including orbital dynamics
- **Noise Injection**: Preserves natural variance (30% neighbor std, min 5%)
- **Stratified Validation**: Performance assessment across impact parameter bins
- **RobustScaler**: Outlier-resilient feature scaling

### Optimized Model Architecture
- **Base Models**: RF (200 est.), GB (lr=0.2), SVM (C=10), LR (C=10, L1)
- **Meta-learner**: Logistic Regression with stacking cross-validation
- **Feature Selection**: F-score based SelectKBest (top 12/18 features)
- **Class Balancing**: Random Over-Sampling for 1:1 class ratio

### Deployment Package
- **Complete Pipeline**: All preprocessing and model components saved
- **Metadata Tracking**: Training parameters, performance metrics, feature importance
- **Production Guide**: Detailed README with inference examples
- **Reproducibility**: Fixed random seeds, versioned dependencies

## 📈 Validation Results

### Cross-Validation Performance
- **Random Forest**: 98.72% AUC (best individual)
- **Gradient Boosting**: 99.12% AUC 
- **SVM**: 96.54% AUC
- **Logistic Regression**: 89.66% AUC
- **Stacking Ensemble**: 96.64% AUC (final)

### Overfitting Analysis
- Train-Test AUC Gap: 3.36% (healthy)
- Train-Test F1 Gap: 15.52% (acceptable)
- Strong generalization performance

## 🧪 Scientific Validation

### Physics Consistency
- **Stellar Mass Accuracy**: 2.19e-16 max relative error
- **Critical Error Correction**: 3 types of impossible values detected/fixed
- **Physics-derived Features**: Mass-radius relationships, surface gravity

### Feature Engineering Excellence
- **9 Enhanced KNN Features**: Orbital period, stellar properties, transit geometry
- **Interaction Terms**: Duration × radius products, scaled semi-major axis
- **Derived Parameters**: Equilibrium temperature, insolation flux ratios

## 👥 Contributors

- **Akshat** - Lead ML Engineer & Data Scientist

## 📄 License

This project is part of NASA SpaceApps Challenge 2025.

## 🔗 Links

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [SpaceApps Challenge](https://www.spaceappschallenge.org/)

---

*Built with ❤️ for space exploration and machine learning excellence*