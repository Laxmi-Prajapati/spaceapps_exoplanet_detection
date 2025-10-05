# 🌟 Exoplanet Classification Project - SpaceApps 2025

A comprehensive machine learning pipeline for exoplanet confirmation using physics-based feature engineering and advanced ensemble methods with production-ready deployment capabilities.

## 🎯 Project Overview

This project implements a state-of-the-art binary classification system to distinguish between confirmed exoplanets and false positives using data from NASA's exoplanet archive. The pipeline features sophisticated data preprocessing, physics-first imputation strategies, optimized ensemble learning, and a streamlined deployment architecture.

### 🏆 Key Features

- **Physics-first approach** for stellar parameter calculations
- **Simplified 8-feature pipeline** optimized for deployment
- **Self-contained validation system** for independent testing
- **Production-ready artifacts** with comprehensive metadata
- **Streamlit-optimized architecture** for web applications

## 🔬 Technical Architecture

### Data Processing Pipeline (`unified.ipynb`)

- **Enhanced KNN Imputation**: Physics-motivated features with noise injection
- **Physics-first Calculations**: Stellar mass derived from fundamental physics
- **Comprehensive Error Detection**: Automatic correction of impossible values
- **Stratified Validation**: Robust imputation quality assessment

### Machine Learning Pipeline (`model.ipynb`)

- **Streamlined Feature Selection**: Optimized 8-feature architecture
- **Stacking Ensemble**: Random Forest + Gradient Boosting + SVM + Logistic Regression
- **Grid Search Optimization**: Hyperparameter tuning with cross-validation
- **Class Imbalance Handling**: Random Over-Sampling (ROS)
- **Self-contained Validation**: Independent testing capabilities

### Core Features (8 Selected)

1. **transit_epoch_bjd** - Transit timing precision
2. **transit_depth_ppm** - Signal strength measure
3. **equilibrium_temp_k** - Planet equilibrium temperature
4. **impact_parameter** - Orbital geometry parameter
5. **stellar_teff_k** - Stellar effective temperature
6. **stellar_mass_msun** - Host star mass
7. **stellar_logg** - Surface gravity indicator
8. **radius_ratio_est** - Planet-to-star radius ratio

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

### Production Inference (Simplified)

```python
import pickle
import pandas as pd

# Load the streamlined pipeline
with open('models/pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
with open('models/selected_features.pkl', 'rb') as f:
    features = pickle.load(f)

# Make predictions with just 8 features
def predict_exoplanet(data):
    """
    Predict exoplanet confirmation probability
    data: DataFrame with 8 required features in correct order
    """
    X = data[features]  # Ensure correct feature order
    return pipeline.predict_proba(X)[:, 1]

# Example usage
sample_data = pd.DataFrame({
    'transit_epoch_bjd': [2455000.0],
    'transit_depth_ppm': [1000.0],
    'equilibrium_temp_k': [300.0],
    'impact_parameter': [0.5],
    'stellar_teff_k': [5800.0],
    'stellar_mass_msun': [1.0],
    'stellar_logg': [4.5],
    'radius_ratio_est': [0.1]
})

probability = predict_exoplanet(sample_data)
```

## 📁 Project Structure

```
SpaceApps_VERSION2/
├── unified.ipynb              # Data preprocessing & feature engineering
├── model.ipynb               # Model training & validation
├── exoplanets_dataset_changed.csv  # Raw dataset
├── data/
│   └── unified_exoplanets_final_imputed.csv  # Processed dataset
├── models/
│   ├── README.md             # Model documentation & deployment guide
│   ├── metadata.json         # Model configuration & validation
│   ├── pipeline.pkl          # Complete inference pipeline
│   ├── selected_features.pkl # Feature list (8 optimized features)
│   └── artifacts/            # Individual model components
└── requirements.txt          # Python dependencies
```

## � Model Architecture

### Pipeline Overview

1. **Data Input**: 8 carefully selected features from exoplanet transit data
2. **Preprocessing**: StandardScaler for feature normalization
3. **Classification**: Stacking ensemble with regularized meta-learner
4. **Output**: Confirmation probability (0-1 scale)

### Ensemble Components

- **Base Learners**: RandomForest, GradientBoosting, SVM
- **Meta-Learner**: LogisticRegression with L2 regularization
- **Cross-Validation**: 5-fold stratified for robust performance

### Performance Characteristics

- **Architecture**: Optimized for deployment simplicity
- **Feature Count**: Streamlined to 8 essential parameters
- **Inference Time**: < 10ms per prediction
- **Memory Footprint**: < 50MB model size
- **Random Forest**: 98.72% AUC (best individual)
- **Gradient Boosting**: 99.12% AUC
- **SVM**: 96.54% AUC
- **Logistic Regression**: 89.66% AUC
- **Stacking Ensemble**: 96.64% AUC (final)

## 🔧 Technical Implementation

### Data Processing Pipeline

- **Comprehensive Imputation**: KNN-based approach with physics-derived features
- **Feature Engineering**: 9 enhanced transit and stellar parameters
- **Quality Validation**: Physics consistency checks and error correction
- **Preprocessing Artifacts**: Scalers and configuration saved for reproducibility

### Model Training Workflow

- **Base Learners**: RandomForest (200 trees), GradientBoosting (0.2 LR), SVM (C=10)
- **Meta-Learner**: LogisticRegression with stacking cross-validation
- **Class Balancing**: Random Over-Sampling for optimal class distribution
- **Feature Selection**: Streamlined 8-feature architecture for deployment

### Deployment Architecture

- **Simplified Pipeline**: Direct 8-feature input, no selector complexity
- **Self-Contained Validation**: Comprehensive testing and verification code
- **Production Ready**: Optimized for Streamlit and web deployment
- **Complete Documentation**: Technical guides and inference examples

## 🚀 Deployment

### Streamlit Integration

```python
# Load simplified pipeline for web deployment
import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('models/selected_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return pipeline, features

# User interface for 8 key features
st.title("Exoplanet Confirmation Predictor")
# Interface implementation...
```

### Production Considerations

- **Memory Efficient**: < 50MB total model size
- **Fast Inference**: < 10ms prediction time
- **Robust**: Handles missing values and edge cases
- **Scalable**: Containerized deployment ready

## 📄 License

This project is part of NASA SpaceApps Challenge 2025.

## 🔗 Links

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [SpaceApps Challenge](https://www.spaceappschallenge.org/)

---

*Built with ❤️ for space exploration and machine learning excellence*
