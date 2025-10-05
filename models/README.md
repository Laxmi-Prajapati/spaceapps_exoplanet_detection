# Exoplanet Stacking Model

- **ROC AUC (test)**: 0.954
- **F1 (test)**: 0.823
- **Features used** (8): transit_epoch_bjd, transit_depth_ppm, equilibrium_temp_k, impact_parameter, stellar_teff_k, stellar_mass_msun, stellar_logg, radius_ratio_est

## Artifacts (PKL files)
- `pipeline.pkl`: A scikit-learn `Pipeline` with `StandardScaler` and `StackingClassifier`. Expects exactly 8 features in the order listed below.
- `selected_features.pkl`: List of the 8 feature names the pipeline expects.

## Required Features (in this exact order)
1. transit_epoch_bjd
2. transit_depth_ppm
3. equilibrium_temp_k
4. impact_parameter
5. stellar_teff_k
6. stellar_mass_msun
7. stellar_logg
8. radius_ratio_est

## Quick predict (recommended)
```python
import pickle
import pandas as pd

# Load artifacts
with open('models/pipeline.pkl', 'rb') as fh:
    pipe = pickle.load(fh)
with open('models/selected_features.pkl', 'rb') as fh:
    features = pickle.load(fh)

# Your input data must have exactly these 8 features in the correct order
# df = pd.DataFrame with columns matching 'features' list
X = df[features]  # Ensure correct feature order
probas = pipe.predict_proba(X)[:, 1]

# Convert to class predictions using threshold 0.5
preds = (probas >= 0.5).astype(int)
```

## Streamlit Deployment (Simplified!)

Perfect for Streamlit! Just collect these 8 features from users:

```python
# In your Streamlit app
import pickle
import pandas as pd

# Load the pipeline
with open('models/pipeline.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)
with open('models/selected_features.pkl', 'rb') as fh:
    required_features = pickle.load(fh)

# Collect user inputs for the 8 features
user_inputs = []
for feature in required_features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_inputs.append(value)

# Create DataFrame with user inputs
user_data = pd.DataFrame([user_inputs], columns=required_features)

# Make prediction
if st.button("Predict"):
    prediction_proba = pipeline.predict_proba(user_data)[:, 1]
    prediction = (prediction_proba >= 0.5).astype(int)
    
    st.write(f"Probability: {prediction_proba[0]:.3f}")
    st.write(f"Prediction: {'Confirmed' if prediction[0] else 'Not Confirmed'}")
```

## Complete Self-Contained Validation Cell (Copy-Paste Ready)

```python
# STANDALONE VALIDATION CELL - Simplified 8-feature pipeline
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, log_loss

# Define label mapping (crucial step that must be repeated)
label_map = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
    "REFUTED": 0,
    "FA": 0,
}

# Define the 8 features we're using
selected_features = [
    "transit_epoch_bjd",
    "transit_depth_ppm", 
    "equilibrium_temp_k",
    "impact_parameter",
    "stellar_teff_k",
    "stellar_mass_msun",
    "stellar_logg",
    "radius_ratio_est"
]

def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw exoplanet data to supervised learning format with binary labels."""
    lbl = df["disposition"].map(label_map)
    mask = lbl.notna()
    sup_df = df.loc[mask].copy()
    sup_df["label"] = lbl.loc[mask].astype(int)
    return sup_df

def collect_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Collect comprehensive classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
    }

# Load the validation dataset
df_validation = pd.read_csv('data/unified_exoplanets_final_imputed.csv')

# Load the pipeline
with open('models/pipeline.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)

# Check if validation dataset has the required features
missing_features = [f for f in selected_features if f not in df_validation.columns]
if missing_features:
    print(f"Error: Missing features in validation dataset: {missing_features}")
    print(f"Required features: {selected_features}")
else:
    # Prepare validation data
    val_sup_df = build_supervised(df_validation)
    
    # Extract the 8 required features and labels
    X_val = val_sup_df[selected_features].fillna(val_sup_df[selected_features].median())
    y_val = val_sup_df["label"].astype(int)
    
    # Make predictions using the simplified pipeline
    y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Validation Dataset Shape: {df_validation.shape}")
    print(f"Validation Samples with Labels: {len(val_sup_df)}")
    print(f"Features Used: {len(selected_features)}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")
    
    # Additional metrics
    val_metrics = collect_metrics(y_val, y_val_pred, y_val_pred_proba)
    print(f"\nDetailed Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
```

## Metadata (models/metadata.json)
The metadata contains the following keys:
- `timestamp`: training finish time (UTC)
- `config`: configuration used for training (top_k, cv, score, etc.)
- `data`: dataset counts and label ratios
- `features`: original / filtered / selected lists and ranking with F-scores
- `resampling`: resampling method and resampled sizes
- `tuning`: best hyperparameters and CV train/val scores for each base learner
- `artifacts`: path and file sizes for saved PKL files
- `requirements`: contents of `requirements.txt` used for reproducibility
- `python_version`: minimal Python version recommended
- `model_version`: branch/timestamp tag identifying this build

## Notes
- Pipeline expects exactly 8 features in the specified order
- No feature selection step - pipeline works directly with the 8 selected features
- Perfect for Streamlit deployment - just collect these 8 values from users
- Pipeline handles: scaling → prediction (no selector needed)

