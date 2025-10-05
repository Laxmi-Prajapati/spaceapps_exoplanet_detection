# Exoplanet Stacking Model

- **ROC AUC (test)**: 0.954
- **F1 (test)**: 0.823
- **Selected features** (8): transit_epoch_bjd, transit_depth_ppm, equilibrium_temp_k, impact_parameter, stellar_teff_k, stellar_mass_msun, stellar_logg, radius_ratio_est
- **Filtered features** (16): orbital_period_days, transit_epoch_bjd, transit_duration_hours, transit_depth_ppm, equilibrium_temp_k, insolation_flux, impact_parameter, stellar_teff_k, stellar_radius_rsun, stellar_mass_msun, mass_rad_ratio, stellar_logg, acc_grav_stellar_surface, ra, dec, radius_ratio_est

## Artifacts (PKL files)

- `pipeline.pkl`: A single scikit-learn `Pipeline` that includes the fitted `SelectKBest` selector, `StandardScaler`, and the final stacking classifier. Use this for inference: it expects a `pandas.DataFrame` with the columns listed in "Filtered features" in the same order.
- `filtered_features.pkl`: Pickled `list` of feature names (the columns the pipeline expects). Load this to validate or pre-select columns before passing data to the pipeline.
- `selected_features.pkl`: Pickled `list` of the top-k features selected (these are the features after SelectKBest).

## Quick predict (recommended)

```python
import pickle
import pandas as pd

# Load artifacts
with open('models/pipeline.pkl', 'rb') as fh:
    pipe = pickle.load(fh)
with open('models/filtered_features.pkl', 'rb') as fh:
    flt = pickle.load(fh)

# df is your input DataFrame containing at least the filtered features
# Ensure dtype consistency (numeric columns)
X = df[flt]
probas = pipe.predict_proba(X)[:, 1]

# Convert to class predictions using threshold 0.5
preds = (probas >= 0.5).astype(int)
```

## Complete Self-Contained Validation Cell (Copy-Paste Ready)

Use this exact code as a standalone notebook cell or script. This is the complete self-contained validation cell from the main notebook:

```python
# STANDALONE VALIDATION CELL - Self-contained validation that doesn't depend on previous cells
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

def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw exoplanet data to supervised learning format with binary labels.
  
    This function is crucial and must be repeated for any validation dataset
    to ensure consistent label encoding.
    """
    lbl = df["disposition"].map(label_map)
    mask = lbl.notna()
    sup_df = df.loc[mask].copy()
    sup_df["label"] = lbl.loc[mask].astype(int)
    return sup_df

def collect_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Collect comprehensive classification metrics.
  
    This function is crucial for consistent metric calculation across
    training and validation phases.
    """
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

# Load the required artifacts
with open('models/pipeline.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)
with open('models/filtered_features.pkl', 'rb') as fh:
    filtered_features = pickle.load(fh)

# Check if the validation dataset has the required features
missing_features = [f for f in filtered_features if f not in df_validation.columns]
if missing_features:
    print(f"Warning: Missing features in validation dataset: {missing_features}")

# Prepare validation data using the same preprocessing as training (CRUCIAL STEP)
val_sup_df = build_supervised(df_validation)
available_features = [f for f in filtered_features if f in val_sup_df.columns]

# Extract features and labels
X_val = val_sup_df[available_features].fillna(val_sup_df[available_features].median())
y_val = val_sup_df["label"].astype(int)

# Make predictions using the pipeline
y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# Calculate accuracy
validation_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Dataset Shape: {df_validation.shape}")
print(f"Validation Samples with Labels: {len(val_sup_df)}")
print(f"Available Features: {len(available_features)}/{len(filtered_features)}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")

# Additional metrics
val_metrics = collect_metrics(y_val, y_val_pred, y_val_pred_proba)
print(f"\nDetailed Validation Metrics:")
for metric, value in val_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

**Note**: Change the CSV path `'data/unified_exoplanets_final_imputed.csv'` to your validation dataset path.

## Streamlit Deployment Notes

**IMPORTANT for Streamlit**: You absolutely need the selector in your pipeline! Here's why:

- Your pipeline expects **16 filtered features** (from `filtered_features.pkl`)
- But your model was trained on only **8 selected features**
- The selector reduces 16 → 8 features before scaling and prediction
- Without the selector, you'll get shape mismatch errors

For Streamlit, use this exact workflow:

```python
# In your Streamlit app
import pickle
import pandas as pd

# Load the complete pipeline (includes selector, scaler, classifier)
with open('models/pipeline.pkl', 'rb') as fh:
    pipeline = pickle.load(fh)
with open('models/filtered_features.pkl', 'rb') as fh:
    filtered_features = pickle.load(fh)

# Your user input should provide the 16 filtered features
user_data = pd.DataFrame([{feat: value for feat, value in zip(filtered_features, user_inputs)}])

# Pipeline handles: feature selection (16→8) → scaling → prediction
prediction_proba = pipeline.predict_proba(user_data)[:, 1]
prediction = (prediction_proba >= 0.5).astype(int)
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

- Always pass the `DataFrame` columns in the same order as `filtered_features.pkl`.
- The `pipeline.pkl` encapsulates selection and scaling. Do not apply extra scaling or selection before calling `pipe.predict_proba`.
- The selector is CRUCIAL for deployment - it reduces 16 features to 8 as expected by the model.
