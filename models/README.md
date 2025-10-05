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

