# Exoplanet Stacking Model

- **ROC AUC (test)**: 0.954
- **F1 (test)**: 0.823
- **Selected features** (8): transit_epoch_bjd, transit_depth_ppm, equilibrium_temp_k, impact_parameter, stellar_teff_k, stellar_mass_msun, stellar_logg, radius_ratio_est
- **Oversampling**: RandomOverSampler (balanced ratio 0.50)

## Quick predict
```python
import numpy as np
import pickle
import pandas as pd

with open('models/stacking_classifier.pkl', 'rb') as fh:
    model = pickle.load(fh)
with open('models/scaler.pkl', 'rb') as fh:
    scaler = pickle.load(fh)
with open('models/feature_selector.pkl', 'rb') as fh:
    selector = pickle.load(fh)
with open('models/selected_features.pkl', 'rb') as fh:
    feats = pickle.load(fh)

def predict(df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(df[feats])
    X = selector.transform(X)
    return model.predict_proba(X)[:, 1]
```
