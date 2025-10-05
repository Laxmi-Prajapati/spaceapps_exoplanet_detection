import joblib
import os

# --- Configuration ---
# Determine models directory
base = os.path.abspath(os.getcwd())
models_dir = os.path.join(base, "..", 'models')
# Check current directory's 'models' first
if not os.path.exists(models_dir):
    # Check parent directory's 'models' if not found
    potential_models_dir = os.path.join(base, '..', 'models')
    if os.path.exists(potential_models_dir):
        models_dir = potential_models_dir
    else:
        # If neither path exists, default back to the first one and rely on load to fail
        pass 
        
print('models_dir ->', models_dir)

model_path = os.path.join(models_dir, 'stacking_classifier.pkl')
scaler_path = os.path.join(models_dir, 'scaler.pkl')
selector_path = os.path.join(models_dir, 'feature_selector.pkl')

print(f"Attempting to load files with old numpy ({joblib.__version__}) and scikit-learn...")

# Load with the old environment
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    print("\nSUCCESS: All files loaded in the older environment!")

    # --- RESAVING IS THE CRITICAL STEP ---
    # The objects are now successfully deserialized into Python objects 
    # compatible with this environment. Now we save them back, which 
    # serializes them using the current (1.16.6) NumPy structure.

    # Rename the new files so we don't accidentally rely on the old ones later.
    joblib.dump(model, os.path.join(models_dir, 'stacking_classifier_FIXED.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_FIXED.pkl'))
    joblib.dump(selector, os.path.join(models_dir, 'selector_FIXED.pkl'))

    print("\nSUCCESS: Files re-saved with '_FIXED' suffix for future compatibility.")

except Exception as e:
    print("\nFATAL ERROR: Loading failed even with the older libraries.")
    print(f"Error details: {e}")
    print("This means the original environment was even older, or there is another dependency conflict.")
    print("If this fails, you MUST revert to retraining the model.")