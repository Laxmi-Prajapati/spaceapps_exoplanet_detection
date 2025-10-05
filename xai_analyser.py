import os
import pickle
import json
import pandas as pd
import numpy as np
import joblib
import shap
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

# --- 1. CONFIGURATION & CONSTANTS ---
# Use an explicit path resolution approach for robustness
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR,'models')
DATA_DIR = os.path.join(BASE_DIR, 'data') # Assuming data is in a sibling 'data' folder

MODEL_PATHS = {
    'classifier': os.path.join(MODELS_DIR, 'stacking_classifier_FIXED.pkl'),
    'scaler': os.path.join(MODELS_DIR, 'scaler_FIXED.pkl'),
    'selector': os.path.join(MODELS_DIR, 'selector_FIXED.pkl'),
    'train_data': os.path.join(DATA_DIR, 'unified_exoplanets_final_imputed.csv')
}

GEMINI_CONFIG = {
    'MODEL_NAME': 'gemini-2.0-flash', 
    'TOP_K': 8,
    'MAX_TOKENS': 768,
    'TEMPERATURE': 0.15,
    'API_OUTPUT_FILE': os.path.join(BASE_DIR, 'output', 'xai_explanation.txt'),
    'FALLBACK_PROMPT_FILE': os.path.join(BASE_DIR, 'output', 'xai_prompt.json')
}

FINAL_SCALER_FEATURES = [
    "orbital_period_days", "transit_epoch_bjd", "transit_duration_hours",
    "transit_depth_ppm", "planet_radius_re", "equilibrium_temp_k",
    "insolation_flux", "impact_parameter", "stellar_teff_k",
    "stellar_radius_rsun", "stellar_radius_normal", "stellar_mass_msun",
    "mass_rad_ratio", "stellar_logg", "acc_grav_stellar_surface",
    "ra", "dec", "radius_ratio_est"
]

# --- 2. MODEL LOADING & PREDICTION FUNCTIONS ---

def load_models():
    """Loads all required model artifacts using joblib."""
    print("[*] Loading Model Artifacts...")
    models = {}
    for name, path in MODEL_PATHS.items():
        if name != 'train_data':
            try:
                # The notebook used relative paths like '../models/', we check both.
                try:
                    models[name] = joblib.load(path)
                except FileNotFoundError:
                    # Try parent directory relative path as used in the notebook
                    models[name] = joblib.load(os.path.join(os.path.dirname(path), '..', os.path.basename(path)))
                print(f"  - Successfully loaded {name}")
            except Exception as e:
                raise IOError(f"Error loading {name} from {path}. Check file existence and path resolution: {e}")
    
    # Load selected features if available (notebook logic)
    selector = models.get('selector')
    if selector and hasattr(selector, 'get_support'):
        support = selector.get_support(indices=True)
        feature_names_selected = [FINAL_SCALER_FEATURES[i] for i in support]
        models['selected_features'] = feature_names_selected
        print(f"  - Identified {len(feature_names_selected)} features via selector.")
    else:
        # Fallback if selector doesn't provide feature support (unlikely but safe)
        models['selected_features'] = FINAL_SCALER_FEATURES 
        
    return models

def predict_exoplanet_robust(data: pd.DataFrame, models: dict):
    """Applies preprocessing and makes the final prediction."""
    
    X_processed = data.copy()
    
    # 1. Feature Engineering (as defined in the notebook)
    X_processed.loc[:, 'acc_grav_stellar_surface'] = 10**X_processed['stellar_logg']
    
    # 2. Select the FINAL 18 Features in the CORRECT ORDER and handle missing values
    X_final = X_processed.reindex(columns=FINAL_SCALER_FEATURES).fillna(0)
    
    # 3. Apply Pipeline Steps
    X_scaled = models['scaler'].transform(X_final)
    X_selected = models['selector'].transform(X_scaled)
    
    # 4. Predict
    # Predict probability for the positive class (index 1)
    pred_proba = float(models['classifier'].predict_proba(X_selected)[:, 1][0])
    pred_label = int(pred_proba >= 0.5)
    
    print(f"\n--- Prediction Results ---")
    print(f"Prediction: label={pred_label}, prob={pred_proba:.4f}")

    return X_scaled, X_selected, pred_label, pred_proba

# --- 3. SHAP COMPUTATION ---

def compute_shap(X_selected, models):
    """Computes SHAP values using TreeExplainer or robust KernelExplainer fallback."""
    
    classifier = models['classifier']
    
    # --- 3a. TreeExplainer Attempt (Fastest) ---
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_selected)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] # Take the SHAP values for the positive class (Label 1)
        shap_for_sample = np.array(shap_vals[0]).flatten()
        print("\n[*] SHAP: Used TreeExplainer.")
        
    # --- 3b. KernelExplainer Fallback (Robust but Slow) ---
    except Exception as e:
        print(f"\n[*] SHAP: TreeExplainer failed ({e}), falling back to KernelExplainer...")
        
        # Load and preprocess the training data for the background dataset
        try:
            df_train = pd.read_csv(MODEL_PATHS['train_data'])
        except FileNotFoundError:
            raise FileNotFoundError(f"KernelExplainer requires training data, but file not found at: {MODEL_PATHS['train_data']}")
            
        df_train.loc[:, 'acc_grav_stellar_surface'] = 10**df_train['stellar_logg']
        X_train_final = df_train[FINAL_SCALER_FEATURES].fillna(0)
        X_train_scaled = models['scaler'].transform(X_train_final)
        X_train_selected = models['selector'].transform(X_train_scaled)
        
        # Create background sample
        background = shap.sample(X_train_selected, min(100, X_train_selected.shape[0]))
        
        # Run Kernel Explainer
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_vals = explainer.shap_values(X_selected, nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_for_sample = np.array(shap_vals[0]).flatten()
        print("[*] SHAP: Used KernelExplainer (Computation Complete).")

    # --- 3c. Post-processing and Mapping ---
    feature_names_selected = models['selected_features']
    n_expected_features = len(feature_names_selected)
    
    # Handle the length mismatch issue seen in the notebook output
    if len(shap_for_sample) != n_expected_features:
        print(f"WARNING: SHAP array length ({len(shap_for_sample)}) does not match expected feature count ({n_expected_features}). Slicing array.")
        shap_for_sample = shap_for_sample[:n_expected_features]
    
    shap_series = pd.Series(shap_for_sample, index=feature_names_selected).sort_values(ascending=False, key=abs)
    
    return shap_series

# --- 4. GEMINI API COMMUNICATION ---

def create_shap_summary(shap_series: pd.Series, pred_label: int, pred_proba: float, test_sample: pd.DataFrame) -> dict:
    """Prepares the quantitative SHAP summary dictionary for the prompt."""
    
    df_shap = pd.DataFrame({'feature': shap_series.index, 'shap_value': shap_series.values})
    df_shap['abs_shap'] = df_shap['shap_value'].abs()
    df_shap = df_shap.sort_values('abs_shap', ascending=False).reset_index(drop=True)
    
    total_abs = df_shap['abs_shap'].sum()
    if total_abs == 0: total_abs = 1.0
        
    df_shap['pct_contrib'] = df_shap['abs_shap'] / total_abs * 100.0

    def safe_value(f):
        # Extract the original feature value from the test sample
        try:
            return float(test_sample[f].values[0])
        except Exception:
            # Note: This returns NaN for 'acc_grav_stellar_surface' as expected
            return np.nan 

    df_shap['value'] = df_shap['feature'].apply(safe_value)
    
    # Print the final contribution table before sending to Gemini
    print('\nTop quantitative contributors (Sent to Gemini):')
    print(df_shap.head(GEMINI_CONFIG['TOP_K'])[['feature','value','shap_value','abs_shap','pct_contrib']].to_string(index=False))

    summary_stats = {
        'prediction': {'label': int(pred_label), 'probability': float(pred_proba)},
        'num_reported_features': int(min(GEMINI_CONFIG['TOP_K'], len(df_shap))),
        'total_abs_shap': float(total_abs),
        'positive_shap_sum': float(df_shap[df_shap['shap_value']>0]['shap_value'].sum()),
        'negative_shap_sum': float(df_shap[df_shap['shap_value']<0]['shap_value'].sum())
    }

    structured_data = {
        'dataset': 'unified_exoplanets_final_imputed.csv',
        'model': 'stacking ensemble (RF+GB+SVM+LR) with feature selection + scaling',
        'prediction': summary_stats['prediction'],
        'quantitative_shap_top': df_shap.head(GEMINI_CONFIG['TOP_K']).to_dict(orient='records'),
        'summary_stats': summary_stats
    }
    
    return structured_data, df_shap.head(GEMINI_CONFIG['TOP_K'])

def construct_production_prompt(structured_data: dict) -> str:
    """Constructs the final prompt for Gemini."""
    
    human_instructions = (
        'You are a senior ML engineer. Given the quantitative SHAP analysis (JSON) and the sample feature values, '
        'produce a concise technical explanation (5-8 bullet points) describing why the model produced the prediction, '
        'include likely causes, model confidence caveats, and 3 concrete suggestions to validate or improve model reliability. '
        'Reference the top contributing features and their directional effects. Keep the explanation technical and targeted to a data-science audience.'
    )
    
    data_preamble = "Analyze the following machine learning prediction data and SHAP values and return the explanation for the prediction:"
    data_json = json.dumps(structured_data, indent=2)

    prompt = (
        f"{data_preamble}\n\n"
        f"--- SHAP DATA START ---\n{data_json}\n--- SHAP DATA END ---\n\n"
        f"{human_instructions}"
    )
    return prompt

def generate_xai_explanation(prompt: str) -> str:
    """Initializes the Gemini client and generates the explanation."""
    
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        return "ERROR: API Key Missing."

    try:
        client = genai.Client()
        
        print(f"\n[*] Sending data to {GEMINI_CONFIG['MODEL_NAME']} for analysis (using SDK)...")

        response = client.models.generate_content(
            model=GEMINI_CONFIG['MODEL_NAME'],
            contents=prompt,
            config={
                'max_output_tokens': GEMINI_CONFIG['MAX_TOKENS'],
                'temperature': GEMINI_CONFIG['TEMPERATURE']
            }
        )
        
        explanation_text = getattr(response, "text", None)

        if not explanation_text:
            print(f"ERROR: Model returned an empty response. Check safety settings or content filters. Response: {response}")
            return "ERROR: Empty response from model."

        return explanation_text

    except APIError as e:
        return f"ERROR: API Call Failed - {e}"

    except Exception as e:
        return f"ERROR: Unexpected Error - {e}"

# --- 5. MAIN EXECUTION ---

def main():
    """The main execution pipeline."""
    
    # 0. Setup Environment
    load_dotenv() # Load .env file
    if not os.environ.get('GEMINI_API_KEY'):
        print("\nFATAL: GEMINI_API_KEY is not set. Please set the environment variable and retry.")
        return
    
    os.makedirs(os.path.dirname(GEMINI_CONFIG['API_OUTPUT_FILE']), exist_ok=True)
    
    # NOTE: In a production environment, this sample_data should be loaded from a request/queue.
    # Using the notebook's sample data here for continuity.
    sample_data = {
        "orbital_period_days": [5.72], "transit_epoch_bjd": [2457000.12345],
        "transit_duration_hours": [2.1], "transit_depth_ppm": [1300.0],
        "planet_radius_re": [1.12], "equilibrium_temp_k": [1100.0],
        "insolation_flux": [800.0], "impact_parameter": [0.45],
        "stellar_teff_k": [5700.0], "stellar_radius_rsun": [0.98],
        "stellar_radius_normal": [1.0], "stellar_mass_msun": [1.02],
        "mass_rad_ratio": [1.04], "stellar_logg": [4.38],
        "ra": [299.123], "dec": [45.789], "radius_ratio_est": [0.011]
    }
    test_sample = pd.DataFrame(sample_data)

    try:
        # 1. Load Models
        models = load_models()
        
        # 2. Predict
        X_scaled, X_selected, pred_label, pred_proba = predict_exoplanet_robust(test_sample, models)
        
        # 3. Compute SHAP
        shap_series = compute_shap(X_selected, models)
        
        # 4. Prepare Gemini Data and Prompt
        structured_data, df_top_shap = create_shap_summary(shap_series, pred_label, pred_proba, test_sample)
        prompt = construct_production_prompt(structured_data)

        # 5. Generate Explanation
        explanation = generate_xai_explanation(prompt)
        
    except (IOError, FileNotFoundError, Exception) as e:
        print(f"\nFATAL PIPELINE ERROR: {e}")
        return

    # 6. Handle Output
    if explanation.startswith("ERROR:"):
        print("\n--- FATAL API ERROR ---")
        print(explanation)
        
        # Save prompt as a manual fallback
        with open(GEMINI_CONFIG['FALLBACK_PROMPT_FILE'], 'w', encoding='utf-8') as pf:
            json.dump({'prompt': prompt, 'structured': structured_data}, pf, indent=2)
        print(f"\nPrompt saved for manual analysis: {GEMINI_CONFIG['FALLBACK_PROMPT_FILE']}")
        
    else:
        print('\n--- FINAL XAI EXPLANATION GENERATED ---')
        # Print the beginning of the explanation for quick view
        print("\n--- PROMPT JSON ---")
        with open(GEMINI_CONFIG['FALLBACK_PROMPT_FILE'], 'w', encoding='utf-8') as pf:
            json.dump({'prompt': prompt, 'structured': structured_data}, pf, indent=2)
        print(f"\nPrompt saved for manual analysis: {GEMINI_CONFIG['FALLBACK_PROMPT_FILE']}\n")

        print(explanation[:800] + '...') 
        # Save the successful output
        with open(GEMINI_CONFIG['API_OUTPUT_FILE'], 'w', encoding='utf-8') as of:
            of.write(explanation)
        print(f"\nSuccessfully saved explanation to {GEMINI_CONFIG['API_OUTPUT_FILE']}")

if __name__ == '__main__':
    main()