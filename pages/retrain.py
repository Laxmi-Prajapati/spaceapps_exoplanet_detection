import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from io import BytesIO
from datetime import datetime
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

def show_retrain_page():
    """Display the Retrain page content"""
    
    st.header("🔄 Model Retraining")
    st.markdown("Upload your data to retrain the exoplanet detection model with additional samples.")
    
    # Define the required features
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
    
    # Configuration from the notebook
    cfg = {
        "seed": 42,
        "top_k": 8,
        "test_split": 0.2,
        "cv": 5,
        "scoring": "roc_auc",
    }
    
    # Label mapping
    label_map = {
        "CONFIRMED": 1,
        "FALSE POSITIVE": 0,
        "REFUTED": 0,
        "FA": 0,
        "CANDIDATE": 0,
    }
    
    st.subheader("📋 Data Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required Columns (9 total):**")
        st.markdown("**Features (8):**")
        for i, feature in enumerate(selected_features, 1):
            st.write(f"{i}. `{feature}`")
    
    with col2:
        st.markdown("**Target Column (1):**")
        st.write("9. `disposition` - Must contain values:")
        st.write("   - CONFIRMED")
        st.write("   - FALSE POSITIVE") 
        st.write("   - REFUTED")
        st.write("   - CANDIDATE")
    
    st.warning("⚠️ Your CSV must have exactly these 9 columns with exact column names (case-sensitive).")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with 8 feature columns + 1 disposition column"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            user_df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Uploaded Data Preview")
            st.write(f"Shape: {user_df.shape}")
            st.dataframe(user_df.head())
            
            # Validate the data
            required_columns = selected_features + ["disposition"]
            
            # Check columns
            if len(user_df.columns) != 9:
                st.error(f"CSV must have exactly 9 columns. Found {len(user_df.columns)} columns.")
                return
            
            if list(user_df.columns) != required_columns:
                missing_cols = [col for col in required_columns if col not in user_df.columns]
                extra_cols = [col for col in user_df.columns if col not in required_columns]
                
                st.error("Column names do not match required format.")
                st.write(f"**Required:** {required_columns}")
                st.write(f"**Found:** {list(user_df.columns)}")
                if missing_cols:
                    st.write(f"**Missing:** {missing_cols}")
                if extra_cols:
                    st.write(f"**Extra/incorrect:** {extra_cols}")
                return
            
            # Check if features are numeric
            for col in selected_features:
                if not pd.api.types.is_numeric_dtype(user_df[col]):
                    st.error(f"Column '{col}' is not numeric. All feature columns must be int or float.")
                    return
            
            # Check disposition values
            valid_dispositions = set(label_map.keys())
            user_dispositions = set(user_df['disposition'].unique())
            invalid_dispositions = user_dispositions - valid_dispositions
            
            if invalid_dispositions:
                st.error(f"Invalid disposition values found: {invalid_dispositions}")
                st.write(f"Valid values are: {list(valid_dispositions)}")
                return
            
            st.success("✅ Data validation successful!")
            
            # Show data statistics
            st.subheader("📈 Data Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(user_df))
            
            disposition_counts = user_df['disposition'].value_counts()
            confirmed_count = disposition_counts.get('CONFIRMED', 0)
            total_count = len(user_df)
            
            with col2:
                st.metric("Confirmed Exoplanets", confirmed_count)
            
            with col3:
                positive_ratio = confirmed_count / total_count if total_count > 0 else 0
                st.metric("Positive Ratio", f"{positive_ratio:.3f}")
            
            st.write("**Disposition Distribution:**")
            st.write(disposition_counts)
            
            # Hyperparameter customization section
            st.subheader("⚙️ Model Configuration")
            
            # Toggle for hyperparameter customization
            use_custom_params = st.toggle(
                "Customize Hyperparameters", 
                value=False,
                help="Toggle ON to customize model hyperparameters, or keep OFF to use pre-tuned optimal values"
            )
            
            if use_custom_params:
                st.info("🔧 **Custom Mode**: Adjust hyperparameters below. Advanced users can fine-tune for their specific data.")
                
                # Create tabs for different models
                tab1, tab2, tab3, tab4 = st.tabs(["🌲 Random Forest", "📈 Gradient Boosting", "🎯 SVM", "📊 Logistic Regression"])
                
                with tab1:
                    st.subheader("Random Forest Parameters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        rf_n_estimators = st.slider("Number of Trees", 50, 200, 120, 10, 
                                                   help="More trees = better performance but slower training")
                        rf_max_depth = st.slider("Maximum Depth", 3, 20, 10, 1,
                                                help="Deeper trees can overfit, shallower trees may underfit")
                        rf_min_samples_split = st.slider("Min Samples Split", 2, 20, 4, 1,
                                                        help="Minimum samples required to split a node")
                    
                    with col2:
                        rf_min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 2, 1,
                                                       help="Minimum samples required at each leaf node")
                        rf_max_features = st.slider("Max Features Ratio", 0.1, 1.0, 0.7, 0.1,
                                                   help="Fraction of features to consider for each split")
                
                with tab2:
                    st.subheader("Gradient Boosting Parameters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        gb_n_estimators = st.slider("Number of Boosting Stages", 50, 200, 120, 10,
                                                   help="More stages = better performance but risk of overfitting")
                        gb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.06, 0.01,
                                                    help="Lower rates need more estimators but often perform better")
                        gb_max_depth = st.slider("Maximum Depth", 3, 10, 5, 1,
                                                help="Depth of individual trees in the ensemble")
                    
                    with col2:
                        gb_min_samples_split = st.slider("Min Samples Split", 2, 20, 8, 1,
                                                        help="Minimum samples required to split a node")
                        gb_subsample = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1,
                                                help="Fraction of samples used for fitting individual trees")
                
                with tab3:
                    st.subheader("Support Vector Machine Parameters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        svm_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0,
                                                 help="rbf: good for non-linear data, linear: faster and interpretable")
                        svm_C = st.slider("Regularization (C)", 0.01, 10.0, 0.5, 0.01,
                                         help="Higher C = less regularization (may overfit)")
                    
                    with col2:
                        if svm_kernel in ["rbf", "poly"]:
                            svm_gamma = st.selectbox("Gamma", ["scale", "auto"], index=0,
                                                   help="scale: 1/(n_features * X.var()), auto: 1/n_features")
                        else:
                            svm_gamma = "scale"
                
                with tab4:
                    st.subheader("Logistic Regression Parameters")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        lr_C = st.slider("Regularization (C)", 0.01, 10.0, 0.5, 0.01,
                                        help="Higher C = less regularization")
                        lr_penalty = st.selectbox("Penalty", ["l1", "l2"], index=0,
                                                 help="l1: feature selection, l2: prevents overfitting")
                    
                    with col2:
                        if lr_penalty == "l1":
                            lr_solver = "liblinear"
                            st.info("Solver: liblinear (optimal for l1 penalty)")
                        else:
                            lr_solver = st.selectbox("Solver", ["liblinear", "lbfgs"], index=1,
                                                   help="lbfgs: faster for small datasets, liblinear: good for large datasets")
                
                # Display current custom parameters
                st.subheader("📋 Current Configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json({
                        "Random Forest": {
                            "n_estimators": rf_n_estimators,
                            "max_depth": rf_max_depth,
                            "min_samples_split": rf_min_samples_split,
                            "min_samples_leaf": rf_min_samples_leaf,
                            "max_features": rf_max_features
                        },
                        "Gradient Boosting": {
                            "n_estimators": gb_n_estimators,
                            "learning_rate": gb_learning_rate,
                            "max_depth": gb_max_depth,
                            "min_samples_split": gb_min_samples_split,
                            "subsample": gb_subsample
                        }
                    })
                
                with col2:
                    svm_params = {
                        "kernel": svm_kernel,
                        "C": svm_C,
                        "gamma": svm_gamma
                    }
                    lr_params = {
                        "C": lr_C,
                        "penalty": lr_penalty,
                        "solver": lr_solver
                    }
                    
                    st.json({
                        "SVM": svm_params,
                        "Logistic Regression": lr_params
                    })
                
                # Build custom parameters dictionary
                custom_params = {
                    "rf": {
                        'n_estimators': rf_n_estimators,
                        'max_depth': rf_max_depth,
                        'min_samples_split': rf_min_samples_split,
                        'min_samples_leaf': rf_min_samples_leaf,
                        'max_features': rf_max_features
                    },
                    "gb": {
                        'n_estimators': gb_n_estimators,
                        'learning_rate': gb_learning_rate,
                        'max_depth': gb_max_depth,
                        'min_samples_split': gb_min_samples_split,
                        'subsample': gb_subsample
                    },
                    "svm": svm_params,
                    "lr": lr_params
                }
                
                st.warning("⚠️ **Note**: Custom hyperparameters may require longer training time and could impact model performance. Pre-tuned parameters are optimized for general exoplanet detection tasks.")
            
            else:
                st.success("✅ **Pre-tuned Mode**: Using optimized hyperparameters from extensive grid search validation (ROC AUC: 96.64%)")
                
                # Show pre-tuned parameters in an expandable section
                with st.expander("📋 View Pre-tuned Parameters"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json({
                            "Random Forest": {
                                "n_estimators": 120,
                                "max_depth": 10,
                                "min_samples_split": 4,
                                "min_samples_leaf": 2,
                                "max_features": 0.7
                            },
                            "Gradient Boosting": {
                                "n_estimators": 120,
                                "learning_rate": 0.06,
                                "max_depth": 5,
                                "min_samples_split": 8,
                                "subsample": 0.8
                            }
                        })
                    
                    with col2:
                        st.json({
                            "SVM": {
                                "kernel": "rbf",
                                "C": 0.5,
                                "gamma": "scale"
                            },
                            "Logistic Regression": {
                                "C": 0.5,
                                "penalty": "l1",
                                "solver": "liblinear"
                            }
                        })
                
                # Use pre-tuned parameters
                custom_params = {
                    "rf": {
                        'max_depth': 10, 
                        'max_features': 0.7, 
                        'min_samples_leaf': 2, 
                        'min_samples_split': 4, 
                        'n_estimators': 120
                    },
                    "gb": {
                        'learning_rate': 0.06, 
                        'max_depth': 5, 
                        'min_samples_split': 8, 
                        'n_estimators': 120, 
                        'subsample': 0.8
                    },
                    "svm": {
                        'C': 0.5, 
                        'gamma': 'scale', 
                        'kernel': 'rbf'
                    },
                    "lr": {
                        'C': 0.5, 
                        'penalty': 'l1', 
                        'solver': 'liblinear'
                    }
                }
            
            # Training button
            if st.button("🚀 Start Retraining", type="primary"):
                with st.spinner("Loading existing data and retraining model..."):
                    
                    # Load existing data
                    try:
                        existing_df = pd.read_csv('data/unified_exoplanets_final_imputed.csv')
                        st.success(f"✅ Loaded existing data: {existing_df.shape}")
                    except Exception as e:
                        st.error(f"Error loading existing data: {str(e)}")
                        return
                    
                    # Select only required columns from existing data
                    if 'disposition' not in existing_df.columns:
                        st.error("Existing dataset missing 'disposition' column")
                        return
                    
                    # Check if existing data has required features
                    missing_features = [f for f in selected_features if f not in existing_df.columns]
                    if missing_features:
                        st.error(f"Existing dataset missing features: {missing_features}")
                        return
                    
                    existing_subset = existing_df[required_columns].copy()
                    
                    # Combine datasets
                    combined_df = pd.concat([existing_subset, user_df], ignore_index=True)
                    
                    st.success(f"✅ Combined dataset size: {combined_df.shape}")
                    
                    # Show combined statistics
                    st.subheader("📊 Combined Dataset Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Existing Data", f"{len(existing_subset):,}")
                    with col2:
                        st.metric("New Data", f"{len(user_df):,}")
                    with col3:
                        st.metric("Combined Total", f"{len(combined_df):,}")
                    with col4:
                        increase_pct = (len(user_df) / len(existing_subset)) * 100
                        st.metric("Data Increase", f"{increase_pct:.1f}%")
                    
                    # Build supervised dataset
                    def build_supervised(df):
                        lbl = df["disposition"].map(label_map)
                        mask = lbl.notna()
                        sup_df = df.loc[mask].copy()
                        sup_df["label"] = lbl.loc[mask].astype(int)
                        return sup_df
                    
                    sup_df = build_supervised(combined_df)
                    
                    if len(sup_df) == 0:
                        st.error("No valid labeled samples found after combining data")
                        return
                    
                    st.write(f"**Labeled samples:** {len(sup_df)}")
                    st.write(f"**Positive ratio:** {sup_df['label'].mean():.3f}")
                    
                    # Prepare features and labels
                    X = sup_df[selected_features].fillna(sup_df[selected_features].median())
                    y = sup_df["label"].astype(int)
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=cfg["test_split"], random_state=cfg["seed"], stratify=y
                    )
                    
                    st.write(f"**Train/Test split:** {X_train.shape} / {X_test.shape}")
                    
                    # Feature scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Random oversampling
                    ros = RandomOverSampler(random_state=cfg["seed"])
                    X_train_balanced, y_train_balanced = ros.fit_resample(X_train_scaled, y_train)
                    
                    st.write(f"**Balanced samples:** {len(y_train_balanced)} (pos ratio: {y_train_balanced.mean():.3f})")
                    
                    # Define hyperparameters 
                    st.subheader("🔧 Training Models")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Use either custom or pre-tuned parameters
                    best_params = custom_params
                    
                    # Create model stats for display
                    if use_custom_params:
                        model_stats = {
                            "rf": {"params": best_params["rf"], "val": "Custom", "gap": "TBD"},
                            "gb": {"params": best_params["gb"], "val": "Custom", "gap": "TBD"},
                            "svm": {"params": best_params["svm"], "val": "Custom", "gap": "TBD"},
                            "lr": {"params": best_params["lr"], "val": "Custom", "gap": "TBD"},
                        }
                    else:
                        # Pre-tuned validation scores
                        model_stats = {
                            "rf": {"params": best_params["rf"], "val": 0.9660, "gap": 0.0145},
                            "gb": {"params": best_params["gb"], "val": 0.9663, "gap": 0.0174},
                            "svm": {"params": best_params["svm"], "val": 0.9432, "gap": 0.0017},
                            "lr": {"params": best_params["lr"], "val": 0.8364, "gap": 0.0006},
                        }
                    
                    best_models = {}
                    total_models = len(best_params)
                    
                    for i, (name, params) in enumerate(best_params.items()):
                        if use_custom_params:
                            status_text.text(f"Training {name.upper()} with custom hyperparameters...")
                        else:
                            status_text.text(f"Training {name.upper()} with pre-tuned hyperparameters...")
                        progress_bar.progress((i + 1) / (total_models + 1))
                        
                        # Create model with hyperparameters
                        if name == "rf":
                            model = RandomForestClassifier(random_state=cfg["seed"], n_jobs=-1, **params)
                        elif name == "gb":
                            model = GradientBoostingClassifier(
                                random_state=cfg["seed"], 
                                validation_fraction=0.2, 
                                n_iter_no_change=10, 
                                **params
                            )
                        elif name == "svm":
                            model = SVC(probability=True, random_state=cfg["seed"], cache_size=500, **params)
                        elif name == "lr":
                            model = LogisticRegression(random_state=cfg["seed"], max_iter=2000, **params)
                        
                        # Fit model
                        model.fit(X_train_balanced, y_train_balanced)
                        best_models[name] = model
                    
                    # Build stacking classifier
                    status_text.text("Building stacking ensemble...")
                    progress_bar.progress(1.0)
                    
                    meta = LogisticRegression(random_state=cfg["seed"], C=0.25, penalty="l2", max_iter=2000)
                    stacking_clf = StackingClassifier(
                        estimators=[(name, est) for name, est in best_models.items()],
                        final_estimator=meta,
                        cv=cfg["cv"],
                        stack_method="predict_proba",
                        n_jobs=-1,
                        passthrough=False,
                    )
                    stacking_clf.fit(X_train_balanced, y_train_balanced)
                    
                    # Create pipeline
                    pipeline = Pipeline([
                        ("scaler", scaler),
                        ("clf", stacking_clf),
                    ])
                    
                    # Fit pipeline on original features (not scaled)
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
                    y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
                    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
                    
                    # Calculate metrics
                    def collect_metrics(y_true, y_pred, y_proba):
                        return {
                            "accuracy": float(accuracy_score(y_true, y_pred)),
                            "f1": float(f1_score(y_true, y_pred)),
                            "precision": float(precision_score(y_true, y_pred)),
                            "recall": float(recall_score(y_true, y_pred)),
                            "roc_auc": float(roc_auc_score(y_true, y_proba)),
                            "avg_precision": float(average_precision_score(y_true, y_proba)),
                            "log_loss": float(log_loss(y_true, y_proba)),
                        }
                    
                    train_metrics = collect_metrics(y_train, y_train_pred, y_train_pred_proba)
                    test_metrics = collect_metrics(y_test, y_test_pred, y_test_pred_proba)
                    
                    status_text.text("Training completed!")
                    
                    # Display results
                    st.subheader("🎯 Training Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Test Metrics:**")
                        st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
                        st.metric("F1-Score", f"{test_metrics['f1']:.4f}")
                        st.metric("ROC AUC", f"{test_metrics['roc_auc']:.4f}")
                        st.metric("Precision", f"{test_metrics['precision']:.4f}")
                        st.metric("Recall", f"{test_metrics['recall']:.4f}")
                    
                    with col2:
                        st.write("**Train Metrics:**")
                        st.metric("Accuracy", f"{train_metrics['accuracy']:.4f}")
                        st.metric("F1-Score", f"{train_metrics['f1']:.4f}")
                        st.metric("ROC AUC", f"{train_metrics['roc_auc']:.4f}")
                        st.metric("Precision", f"{train_metrics['precision']:.4f}")
                        st.metric("Recall", f"{train_metrics['recall']:.4f}")
                    
                    # Model hyperparameters
                    st.subheader("🔧 Model Hyperparameters Used")
                    
                    if use_custom_params:
                        st.info("📝 **Custom hyperparameters were used for training**")
                    else:
                        st.success("✅ **Pre-tuned optimal hyperparameters were used for training**")
                    
                    for name, stats in model_stats.items():
                        st.write(f"**{name.upper()}:** {stats['params']}")
                        if not use_custom_params:
                            st.write(f"  - Validation AUC: {stats['val']:.4f}")
                            st.write(f"  - Train-Val Gap: {stats['gap']:.4f}")
                        else:
                            st.write(f"  - Configuration: Custom parameters")
                    
                    # Add performance comparison note
                    if use_custom_params:
                        st.warning("""
                        **Note**: Custom hyperparameters were used. Performance may vary from the pre-tuned baseline.
                        For optimal results, consider using the pre-tuned parameters which achieved 96.64% ROC AUC.
                        """)
                    else:
                        st.info("""
                        **Note**: Pre-tuned hyperparameters were optimized through extensive grid search and cross-validation.
                        These parameters achieved the best performance on similar exoplanet detection tasks.
                        """)
                    
                    # Confusion Matrix
                    st.subheader("📊 Confusion Matrix (Test Set)")
                    cm = confusion_matrix(y_test, y_test_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Confusion Matrix:**")
                        cm_df = pd.DataFrame(cm, 
                                           index=['Actual: False Positive', 'Actual: Confirmed'], 
                                           columns=['Pred: False Positive', 'Pred: Confirmed'])
                        st.dataframe(cm_df)
                    
                    with col2:
                        st.write("**Classification Report:**")
                        report = classification_report(y_test, y_test_pred, 
                                                     target_names=["False Positive", "Confirmed"],
                                                     output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))
                    
                    # Save model
                    st.subheader("💾 Download Retrained Model")
                    
                    # Create metadata
                    metadata = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'training_config': cfg,
                        'hyperparameter_mode': 'custom' if use_custom_params else 'pre-tuned',
                        'data_info': {
                            'existing_samples': len(existing_subset),
                            'new_samples': len(user_df),
                            'total_samples': len(combined_df),
                            'labeled_samples': len(sup_df),
                            'train_samples': len(X_train),
                            'test_samples': len(X_test),
                            'positive_ratio': float(y.mean()),
                        },
                        'features': {
                            'selected_features': selected_features,
                            'feature_count': len(selected_features)
                        },
                        'model_performance': {
                            'test_metrics': test_metrics,
                            'train_metrics': train_metrics,
                        },
                        'hyperparameters': model_stats,
                        'model_version': f"retrained-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
                    }
                    
                    # Serialize model
                    model_buffer = BytesIO()
                    pickle.dump(pipeline, model_buffer)
                    model_buffer.seek(0)
                    
                    # Serialize metadata
                    metadata_buffer = BytesIO()
                    metadata_json = json.dumps(metadata, indent=2)
                    metadata_buffer.write(metadata_json.encode('utf-8'))
                    metadata_buffer.seek(0)
                    
                    # Model size
                    model_size_mb = len(model_buffer.getvalue()) / (1024 * 1024)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download Retrained Model (.pkl)",
                            data=model_buffer.getvalue(),
                            file_name=f"retrained_exoplanet_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                            mime="application/octet-stream",
                            type="primary"
                        )
                        st.write(f"**Model size:** {model_size_mb:.2f} MB")
                    
                    with col2:
                        st.download_button(
                            label="📋 Download Training Metadata (.json)",
                            data=metadata_buffer.getvalue(),
                            file_name=f"retrained_model_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        metadata_size_kb = len(metadata_buffer.getvalue()) / 1024
                        st.write(f"**Metadata size:** {metadata_size_kb:.2f} KB")
                    
                    st.success("🎉 Model retraining completed successfully!")
                    
                    st.info("""
                    **Next Steps:**
                    1. Download the retrained model and metadata
                    2. Replace the old `pipeline.pkl` in your `models/` folder
                    3. The model is ready to use with the same 8 features
                    4. Performance metrics show how well the model learned from the combined data
                    """)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file is a valid CSV with the correct format.")
    
    # Footer information
    st.markdown("---")
    st.subheader("ℹ️ Important Notes")
    st.markdown("""
    - **Data Quality:** Ensure your data is clean and follows the same format as the training data
    - **Feature Consistency:** All 8 features must be present and numeric
    - **Disposition Values:** Must use exact values: CONFIRMED, FALSE POSITIVE, REFUTED, CANDIDATE
    - **Model Performance:** More data generally improves model performance, but quality matters more than quantity
    - **Computational Time:** Training time depends on dataset size and may take several minutes
    - **Model Compatibility:** The retrained model will have the same interface as the original model
    """)