import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from io import BytesIO

def load_model_artifacts():
    """Load the model pipeline"""
    pipeline_path = "models/pipeline.pkl"
    
    if not os.path.exists(pipeline_path):
        st.error(f"Model file not found: {pipeline_path}")
        return None
    
    try:
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def make_prediction(pipeline, input_values):
    """Make a single prediction"""
    try:
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
        
        # Create input DataFrame with the 8 features
        input_data = pd.DataFrame([input_values], columns=selected_features)
        
        # Make prediction
        prediction_proba = pipeline.predict_proba(input_data)[0, 1]
        prediction = int(prediction_proba >= 0.5)
        
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def validate_csv_data(df):
    """Validate uploaded CSV data"""
    required_columns = [
        "transit_epoch_bjd",
        "transit_depth_ppm", 
        "equilibrium_temp_k",
        "impact_parameter",
        "stellar_teff_k",
        "stellar_mass_msun",
        "stellar_logg",
        "radius_ratio_est"
    ]
    
    # Check if it has exactly 8 columns
    if len(df.columns) != 8:
        return False, f"CSV must have exactly 8 columns. Found {len(df.columns)} columns."
    
    # Check if column names match exactly
    df_columns = df.columns.tolist()
    if df_columns != required_columns:
        missing_cols = [col for col in required_columns if col not in df_columns]
        extra_cols = [col for col in df_columns if col not in required_columns]
        
        error_msg = "Column names do not match required format.\n"
        error_msg += f"Required: {required_columns}\n"
        error_msg += f"Found: {df_columns}\n"
        
        if missing_cols:
            error_msg += f"Missing columns: {missing_cols}\n"
        if extra_cols:
            error_msg += f"Extra/incorrect columns: {extra_cols}"
            
        return False, error_msg
    
    # Check if all columns are numeric
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' is not numeric. All columns must be int or float."
    
    return True, "CSV validation successful!"

def make_batch_predictions(pipeline, df):
    """Make predictions for batch data"""
    try:
        # The columns should already match the required feature names after validation
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
        
        # Use the DataFrame directly since it has the correct columns
        input_data = df[selected_features]
        
        # Make predictions
        prediction_probas = pipeline.predict_proba(input_data)[:, 1]
        predictions = (prediction_probas >= 0.5).astype(int)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['prediction'] = predictions
        results_df['confidence_score'] = prediction_probas
        results_df['predicted_class'] = ['CONFIRMED EXOPLANET' if p == 1 else 'FALSE POSITIVE' for p in predictions]
        
        return results_df
    except Exception as e:
        st.error(f"Error making batch predictions: {str(e)}")
        return None

def show_predict_page():
    """Display the Predict page content"""
    
    st.header("🔮 Exoplanet Prediction")
    st.markdown("Choose your prediction method: manual input or batch prediction with CSV upload.")
    
    # Load model artifacts
    pipeline = load_model_artifacts()
    if pipeline is None:
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Create tabs for different prediction methods
    tab1, tab2 = st.tabs(["🎛️ Manual Input", "📊 Batch Prediction"])
    
    with tab1:
        st.subheader("Manual Input")
        st.markdown("Enter the planetary and stellar parameters below:")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            transit_epoch_bjd = st.number_input(
                "Transit Epoch (BJD)",
                min_value=0.0,
                value=2454833.0,
                step=0.1,
                format="%.6f",
                help="Barycentric Julian Date of transit center"
            )
            
            transit_depth_ppm = st.number_input(
                "Transit Depth (ppm)",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                format="%.2f",
                help="Depth of transit in parts per million"
            )
            
            equilibrium_temp_k = st.number_input(
                "Equilibrium Temperature (K)",
                min_value=0.0,
                value=1500.0,
                step=10.0,
                format="%.2f",
                help="Planet equilibrium temperature in Kelvin"
            )
            
            impact_parameter = st.number_input(
                "Impact Parameter",
                min_value=0.0,
                value=0.5,
                step=0.01,
                format="%.4f",
                help="Orbital geometry parameter"
            )
        
        with col2:
            stellar_teff_k = st.number_input(
                "Stellar Effective Temperature (K)",
                min_value=0.0,
                value=5778.0,
                step=50.0,
                format="%.2f",
                help="Effective temperature of the host star"
            )
            
            stellar_mass_msun = st.number_input(
                "Stellar Mass (Solar Masses)",
                min_value=0.0,
                value=1.0,
                step=0.01,
                format="%.4f",
                help="Mass of the host star in solar masses"
            )
            
            stellar_logg = st.number_input(
                "Stellar Log(g)",
                min_value=0.0,
                value=4.44,
                step=0.01,
                format="%.4f",
                help="Logarithm of surface gravity (cm/s²)"
            )
            
            radius_ratio_est = st.number_input(
                "Radius Ratio (Rp/R*)",
                min_value=0.0,
                value=0.1,
                step=0.001,
                format="%.6f",
                help="Estimated planet-to-star radius ratio"
            )
        
        # Prediction button
        if st.button("🚀 Predict Exoplanet", type="primary"):
            input_values = [
                transit_epoch_bjd,
                transit_depth_ppm,
                equilibrium_temp_k,
                impact_parameter,
                stellar_teff_k,
                stellar_mass_msun,
                stellar_logg,
                radius_ratio_est
            ]
            
            prediction, prediction_proba = make_prediction(pipeline, input_values)
            
            if prediction is not None:
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction", 
                        "CONFIRMED EXOPLANET" if prediction == 1 else "FALSE POSITIVE",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Confidence Score", 
                        f"{prediction_proba:.4f}",
                        delta=None
                    )
                
                with col3:
                    confidence_percentage = prediction_proba * 100 if prediction == 1 else (1 - prediction_proba) * 100
                    st.metric(
                        "Confidence", 
                        f"{confidence_percentage:.2f}%",
                        delta=None
                    )
                
                # Progress bar for confidence
                st.subheader("Confidence Visualization")
                if prediction == 1:
                    st.success(f"The model predicts this is a CONFIRMED EXOPLANET with {confidence_percentage:.2f}% confidence")
                    st.progress(prediction_proba)
                else:
                    st.warning(f"The model predicts this is a FALSE POSITIVE with {confidence_percentage:.2f}% confidence")
                    st.progress(1 - prediction_proba)
    
    with tab2:
        st.subheader("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with exactly 8 columns with the following **exact** column names:")
        
        # Display expected column format
        expected_columns = [
            "transit_epoch_bjd",
            "transit_depth_ppm", 
            "equilibrium_temp_k",
            "impact_parameter",
            "stellar_teff_k",
            "stellar_mass_msun",
            "stellar_logg",
            "radius_ratio_est"
        ]
        
        st.code(", ".join(expected_columns))
        st.warning("⚠️ Column names must match exactly (case-sensitive) and be in the same order as shown above.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with 8 numeric columns with exact column names as specified above"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded CSV
                df = pd.read_csv(uploaded_file)
                
                st.subheader("📋 Uploaded Data Preview")
                st.write(f"Shape: {df.shape}")
                st.dataframe(df.head())
                
                # Validate the data
                is_valid, validation_message = validate_csv_data(df)
                
                if is_valid:
                    st.success(validation_message)
                    
                    # Make batch predictions
                    if st.button("🚀 Make Batch Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            results_df = make_batch_predictions(pipeline, df)
                        
                        if results_df is not None:
                            st.subheader("🎯 Prediction Results")
                            
                            # Display summary statistics
                            col1, col2, col3 = st.columns(3)
                            
                            total_samples = len(results_df)
                            confirmed_count = (results_df['prediction'] == 1).sum()
                            false_positive_count = total_samples - confirmed_count
                            avg_confidence = results_df['confidence_score'].mean()
                            
                            with col1:
                                st.metric("Total Samples", total_samples)
                            with col2:
                                st.metric("Confirmed Exoplanets", confirmed_count)
                            with col3:
                                st.metric("False Positives", false_positive_count)
                            
                            st.metric("Average Confidence", f"{avg_confidence:.4f}")
                            
                            # Display results table
                            st.subheader("📊 Detailed Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download button for results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="exoplanet_predictions.csv",
                                mime="text/csv",
                                type="primary"
                            )
                            
                            # Additional insights
                            st.subheader("📈 Prediction Insights")
                            
                            if confirmed_count > 0:
                                st.write(f"**Confirmed Exoplanets ({confirmed_count}):**")
                                confirmed_df = results_df[results_df['prediction'] == 1]
                                st.write(f"- Average confidence: {confirmed_df['confidence_score'].mean():.4f}")
                                st.write(f"- Highest confidence: {confirmed_df['confidence_score'].max():.4f}")
                                st.write(f"- Lowest confidence: {confirmed_df['confidence_score'].min():.4f}")
                            
                            if false_positive_count > 0:
                                st.write(f"**False Positives ({false_positive_count}):**")
                                false_positive_df = results_df[results_df['prediction'] == 0]
                                st.write(f"- Average confidence: {(1 - false_positive_df['confidence_score']).mean():.4f}")
                                st.write(f"- Highest confidence: {(1 - false_positive_df['confidence_score']).max():.4f}")
                                st.write(f"- Lowest confidence: {(1 - false_positive_df['confidence_score']).min():.4f}")
                            
                else:
                    st.error(validation_message)
                    st.info("Please ensure your CSV has exactly 8 numeric columns in the specified order.")
                    
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please make sure your file is a valid CSV format.")
    
