import streamlit as st

def show_about_page():
    """Display the About Us page content"""
    
    st.header("🎯 Project Overview")
    
    st.markdown("""
    A comprehensive machine learning pipeline for exoplanet confirmation using physics-based feature engineering and advanced ensemble methods.
    
    This project implements a state-of-the-art binary classification system to distinguish between confirmed exoplanets and false positives using data from NASA's exoplanet archive. The pipeline features sophisticated data preprocessing, physics-first imputation strategies, and optimized ensemble learning.
    """)
    
    # Application Pages
    st.header("📱 Application Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏠 About Us")
        st.markdown("""
        **Current Page**
        - Project overview and objectives
        - Technical achievements and metrics
        - Model performance details
        - Feature importance analysis
        - Development methodology
        """)
    
    with col2:
        st.subheader("🔮 Predict")
        st.markdown("""
        **Exoplanet Classification**
        - Manual feature input for single predictions
        - CSV batch processing for multiple candidates
        - Real-time prediction with confidence scores
        - Input validation and error handling
        - Download results functionality
        """)
    
    with col3:
        st.subheader("🔄 Retrain")
        st.markdown("""
        **Model Retraining**
        - Upload custom training datasets
        - Automated data validation
        - Multi-algorithm ensemble training
        - Performance metrics evaluation
        - Model comparison and selection
        """)
    
    st.info("💡 **Navigation Tip**: Use the sidebar on the left to switch between different pages and explore all the features of our exoplanet detection system!")
    
    # Key Achievements
    st.header("🏆 Key Achievements")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC AUC", "96.64%", "Outstanding")
    with col2:
        st.metric("F1-Score", "84.45%", "Excellent")
    with col3:
        st.metric("Accuracy", "89.90%", "High")
    with col4:
        st.metric("Precision", "85.32%", "Strong")
    
    # Technical Features
    st.header("🔬 Technical Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Processing Pipeline")
        st.markdown("""
        - **Enhanced KNN Imputation**: 9 physics-motivated features with noise injection
        - **Physics-first Calculations**: Stellar mass derived from fundamental physics
        - **Comprehensive Error Detection**: Automatic correction of impossible values
        - **Stratified Validation**: Robust imputation quality assessment
        """)
    
    with col2:
        st.subheader("Machine Learning Pipeline")
        st.markdown("""
        - **F-Score Feature Selection**: Top 12 features from 18 candidates
        - **Stacking Ensemble**: Random Forest + Gradient Boosting + SVM + Logistic Regression
        - **Grid Search Optimization**: Hyperparameter tuning with 3-fold cross-validation
        - **Class Imbalance Handling**: Random Over-Sampling (ROS)
        """)
    
    # Key Features by Importance
    st.header("📊 Key Features by Importance")
    
    features_importance = [
        ("equilibrium_temp_k", 18.1, "Planet equilibrium temperature"),
        ("impact_parameter", 13.4, "Orbital geometry parameter"),
        ("transit_depth_ppm", 11.9, "Signal strength measure"),
        ("radius_ratio_est", 10.7, "Planet-to-star radius ratio"),
        ("transit_epoch_bjd", 10.5, "Transit timing")
    ]
    
    for feature, importance, description in features_importance:
        st.markdown(f"**{feature}** ({importance}%) - {description}")
    
    # Model Performance Details
    st.header("📈 Detailed Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Performance")
        metrics_data = {
            "Metric": ["ROC AUC", "F1-Score", "Accuracy", "Precision", "Recall"],
            "Value": ["0.9664", "0.8445", "0.8990", "0.8532", "0.8361"]
        }
        st.table(metrics_data)
    
    with col2:
        st.subheader("Cross-Validation Results")
        cv_data = {
            "Model": ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "Stacking Ensemble"],
            "AUC": ["98.72%", "99.12%", "96.54%", "89.66%", "96.64%"]
        }
        st.table(cv_data)
    
    # Links
    st.header("🔗 Links")
    st.markdown("""
    - [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
    - [SpaceApps Challenge](https://www.spaceappschallenge.org/)
    """)
    
    st.markdown("---")
    st.markdown("*Built with ❤️ for NASA SpaceApps Challenge 2025*")