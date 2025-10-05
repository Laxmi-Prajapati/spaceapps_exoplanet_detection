import streamlit as st
# Import page modules
from pages.about import show_about_page
from pages.predict import show_predict_page
from pages.retrain import show_retrain_page

# Configure the page
st.set_page_config(
    page_title="Exoplanet Detection System",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    /* Hide the file browser/hamburger menu */
    .stApp > header {
        display: none;
    }
    /* Hide the main menu button */
    #MainMenu {
        display: none;
    }
    /* Hide the footer */
    footer {
        display: none;
    }
    /* Hide the "Made with Streamlit" footer */
    .viewerBadge_container__1QSob {
        display: none;
    }
    /* Hide the top toolbar */
    .stAppToolbar {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌟 Exoplanet Detection System</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚀 Navigation")
st.sidebar.markdown("---")

# Page selection
page = st.sidebar.selectbox(
    "Choose a page:",
    ["About Us", "Predict", "Retrain"],
    index=0,
    help="Select the page you want to view"
)

# Additional sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
**Model Performance:**
- ROC AUC: 96.64%
- F1-Score: 84.45%
- Accuracy: 89.90%
""")

st.sidebar.markdown("### 🎯 Features")
st.sidebar.markdown("""
- **Manual Prediction**: Use number inputs for prediction
- **Batch Prediction**: Upload CSV files
- **Model Retraining**: Upload data to retrain model
- **Downloadable Results**: Get predictions and models as files
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*NASA SpaceApps Challenge 2025*")

# Page routing
if page == "About Us":
    show_about_page()
elif page == "Predict":
    show_predict_page()
elif page == "Retrain":
    show_retrain_page()