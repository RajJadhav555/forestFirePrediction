import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os
from utils import (
    load_data, load_model, create_input_features, 
    prepare_prediction_data, make_prediction,
    display_prediction_result, plot_correlation_heatmap,
    plot_monthly_distribution, plot_feature_importance,
    plot_fire_risk_by_month_temp
)
from model_training import load_and_process_data, train_models, get_feature_importance

st.set_page_config(
    page_title="Forest Fire Prediction App",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for feature importance
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Application title and description
st.title("🔥 Forest Fire Prediction Tool")
st.markdown("""
This application predicts the burned area of forest fires based on various meteorological and environmental factors.
""")

# Check if model exists, if not train it
if not os.path.exists("fire_model.pkl"):
    with st.spinner("Training the model for the first time. This may take a moment..."):
        df = load_and_process_data()
        results, best_model, X_test = train_models(df)
        importance = get_feature_importance(results, best_model, X_test)
        if importance is not None:
            st.session_state.feature_importance = importance

# Load the model
model = load_model()

# Load the dataset
df = load_data()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Exploration", "Model Information"])

# Tab 1: Prediction
with tab1:
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Predict Forest Fire Burned Area")
        
        # Input form
        with st.form(key="prediction_form"):
            input_data = create_input_features()
            submit_button = st.form_submit_button(label="Predict Fire")
        
        # Make prediction when form is submitted
        if submit_button:
            if model is not None:
                with st.spinner("Making prediction..."):
                    # Prepare input data for prediction
                    input_df = prepare_prediction_data(input_data)
                    
                    # Make prediction
                    prediction = make_prediction(model, input_df)
                    
                    if prediction is not None:
                        # Display prediction result
                        display_prediction_result(prediction)
            else:
                st.error("Model could not be loaded. Please check if the model file exists.")
    
    with col2:
        st.header("Fire Risk Factors")
        st.markdown("""
        ### Key Parameters:
        
        - **FFMC**: Fine Fuel Moisture Code - Indicator of ease of ignition
        - **DMC**: Duff Moisture Code - Represents fuel consumption in moderate duff layers
        - **DC**: Drought Code - Represents deep fuel consumption and seasonal drought effects
        - **ISI**: Initial Spread Index - Represents rate of fire spread
        
        ### Environmental Factors:
        
        - **Temperature**: Higher temperatures increase fire risk
        - **Relative Humidity**: Lower humidity increases fire risk
        - **Wind**: Higher wind speeds accelerate fire spread
        - **Rain**: Recent rainfall reduces fire risk
        
        ### Fire Risk Categories:
        
        - **Low Risk** (< 1 ha): Minimal fire spread
        - **Moderate Risk** (1-10 ha): Limited spread
        - **High Risk** (10-50 ha): Significant spread potential
        - **Extreme Risk** (> 50 ha): Critical fire conditions
        """)

# Tab 2: Data Exploration
with tab2:
    st.header("Forest Fire Data Exploration")
    
    if df is not None:
        # Display basic statistics
        st.subheader("Dataset Overview")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**Number of records:** {df.shape[0]}")
            st.write(f"**Features:** {df.shape[1]}")
            
            # Display summary of burned area
            st.subheader("Burned Area Statistics")
            area_stats = df['area'].describe()
            st.write(f"**Mean:** {area_stats['mean']:.2f} hectares")
            st.write(f"**Median:** {area_stats['50%']:.2f} hectares")
            st.write(f"**Max:** {area_stats['max']:.2f} hectares")
            st.write(f"**Min:** {area_stats['min']:.2f} hectares")
            
            # Count fires by severity
            low_risk = len(df[df['area'] < 1])
            moderate_risk = len(df[(df['area'] >= 1) & (df['area'] < 10)])
            high_risk = len(df[(df['area'] >= 10) & (df['area'] < 50)])
            extreme_risk = len(df[df['area'] >= 50])
            
            st.write(f"**Low Risk Fires:** {low_risk} ({low_risk/len(df)*100:.1f}%)")
            st.write(f"**Moderate Risk Fires:** {moderate_risk} ({moderate_risk/len(df)*100:.1f}%)")
            st.write(f"**High Risk Fires:** {high_risk} ({high_risk/len(df)*100:.1f}%)")
            st.write(f"**Extreme Risk Fires:** {extreme_risk} ({extreme_risk/len(df)*100:.1f}%)")
            
        with col2:
            # Sample data
            st.write("Sample Data:")
            st.dataframe(df.head(5))
            
            # Distribution of burned area
            fig = px.histogram(
                df, 
                x='area', 
                nbins=50,
                title='Distribution of Burned Area',
                labels={'area': 'Burned Area (hectares)'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Correlation Matrix", "Monthly Analysis", "Feature Importance", "Fire Risk by Conditions"
        ])
        
        with viz_tab1:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
            
            st.markdown("""
            ### Observations from Correlation Matrix:
            
            - **FFMC, DMC, and DC** show positive correlation with fire area
            - **RH (Relative Humidity)** shows negative correlation with fire area
            - **Temperature** is positively correlated with fire risk
            - **Rain** is negatively correlated with most fire indicators
            """)
        
        with viz_tab2:
            st.plotly_chart(plot_monthly_distribution(df), use_container_width=True)
            
            st.markdown("""
            ### Monthly Fire Patterns:
            
            - **August and September** typically have the highest fire occurrence
            - **Winter months** show significantly fewer fires
            - Fire severity tends to be highest during late summer and early fall
            """)
        
        with viz_tab3:
            # Use stored feature importance if available
            if st.session_state.feature_importance is not None:
                st.plotly_chart(plot_feature_importance(st.session_state.feature_importance), use_container_width=True)
            else:
                st.plotly_chart(plot_feature_importance(), use_container_width=True)
                st.info("Note: This is a representation of typical feature importance. Retrain the model to see actual values.")
            
            st.markdown("""
            ### Key Influencing Factors:
            
            - **FFMC (Fine Fuel Moisture Code)** is typically the most important predictor
            - **Weather conditions** (temperature, humidity) significantly influence fire behavior
            - **Seasonal factors** (month, drought conditions) affect fire risk
            """)
        
        with viz_tab4:
            st.plotly_chart(plot_fire_risk_by_month_temp(df), use_container_width=True)
            
            st.markdown("""
            ### Fire Risk by Environmental Conditions:
            
            - Higher temperatures generally correlate with larger fires
            - Seasonal variations show distinct patterns of fire risk
            - Combined effects of multiple factors create highest risk scenarios
            """)
    
    else:
        st.error("Could not load the dataset. Please check if the file exists.")

# Tab 3: Model Information
with tab3:
    st.header("Fire Prediction Model Information")
    
    # Model Information
    st.subheader("Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Models Evaluated:
        
        1. **Random Forest Regressor**
           - Ensemble learning method
           - Robust to outliers
           - Handles non-linear relationships well
        
        2. **Support Vector Regressor (SVR)**
           - Effective in high-dimensional spaces
           - Versatile kernel functions
           - Good for complex relationships
        
        3. **XGBoost Regressor**
           - Gradient boosting framework
           - High performance and efficiency
           - Handles missing data well
        """)
    
    with col2:
        st.markdown("""
        ### Model Features:
        
        #### Geographic:
        - X, Y: spatial coordinates within the Montesinho park map
        
        #### Temporal:
        - month: month of the year
        - day: day of the week
        
        #### FWI Components:
        - FFMC: Fine Fuel Moisture Code
        - DMC: Duff Moisture Code
        - DC: Drought Code
        - ISI: Initial Spread Index
        
        #### Weather:
        - temp: temperature (°C)
        - RH: relative humidity (%)
        - wind: wind speed (km/h)
        - rain: outside rain (mm/m²)
        
        #### Engineered Features:
        - fire_risk: (FFMC * temp) / (RH + 1)
        - temp_wind: temp * wind
        - humidity_rain: RH * rain
        """)
    
    # How the prediction works
    st.subheader("How Predictions Work")
    st.markdown("""
    The fire prediction model works as follows:
    
    1. **Data Preprocessing**:
       - Numerical features are standardized (zero mean, unit variance)
       - Categorical features (month, day) are one-hot encoded
    
    2. **Prediction Pipeline**:
       - User inputs are processed through the same preprocessing steps as training data
       - The model predicts the burned area in hectares
    
    3. **Risk Categorization**:
       - Predictions are categorized into risk levels based on predicted area
       - Visual indicators show severity levels
    
    4. **Model Selection**:
       - The best performing model among RF, SVR, and XGBoost is selected based on R² score
       - Model is periodically retrained with new data to maintain accuracy
    """)
    
    # Usage tips
    st.subheader("Tips for Using the Prediction Tool")
    st.markdown("""
    - **Seasonal Awareness**: Pay attention to month selection as fire risk varies seasonally
    - **Weather Combinations**: The combination of high temperature, low humidity, and wind creates highest risk
    - **FWI Understanding**: Higher FFMC, DMC, DC, and ISI values generally indicate higher fire risk
    - **Interpretation**: The prediction represents the potential burned area if a fire occurs, not the probability of a fire starting
    """)

# Footer
st.markdown("---")
st.markdown("Forest Fire Prediction Tool | Created with Streamlit")

# Run the app
if __name__ == "__main__":
    pass
