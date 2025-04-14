import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

def load_data(file_path="forestfires.csv"):
    """Load the forest fires dataset"""
    try:
        df = pd.read_csv(file_path)
        # Feature Engineering: Creating new features
        df['fire_risk'] = (df['FFMC'] * df['temp']) / (df['RH'] + 1)
        df['temp_wind'] = df['temp'] * df['wind']
        df['humidity_rain'] = df['RH'] * df['rain']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_model(model_path="fire_model.pkl"):
    """Load the trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_input_features():
    """Create input widgets for all features"""
    st.subheader("Enter Forest Fire Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        X = st.number_input("X-axis spatial coordinate", min_value=0, max_value=10, value=5)
        Y = st.number_input("Y-axis spatial coordinate", min_value=0, max_value=10, value=5)
        month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month = st.selectbox("Month", options=month_options, index=month_options.index('aug'))
        day_options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        day = st.selectbox("Day of week", options=day_options, index=day_options.index('fri'))
        FFMC = st.slider("FFMC (Fine Fuel Moisture Code)", min_value=0.0,max_value=100.0, value=90.0, step=0.1,
                         help="Numeric rating of the moisture content of litter and other cured fine fuels")
        DMC = st.slider("DMC (Duff Moisture Code)", min_value=0.0, max_value=160.0, value=85.0, step=0.1,
                        help="Numeric rating of the average moisture content of loosely compacted organic layers of moderate depth")
        DC = st.slider("DC (Drought Code)", min_value=0.0, max_value=800.0, value=500.0, step=0.1,
                       help="Numeric rating of the average moisture content of deep, compact organic layers")
    
    with col2:
        ISI = st.slider("ISI (Initial Spread Index)", min_value=0.0, max_value=60.0, value=10.0, step=0.1,
                         help="Numeric rating of the expected rate of fire spread")
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        RH = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        rain = st.slider("Outside Rain (mm/m²)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
        # Calculate derived features
        fire_risk = (FFMC * temp) / (RH + 1)
        temp_wind = temp * wind
        humidity_rain = RH * rain
    
    # Return all input features as a dictionary
    input_data = {
        'X': X, 'Y': Y, 'month': month, 'day': day,
        'FFMC': FFMC, 'DMC': DMC, 'DC': DC, 'ISI': ISI,
        'temp': temp, 'RH': RH, 'wind': wind, 'rain': rain,
        'fire_risk': fire_risk, 'temp_wind': temp_wind, 'humidity_rain': humidity_rain
    }
    
    return input_data

def prepare_prediction_data(input_data):
    """Convert input data to DataFrame for prediction"""
    return pd.DataFrame([input_data])

def make_prediction(model, input_df):
    """Make prediction using the loaded model"""
    try:
        prediction = model.predict(input_df)[0]
        return max(0, prediction)  # Ensure non-negative area
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def display_prediction_result(prediction):
    """Display the prediction result with visualization"""
    st.subheader("Fire Prediction Result")
    
    # Display numerical result
    st.metric("Predicted Burned Area", f"{prediction:.2f} hectares")
    
    # Categorize the prediction
    if prediction < 1:
        risk_category = "Low Risk"
        color = "green"
    elif prediction < 10:
        risk_category = "Moderate Risk"
        color = "yellow"
    elif prediction < 50:
        risk_category = "High Risk"
        color = "orange"
    else:
        risk_category = "Extreme Risk"
        color = "red"
    
    # Create a gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fire Risk: {risk_category}"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(0, 250, 0, .25)'},
                {'range': [1, 10], 'color': 'rgba(255, 255, 0, .25)'},
                {'range': [10, 50], 'color': 'rgba(255, 150, 0, .25)'},
                {'range': [50, 100], 'color': 'rgba(255, 0, 0, .25)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display interpretation
    st.markdown(f"### Interpretation")
    if prediction < 1:
        st.write("🟢 **Low Risk**: Minimal likelihood of significant fire spread.")
    elif prediction < 10:
        st.write("🟡 **Moderate Risk**: Fire may spread, but likely controllable with standard resources.")
    elif prediction < 50:
        st.write("🟠 **High Risk**: Significant fire potential with rapid spread possible. Additional resources may be needed.")
    else:
        st.write("🔴 **Extreme Risk**: Critical fire conditions. Extensive, difficult-to-control fire likely if ignition occurs.")

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features"""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr, 
        text_auto=True, 
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(height=700)
    
    return fig

def plot_monthly_distribution(df):
    """Plot distribution of fires by month"""
    monthly_counts = df.groupby('month')['area'].agg(['count', 'mean', 'sum']).reset_index()
    
    # Sort by actual month order
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    monthly_counts['month_num'] = monthly_counts['month'].map(month_order)
    monthly_counts = monthly_counts.sort_values('month_num')
    
    # Create bar chart
    fig = px.bar(
        monthly_counts, 
        x='month', 
        y='count',
        color='sum',
        labels={'count': 'Number of Fires', 'sum': 'Total Area Burned (ha)'},
        title='Fire Occurrence by Month',
        hover_data=['mean']
    )
    
    return fig

def plot_feature_importance(importance_df=None):
    """Plot feature importance if available"""
    if importance_df is None:
        # Create sample importance data if actual data not available
        importance_df = pd.DataFrame({
            'Feature': ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day'],
            'Importance': [0.25, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02]
        })
    
    # Create bar chart
    fig = px.bar(
        importance_df.head(10), 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importance',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance'
    )
    
    return fig

def plot_fire_risk_by_month_temp(df):
    """Plot fire risk by month and temperature"""
    # Create pivot table
    pivot_data = df.pivot_table(
        index='month', 
        columns=pd.cut(df['temp'], bins=[0, 10, 20, 30, 40], labels=['0-10°C', '10-20°C', '20-30°C', '30-40°C']),
        values='area',
        aggfunc='mean'
    ).fillna(0)
    
    # Sort by month
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    pivot_data = pivot_data.reset_index()
    pivot_data['month_num'] = pivot_data['month'].map(month_order)
    pivot_data = pivot_data.sort_values('month_num').drop('month_num', axis=1).set_index('month')
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Temperature Range", y="Month", color="Avg. Burned Area (ha)"),
        title="Average Burned Area by Month and Temperature",
        color_continuous_scale='Viridis'
    )
    
    return fig
