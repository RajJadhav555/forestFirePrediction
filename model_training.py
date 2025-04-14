import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def load_and_process_data(file_path="forestfires.csv"):
    """Load and process the forest fires dataset"""
    # Load Dataset
    df = pd.read_csv(file_path)
    
    # Feature Engineering: Creating new features
    df['fire_risk'] = (df['FFMC'] * df['temp']) / (df['RH'] + 1)
    df['temp_wind'] = df['temp'] * df['wind']
    df['humidity_rain'] = df['RH'] * df['rain']
    
    return df

def train_models(df):
    """Train multiple models and return the best one"""
    # Define features
    categorical_features = ['month', 'day']
    numeric_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 
                        'wind', 'rain', 'fire_risk', 'temp_wind', 'humidity_rain']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    
    # Splitting Data
    y = df['area']  # Target Variable
    X = df.drop(['area'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train & Evaluate Models
    results = {}
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': pipeline,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"{name}: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = name
    
    # Save Best Model
    joblib.dump(results[best_model]['model'], "fire_model.pkl")
    
    return results, best_model, X_test

def get_feature_importance(results, best_model, X_test):
    """Extract feature importance from the best model if possible"""
    if best_model == 'RandomForest' or best_model == 'XGBoost':
        # Get the actual model from the pipeline
        pipeline = results[best_model]['model']
        model = pipeline.named_steps['model']
        
        # Get preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Transform the test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        numeric_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 
                           'wind', 'rain', 'fire_risk', 'temp_wind', 'humidity_rain']
        categorical_features = []
        
        # One-hot encoding expands categorical features
        ohe = preprocessor.named_transformers_['cat']
        if hasattr(ohe, 'get_feature_names_out'):
            cat_features = ohe.get_feature_names_out(['month', 'day'])
            categorical_features = list(cat_features)
        
        all_features = numeric_features + categorical_features
        
        # Get feature importance
        if best_model == 'RandomForest':
            importance = model.feature_importances_
            # Only take the first len(all_features) elements if there's a mismatch
            importance = importance[:len(all_features)] if len(importance) > len(all_features) else importance
            
            # Create a DataFrame for feature importance
            feature_importance = pd.DataFrame({
                'Feature': all_features[:len(importance)],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return feature_importance
        
        elif best_model == 'XGBoost':
            importance = model.feature_importances_
            # Only take the first len(all_features) elements if there's a mismatch
            importance = importance[:len(all_features)] if len(importance) > len(all_features) else importance
            
            # Create a DataFrame for feature importance
            feature_importance = pd.DataFrame({
                'Feature': all_features[:len(importance)],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return feature_importance
    
    return None

if __name__ == "__main__":
    # If model file doesn't exist, train and save it
    if not os.path.exists("fire_model.pkl"):
        print("Training new model...")
        df = load_and_process_data()
        results, best_model, X_test = train_models(df)
        print(f"Best model: {best_model}")
        
        # Get feature importance if available
        importance = get_feature_importance(results, best_model, X_test)
        if importance is not None:
            print("\nFeature Importance:")
            print(importance.head(10))
        
        print("\nModel saved as fire_model.pkl")
    else:
        print("Model already exists. Skipping training.")
