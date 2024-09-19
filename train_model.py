import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Function to train models
def train_models(X, y):
    models = {
        'linear_regression': LinearRegression(),
        'knn': KNeighborsRegressor(n_neighbors=5),
        'random_forest': RandomForestRegressor(n_estimators=100),
        'decision_tree': DecisionTreeRegressor()
    }
    
    trained_models = {}
    
    for model_name, model in models.items():
        model.fit(X, y)
        trained_models[model_name] = model
    
    return trained_models

# Process and train models for all cities
def process_and_train_all_cities(data_directory, model_directory):
    for city_file in os.listdir(data_directory):
        if city_file.endswith('.csv'):
            city_name = city_file.replace('.csv', '')
            print(f"Training models for {city_name}...")

            df = pd.read_csv(os.path.join(data_directory, city_file))

            # Fill NaN values with the mean of the column
            df = df.fillna(df.mean())

            # If the dataset is still empty after filling NaNs, skip it
            if df.empty:
                print(f"No data available for {city_name} after handling NaNs. Skipping...")
                continue

            X = df[['Year', 'Month', 'Day']]
            y = df['HeatIndex']

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            scaled_X = scaler_X.fit_transform(X)
            scaled_y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

            # Train models
            trained_models = train_models(scaled_X, scaled_y)

            # Save models and scalers
            for model_name, model in trained_models.items():
                model_path = os.path.join(model_directory, f'{city_name}_{model_name}_model.pkl')
                joblib.dump(model, model_path)

            scaler_X_path = os.path.join(model_directory, f'{city_name}_scaler_X.pkl')
            scaler_y_path = os.path.join(model_directory, f'{city_name}_scaler_y.pkl')

            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)

            print(f"Models and scalers saved for {city_name}")

if __name__ == "__main__":
    data_directory = '/Users/carol/Documents/School/3rd Year/2nd Sem/Forecast/Heat Index Forecasting App/backend/Data'
    model_directory = '/Users/carol/Documents/School/3rd Year/2nd Sem/Forecast/Heat Index Forecasting App/backend/models'

    process_and_train_all_cities(data_directory, model_directory)
