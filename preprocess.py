import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

def train_linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def process_and_train_all_cities(data_directory, model_directory):
    for city_file in os.listdir(data_directory):
        if city_file.endswith('.csv'):
            city_name = city_file.replace('.csv', '')
            print(f"Processing {city_name}...")

            df = pd.read_csv(os.path.join(data_directory, city_file))

            # Drop rows with NaN values
            df = df.dropna()

            # Check if the DataFrame is empty after dropping NaNs
            if df.empty:
                print(f"No data available for {city_name} after dropping NaNs. Skipping...")
                continue

            X = df[['Year', 'Month', 'Day']]
            y = df['HeatIndex']

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            scaled_X = scaler_X.fit_transform(X)
            scaled_y = scaler_y.fit_transform(y.values.reshape(-1, 1))

            model = train_linear_regression_model(scaled_X, scaled_y)

            # Save the model and scaler
            model_path = os.path.join(model_directory, f'{city_name}_heat_index_model.pkl')
            scaler_X_path = os.path.join(model_directory, f'{city_name}_scaler_X.pkl')
            scaler_y_path = os.path.join(model_directory, f'{city_name}_scaler_y.pkl')

            joblib.dump(model, model_path)
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)

if __name__ == "__main__":
    data_directory = '/Users/carol/Documents/School/3rd Year/2nd Sem/Forecast/Heat Index Forecasting App/backend/Data'
    model_directory = '/Users/carol/Documents/School/3rd Year/2nd Sem/Forecast/Heat Index Forecasting App/backend/models'
    
    process_and_train_all_cities(data_directory, model_directory)
