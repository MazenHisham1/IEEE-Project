import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for forecasting"""
    df = pd.read_csv('covid.csv')
    df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
    
    # Aggregate global data by date
    global_data = df.groupby('ObservationDate').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum'
    }).reset_index()
    
    # Calculate daily new cases
    global_data['New_Cases'] = global_data['Confirmed'].diff().fillna(0)
    global_data['New_Deaths'] = global_data['Deaths'].diff().fillna(0)
    
    # Remove negative values (data corrections)
    global_data['New_Cases'] = global_data['New_Cases'].clip(lower=0)
    global_data['New_Deaths'] = global_data['New_Deaths'].clip(lower=0)
    
    return global_data

def arima_forecast(data, column, days_ahead=7):
    """Perform ARIMA forecasting"""
    # Prepare data
    ts_data = data[column].values
    
    # Find best ARIMA parameters (simplified approach)
    best_aic = float('inf')
    best_order = None
    
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(ts_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    # Fit best model and forecast
    if best_order:
        model = ARIMA(ts_data, order=best_order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=days_ahead)
        conf_int = fitted_model.get_forecast(steps=days_ahead).conf_int()
        
        return forecast, conf_int, best_order
    else:
        return None, None, None

def prophet_forecast(data, column, days_ahead=7):
    """Perform Prophet forecasting"""
    # Prepare data for Prophet
    prophet_data = data[['ObservationDate', column]].copy()
    prophet_data.columns = ['ds', 'y']
    
    # Initialize and fit Prophet model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    return model, forecast

def detect_anomalies(data, column, window=7, threshold=2):
    """Detect anomalies using rolling statistics"""
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    
    # Calculate z-scores
    z_scores = (data[column] - rolling_mean) / rolling_std
    anomalies = np.abs(z_scores) > threshold
    
    return anomalies, z_scores

def calculate_moving_averages(data, columns, windows=[7, 14, 30]):
    """Calculate moving averages for trend analysis"""
    result = data.copy()
    
    for column in columns:
        for window in windows:
            result[f'{column}_MA_{window}'] = data[column].rolling(window=window).mean()
    
    return result

def main():
    # Load data
    print("Loading and preparing data...")
    global_data = load_and_prepare_data()
    
    # Calculate moving averages
    print("Calculating moving averages...")
    global_data = calculate_moving_averages(global_data, ['New_Cases', 'New_Deaths'])
    
    # Detect anomalies
    print("Detecting anomalies...")
    anomalies_cases, z_scores_cases = detect_anomalies(global_data, 'New_Cases')
    anomalies_deaths, z_scores_deaths = detect_anomalies(global_data, 'New_Deaths')
    
    global_data['Anomaly_Cases'] = anomalies_cases
    global_data['Anomaly_Deaths'] = anomalies_deaths
    global_data['Z_Score_Cases'] = z_scores_cases
    global_data['Z_Score_Deaths'] = z_scores_deaths
    
    # ARIMA forecasting
    print("Performing ARIMA forecasting...")
    arima_forecast_cases, arima_conf_cases, arima_order_cases = arima_forecast(global_data, 'New_Cases')
    arima_forecast_deaths, arima_conf_deaths, arima_order_deaths = arima_forecast(global_data, 'New_Deaths')
    
    # Prophet forecasting
    print("Performing Prophet forecasting...")
    prophet_model_cases, prophet_forecast_cases = prophet_forecast(global_data, 'New_Cases')
    prophet_model_deaths, prophet_forecast_deaths = prophet_forecast(global_data, 'New_Deaths')
    
    # Save results
    print("Saving results...")
    global_data.to_csv('processed_data_with_forecasts.csv', index=False)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Data points: {len(global_data)}")
    print(f"Date range: {global_data['ObservationDate'].min()} to {global_data['ObservationDate'].max()}")
    print(f"Anomalies detected in cases: {anomalies_cases.sum()}")
    print(f"Anomalies detected in deaths: {anomalies_deaths.sum()}")
    
    if arima_order_cases:
        print(f"Best ARIMA order for cases: {arima_order_cases}")
    if arima_order_deaths:
        print(f"Best ARIMA order for deaths: {arima_order_deaths}")
    
    return global_data, {
        'arima_cases': (arima_forecast_cases, arima_conf_cases),
        'arima_deaths': (arima_forecast_deaths, arima_conf_deaths),
        'prophet_cases': prophet_forecast_cases,
        'prophet_deaths': prophet_forecast_deaths
    }

if __name__ == "__main__":
    data, forecasts = main()

