import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_processed_data():
    """Load the processed data with forecasts"""
    return pd.read_csv('processed_data_with_forecasts.csv')

def identify_surge_periods(data, column, threshold_percentile=90):
    """Identify surge periods based on percentile threshold"""
    threshold = np.percentile(data[column], threshold_percentile)
    surges = data[column] > threshold
    
    # Find consecutive surge periods
    surge_periods = []
    in_surge = False
    start_date = None
    
    for idx, is_surge in enumerate(surges):
        if is_surge and not in_surge:
            in_surge = True
            start_date = data.iloc[idx]['ObservationDate']
        elif not is_surge and in_surge:
            in_surge = False
            end_date = data.iloc[idx-1]['ObservationDate']
            surge_periods.append((start_date, end_date))
    
    # Handle case where surge continues to end of data
    if in_surge:
        end_date = data.iloc[-1]['ObservationDate']
        surge_periods.append((start_date, end_date))
    
    return surge_periods, threshold

def calculate_trend_indicators(data, column):
    """Calculate various trend indicators"""
    # Linear trend
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data[column])
    
    # Rate of change
    rate_of_change = data[column].pct_change()
    
    # Acceleration (second derivative)
    acceleration = rate_of_change.diff()
    
    # Momentum (rate of change of moving average)
    ma_7 = data[column].rolling(window=7).mean()
    momentum = ma_7.pct_change()
    
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'rate_of_change': rate_of_change,
        'acceleration': acceleration,
        'momentum': momentum
    }

def detect_changepoints(data, column, min_size=5):
    """Detect significant changepoints in the time series"""
    from scipy.signal import find_peaks
    
    # Calculate first and second derivatives
    first_diff = np.diff(data[column])
    second_diff = np.diff(first_diff)
    
    # Find peaks in absolute second derivative (changepoints)
    peaks, _ = find_peaks(np.abs(second_diff), height=np.std(second_diff))
    
    changepoints = []
    for peak in peaks:
        if peak >= min_size and peak < len(data) - min_size:
            changepoints.append(data.iloc[peak]['ObservationDate'])
    
    return changepoints

def create_trend_analysis_plots(data):
    """Create comprehensive trend analysis plots"""
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'New Cases with Moving Averages',
            'New Deaths with Moving Averages',
            'Anomaly Detection - Cases',
            'Anomaly Detection - Deaths',
            'Rate of Change - Cases',
            'Rate of Change - Deaths',
            'Cumulative Cases vs Deaths',
            'Weekly Growth Rate'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: New Cases with Moving Averages
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['New_Cases'],
                  mode='lines', name='New Cases', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['New_Cases_MA_7'],
                  mode='lines', name='7-day MA', line=dict(color='red', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['New_Cases_MA_14'],
                  mode='lines', name='14-day MA', line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Plot 2: New Deaths with Moving Averages
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['New_Deaths'],
                  mode='lines', name='New Deaths', line=dict(color='orange', width=1)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['New_Deaths_MA_7'],
                  mode='lines', name='7-day MA', line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Plot 3: Anomaly Detection - Cases
    normal_cases = data[~data['Anomaly_Cases']]
    anomaly_cases = data[data['Anomaly_Cases']]
    
    fig.add_trace(
        go.Scatter(x=normal_cases['ObservationDate'], y=normal_cases['New_Cases'],
                  mode='markers', name='Normal', marker=dict(color='blue', size=4)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=anomaly_cases['ObservationDate'], y=anomaly_cases['New_Cases'],
                  mode='markers', name='Anomaly', marker=dict(color='red', size=8, symbol='x')),
        row=2, col=1
    )
    
    # Plot 4: Anomaly Detection - Deaths
    normal_deaths = data[~data['Anomaly_Deaths']]
    anomaly_deaths = data[data['Anomaly_Deaths']]
    
    fig.add_trace(
        go.Scatter(x=normal_deaths['ObservationDate'], y=normal_deaths['New_Deaths'],
                  mode='markers', name='Normal', marker=dict(color='orange', size=4)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=anomaly_deaths['ObservationDate'], y=anomaly_deaths['New_Deaths'],
                  mode='markers', name='Anomaly', marker=dict(color='red', size=8, symbol='x')),
        row=2, col=2
    )
    
    # Calculate rate of change
    data['Cases_ROC'] = data['New_Cases'].pct_change() * 100
    data['Deaths_ROC'] = data['New_Deaths'].pct_change() * 100
    
    # Plot 5: Rate of Change - Cases
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['Cases_ROC'],
                  mode='lines', name='Cases ROC (%)', line=dict(color='blue')),
        row=3, col=1
    )
    
    # Plot 6: Rate of Change - Deaths
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=data['Deaths_ROC'],
                  mode='lines', name='Deaths ROC (%)', line=dict(color='orange')),
        row=3, col=2
    )
    
    # Plot 7: Cumulative Cases vs Deaths
    fig.add_trace(
        go.Scatter(x=data['Confirmed'], y=data['Deaths'],
                  mode='markers+lines', name='Cases vs Deaths',
                  marker=dict(color=data.index, colorscale='Viridis', size=6)),
        row=4, col=1
    )
    
    # Plot 8: Weekly Growth Rate
    weekly_growth = data['New_Cases_MA_7'].pct_change(periods=7) * 100
    fig.add_trace(
        go.Scatter(x=data['ObservationDate'], y=weekly_growth,
                  mode='lines', name='Weekly Growth (%)', line=dict(color='purple')),
        row=4, col=2
    )
    
    fig.update_layout(height=1200, showlegend=False, title_text="COVID-19 Trend Analysis Dashboard")
    return fig

def generate_trend_report(data):
    """Generate a comprehensive trend analysis report"""
    report = []
    
    # Basic statistics
    report.append("=== COVID-19 TREND ANALYSIS REPORT ===\n")
    report.append(f"Analysis Period: {data['ObservationDate'].min()} to {data['ObservationDate'].max()}")
    report.append(f"Total Data Points: {len(data)}\n")
    
    # Cases analysis
    cases_trend = calculate_trend_indicators(data, 'New_Cases')
    report.append("--- NEW CASES ANALYSIS ---")
    report.append(f"Overall Trend Slope: {cases_trend['slope']:.2f} cases/day")
    report.append(f"Trend R-squared: {cases_trend['r_squared']:.3f}")
    report.append(f"Statistical Significance (p-value): {cases_trend['p_value']:.3f}")
    
    # Surge periods
    surge_periods, threshold = identify_surge_periods(data, 'New_Cases')
    report.append(f"Surge Threshold (90th percentile): {threshold:.0f} cases")
    report.append(f"Number of Surge Periods: {len(surge_periods)}")
    for i, (start, end) in enumerate(surge_periods, 1):
        report.append(f"  Surge {i}: {start} to {end}")
    
    # Anomalies
    anomaly_count = data['Anomaly_Cases'].sum()
    report.append(f"Anomalous Days: {anomaly_count} ({anomaly_count/len(data)*100:.1f}%)")
    
    # Deaths analysis
    deaths_trend = calculate_trend_indicators(data, 'New_Deaths')
    report.append("\n--- NEW DEATHS ANALYSIS ---")
    report.append(f"Overall Trend Slope: {deaths_trend['slope']:.2f} deaths/day")
    report.append(f"Trend R-squared: {deaths_trend['r_squared']:.3f}")
    
    # Moving averages insights
    report.append("\n--- MOVING AVERAGES INSIGHTS ---")
    latest_ma7_cases = data['New_Cases_MA_7'].iloc[-1]
    latest_ma14_cases = data['New_Cases_MA_14'].iloc[-1]
    report.append(f"Latest 7-day MA (Cases): {latest_ma7_cases:.1f}")
    report.append(f"Latest 14-day MA (Cases): {latest_ma14_cases:.1f}")
    
    if latest_ma7_cases > latest_ma14_cases:
        report.append("Short-term trend is ABOVE long-term trend (potential acceleration)")
    else:
        report.append("Short-term trend is BELOW long-term trend (potential deceleration)")
    
    # Changepoints
    changepoints = detect_changepoints(data, 'New_Cases')
    report.append(f"\n--- SIGNIFICANT CHANGEPOINTS ---")
    report.append(f"Number of Changepoints: {len(changepoints)}")
    for i, cp in enumerate(changepoints, 1):
        report.append(f"  Changepoint {i}: {cp}")
    
    return "\n".join(report)

def main():
    # Load processed data
    print("Loading processed data...")
    data = load_processed_data()
    data['ObservationDate'] = pd.to_datetime(data['ObservationDate'])
    
    # Generate trend analysis plots
    print("Creating trend analysis plots...")
    fig = create_trend_analysis_plots(data)
    fig.write_html('trend_analysis_dashboard.html')
    
    # Generate trend report
    print("Generating trend analysis report...")
    report = generate_trend_report(data)
    
    # Save report
    with open('trend_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Trend analysis completed!")
    print("\nFiles generated:")
    print("- trend_analysis_dashboard.html")
    print("- trend_analysis_report.txt")
    
    return data, report

if __name__ == "__main__":
    data, report = main()
    print("\n" + "="*50)
    print(report)

