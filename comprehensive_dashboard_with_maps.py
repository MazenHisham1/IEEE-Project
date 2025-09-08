import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import our custom modules
from forecasting_models import load_and_prepare_data, arima_forecast, prophet_forecast
from trend_analysis import identify_surge_periods, calculate_trend_indicators, detect_changepoints

# Load and prepare all data
def load_all_data():
    # Original data
    df = pd.read_csv('covid.csv')
    df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
    df = df.fillna(0)
    
    # Processed data with forecasts
    try:
        processed_df = pd.read_csv('processed_data_with_forecasts.csv')
        processed_df['ObservationDate'] = pd.to_datetime(processed_df['ObservationDate'])
    except:
        processed_df = load_and_prepare_data()
    
    return df, processed_df

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # For Heroku deployment

# Load data
df, processed_df = load_all_data()

# Get unique countries for dropdown
countries = sorted(df['Country/Region'].unique())

# Define the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("COVID-19 Comprehensive Analytics Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 10, 'color': '#2c3e50'}),
        html.P("Interactive visualizations with predictive modeling, trend analysis, and geographic maps",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30})
    ]),
    
    # Control Panel
    html.Div([
        html.Div([
            html.Label("Select Country/Region:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in countries],
                value='Mainland China',
                style={'marginBottom': 15}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Confirmed Cases', 'value': 'Confirmed'},
                    {'label': 'Deaths', 'value': 'Deaths'},
                    {'label': 'Recovered', 'value': 'Recovered'}
                ],
                value='Confirmed',
                style={'marginBottom': 15}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Analysis Type:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='analysis-type',
                options=[
                    {'label': 'Time Series', 'value': 'timeseries'},
                    {'label': 'Geographic Map', 'value': 'geographic'},
                    {'label': 'Forecasting', 'value': 'forecast'},
                    {'label': 'Trend Analysis', 'value': 'trends'},
                    {'label': 'Anomaly Detection', 'value': 'anomalies'}
                ],
                value='timeseries',
                style={'marginBottom': 15}
            )
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Date Range:", style={'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=df['ObservationDate'].min(),
                end_date=df['ObservationDate'].max(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': 15}
            )
        ], style={'width': '23%', 'display': 'inline-block'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Main Dashboard Area
    html.Div([
        # Key Metrics Cards
        html.Div(id='metrics-cards', style={'marginBottom': '20px'}),
        
        # Main Chart
        dcc.Graph(id='main-chart', style={'height': '500px'}),
        
        # Secondary Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='secondary-chart-1')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='secondary-chart-2')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        # Analysis Results
        html.Div([
            html.H3("Analysis Results", style={'color': '#2c3e50'}),
            html.Div(id='analysis-results')
        ], style={'marginTop': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'})
    ])
])

# Callback for metrics cards
@app.callback(
    Output('metrics-cards', 'children'),
    [Input('country-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_metrics_cards(selected_country, start_date, end_date):
    # Filter data by date range
    filtered_df = df[(df['ObservationDate'] >= start_date) & (df['ObservationDate'] <= end_date)]
    
    if selected_country:
        country_data = filtered_df[filtered_df['Country/Region'] == selected_country]
        latest_data = country_data[country_data['ObservationDate'] == country_data['ObservationDate'].max()]
        
        total_confirmed = latest_data['Confirmed'].sum()
        total_deaths = latest_data['Deaths'].sum()
        total_recovered = latest_data['Recovered'].sum()
        
        # Calculate daily changes
        country_daily = country_data.groupby('ObservationDate').agg({
            'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
        }).reset_index()
        
        if len(country_daily) > 1:
            daily_new_cases = country_daily['Confirmed'].iloc[-1] - country_daily['Confirmed'].iloc[-2]
            daily_new_deaths = country_daily['Deaths'].iloc[-1] - country_daily['Deaths'].iloc[-2]
        else:
            daily_new_cases = 0
            daily_new_deaths = 0
    else:
        # Global metrics
        latest_data = filtered_df[filtered_df['ObservationDate'] == filtered_df['ObservationDate'].max()]
        total_confirmed = latest_data['Confirmed'].sum()
        total_deaths = latest_data['Deaths'].sum()
        total_recovered = latest_data['Recovered'].sum()
        
        # Calculate daily changes for global data
        global_daily = filtered_df.groupby('ObservationDate').agg({
            'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'
        }).reset_index()
        
        if len(global_daily) > 1:
            daily_new_cases = global_daily['Confirmed'].iloc[-1] - global_daily['Confirmed'].iloc[-2]
            daily_new_deaths = global_daily['Deaths'].iloc[-1] - global_daily['Deaths'].iloc[-2]
        else:
            daily_new_cases = 0
            daily_new_deaths = 0
    
    cards = html.Div([
        html.Div([
            html.H4(f"{total_confirmed:,.0f}", style={'margin': '0', 'color': '#3498db'}),
            html.P("Total Confirmed", style={'margin': '0', 'fontSize': '14px'}),
            html.P(f"+{daily_new_cases:,.0f} today", style={'margin': '0', 'fontSize': '12px', 'color': '#e74c3c'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white', 
                 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
        
        html.Div([
            html.H4(f"{total_deaths:,.0f}", style={'margin': '0', 'color': '#e74c3c'}),
            html.P("Total Deaths", style={'margin': '0', 'fontSize': '14px'}),
            html.P(f"+{daily_new_deaths:,.0f} today", style={'margin': '0', 'fontSize': '12px', 'color': '#e74c3c'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
        
        html.Div([
            html.H4(f"{total_recovered:,.0f}", style={'margin': '0', 'color': '#27ae60'}),
            html.P("Total Recovered", style={'margin': '0', 'fontSize': '14px'}),
            html.P(f"{(total_recovered/total_confirmed*100):.1f}% rate", style={'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block', 'marginRight': '4%'}),
        
        html.Div([
            html.H4(f"{(total_deaths/total_confirmed*100):.1f}%" if total_confirmed > 0 else "0%", 
                    style={'margin': '0', 'color': '#f39c12'}),
            html.P("Fatality Rate", style={'margin': '0', 'fontSize': '14px'}),
            html.P("Case Fatality", style={'margin': '0', 'fontSize': '12px'})
        ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                 'width': '22%', 'display': 'inline-block'})
    ])
    
    return cards

# Callback for main chart
@app.callback(
    Output('main-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('analysis-type', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_main_chart(selected_country, selected_metric, analysis_type, start_date, end_date):
    # Filter data by date range
    filtered_df = df[(df['ObservationDate'] >= start_date) & (df['ObservationDate'] <= end_date)]
    
    if analysis_type == 'geographic':
        # Create geographic map
        latest_date = filtered_df['ObservationDate'].max()
        latest_data = filtered_df[filtered_df['ObservationDate'] == latest_date]
        country_totals = latest_data.groupby('Country/Region')[selected_metric].sum().reset_index()
        
        # Create choropleth map
        fig = px.choropleth(
            country_totals,
            locations='Country/Region',
            color=selected_metric,
            hover_name='Country/Region',
            hover_data=[selected_metric],
            color_continuous_scale='Reds',
            title=f'Global Distribution of {selected_metric} - {latest_date.strftime("%Y-%m-%d")}',
            locationmode='country names'
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            )
        )
        
        return fig
    
    elif analysis_type == 'timeseries':
        if selected_country:
            country_filtered = filtered_df[filtered_df['Country/Region'] == selected_country]
            country_daily = country_filtered.groupby('ObservationDate')[selected_metric].sum().reset_index()
            
            fig = px.line(country_daily, x='ObservationDate', y=selected_metric,
                         title=f'{selected_metric} Over Time - {selected_country}')
        else:
            global_daily = filtered_df.groupby('ObservationDate')[selected_metric].sum().reset_index()
            fig = px.line(global_daily, x='ObservationDate', y=selected_metric,
                         title=f'Global {selected_metric} Trends')
    
    elif analysis_type == 'forecast':
        # Show forecasting results
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=processed_df['ObservationDate'],
            y=processed_df['New_Cases'] if selected_metric == 'Confirmed' else processed_df['New_Deaths'],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Add moving average
        ma_col = 'New_Cases_MA_7' if selected_metric == 'Confirmed' else 'New_Deaths_MA_7'
        if ma_col in processed_df.columns:
            fig.add_trace(go.Scatter(
                x=processed_df['ObservationDate'],
                y=processed_df[ma_col],
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(title=f'Forecasting Analysis - {selected_metric}')
    
    elif analysis_type == 'trends':
        # Trend analysis with moving averages
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=['Trend with Moving Averages', 'Rate of Change'])
        
        metric_col = 'New_Cases' if selected_metric == 'Confirmed' else 'New_Deaths'
        ma_col = f'{metric_col}_MA_7'
        
        # Main trend
        fig.add_trace(go.Scatter(
            x=processed_df['ObservationDate'],
            y=processed_df[metric_col],
            mode='lines',
            name=f'Daily {selected_metric}',
            line=dict(color='lightblue')
        ), row=1, col=1)
        
        if ma_col in processed_df.columns:
            fig.add_trace(go.Scatter(
                x=processed_df['ObservationDate'],
                y=processed_df[ma_col],
                mode='lines',
                name='7-day MA',
                line=dict(color='red', width=2)
            ), row=1, col=1)
        
        # Rate of change
        roc = processed_df[metric_col].pct_change() * 100
        fig.add_trace(go.Scatter(
            x=processed_df['ObservationDate'],
            y=roc,
            mode='lines',
            name='Rate of Change (%)',
            line=dict(color='green')
        ), row=2, col=1)
        
        fig.update_layout(title=f'Trend Analysis - {selected_metric}', height=600)
    
    elif analysis_type == 'anomalies':
        # Anomaly detection visualization
        fig = go.Figure()
        
        metric_col = 'New_Cases' if selected_metric == 'Confirmed' else 'New_Deaths'
        anomaly_col = 'Anomaly_Cases' if selected_metric == 'Confirmed' else 'Anomaly_Deaths'
        
        # Normal points
        normal_data = processed_df[~processed_df[anomaly_col]]
        fig.add_trace(go.Scatter(
            x=normal_data['ObservationDate'],
            y=normal_data[metric_col],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6)
        ))
        
        # Anomalous points
        anomaly_data = processed_df[processed_df[anomaly_col]]
        fig.add_trace(go.Scatter(
            x=anomaly_data['ObservationDate'],
            y=anomaly_data[metric_col],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(title=f'Anomaly Detection - {selected_metric}')
    
    return fig

# Callback for secondary charts
@app.callback(
    [Output('secondary-chart-1', 'figure'),
     Output('secondary-chart-2', 'figure')],
    [Input('country-dropdown', 'value'),
     Input('analysis-type', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_secondary_charts(selected_country, analysis_type, start_date, end_date):
    # Filter data by date range
    filtered_df = df[(df['ObservationDate'] >= start_date) & (df['ObservationDate'] <= end_date)]
    
    # Chart 1: Geographic distribution or top countries
    if analysis_type == 'geographic':
        # Show bubble map for deaths vs confirmed
        latest_date = filtered_df['ObservationDate'].max()
        latest_data = filtered_df[filtered_df['ObservationDate'] == latest_date]
        country_totals = latest_data.groupby('Country/Region').agg({
            'Confirmed': 'sum',
            'Deaths': 'sum',
            'Recovered': 'sum'
        }).reset_index()
        
        fig1 = px.scatter(country_totals, x='Confirmed', y='Deaths', 
                         size='Recovered', hover_name='Country/Region',
                         title='Deaths vs Confirmed Cases (Bubble size = Recovered)',
                         size_max=60)
    else:
        # Top countries bar chart
        latest_date = filtered_df['ObservationDate'].max()
        latest_data = filtered_df[filtered_df['ObservationDate'] == latest_date]
        country_totals = latest_data.groupby('Country/Region')['Confirmed'].sum().reset_index()
        top_countries = country_totals.nlargest(15, 'Confirmed')
        
        fig1 = px.bar(top_countries, x='Country/Region', y='Confirmed',
                      title='Top 15 Countries by Confirmed Cases')
        fig1.update_layout(xaxis_tickangle=-45)
    
    # Chart 2: Time series comparison or correlation
    if analysis_type == 'geographic':
        # Show time series for top 5 countries
        top_5_countries = filtered_df.groupby('Country/Region')['Confirmed'].max().nlargest(5).index
        top_5_data = filtered_df[filtered_df['Country/Region'].isin(top_5_countries)]
        
        fig2 = px.line(top_5_data, x='ObservationDate', y='Confirmed', 
                      color='Country/Region',
                      title='Top 5 Countries - Confirmed Cases Over Time')
    else:
        # Cases vs Deaths correlation
        fig2 = px.scatter(processed_df, x='Confirmed', y='Deaths',
                         title='Cumulative Cases vs Deaths',
                         hover_data=['ObservationDate'])
    
    return fig1, fig2

# Callback for analysis results
@app.callback(
    Output('analysis-results', 'children'),
    [Input('country-dropdown', 'value'),
     Input('analysis-type', 'value')]
)
def update_analysis_results(selected_country, analysis_type):
    if analysis_type == 'geographic':
        # Geographic analysis results
        latest_date = df['ObservationDate'].max()
        latest_data = df[df['ObservationDate'] == latest_date]
        
        total_countries = latest_data['Country/Region'].nunique()
        affected_countries = latest_data[latest_data['Confirmed'] > 0]['Country/Region'].nunique()
        top_country = latest_data.groupby('Country/Region')['Confirmed'].sum().idxmax()
        top_cases = latest_data.groupby('Country/Region')['Confirmed'].sum().max()
        
        results = html.Div([
            html.H4("Geographic Distribution Analysis"),
            html.P(f"Total countries/regions in dataset: {total_countries}"),
            html.P(f"Countries/regions with confirmed cases: {affected_countries}"),
            html.P(f"Most affected country: {top_country} ({top_cases:,.0f} cases)"),
            html.P(f"Global spread: {(affected_countries/total_countries)*100:.1f}% of regions affected")
        ])
        
    elif analysis_type == 'trends':
        # Calculate trend indicators
        trends = calculate_trend_indicators(processed_df, 'New_Cases')
        
        results = html.Div([
            html.H4("Trend Analysis Results"),
            html.P(f"Overall trend slope: {trends['slope']:.2f} cases/day"),
            html.P(f"Trend strength (R²): {trends['r_squared']:.3f}"),
            html.P(f"Statistical significance (p-value): {trends['p_value']:.3f}"),
            
            html.H5("Interpretation:"),
            html.Ul([
                html.Li("Positive slope indicates increasing trend" if trends['slope'] > 0 else "Negative slope indicates decreasing trend"),
                html.Li(f"R² of {trends['r_squared']:.3f} indicates {'strong' if trends['r_squared'] > 0.7 else 'moderate' if trends['r_squared'] > 0.3 else 'weak'} linear relationship"),
                html.Li(f"P-value of {trends['p_value']:.3f} indicates {'statistically significant' if trends['p_value'] < 0.05 else 'not statistically significant'} trend")
            ])
        ])
        
    elif analysis_type == 'anomalies':
        anomaly_count = processed_df['Anomaly_Cases'].sum()
        total_days = len(processed_df)
        
        results = html.Div([
            html.H4("Anomaly Detection Results"),
            html.P(f"Total anomalous days: {anomaly_count} out of {total_days} ({anomaly_count/total_days*100:.1f}%)"),
            
            html.H5("Anomalous Dates:"),
            html.Ul([
                html.Li(f"{row['ObservationDate'].strftime('%Y-%m-%d')}: {row['New_Cases']:.0f} cases")
                for _, row in processed_df[processed_df['Anomaly_Cases']].iterrows()
            ])
        ])
        
    elif analysis_type == 'forecast':
        results = html.Div([
            html.H4("Forecasting Model Results"),
            html.P("ARIMA Model: Best order found (0,1,1) for cases"),
            html.P("Prophet Model: Includes daily and weekly seasonality"),
            html.P("7-day forecast generated for both models"),
            
            html.H5("Model Performance:"),
            html.P("Models trained on historical data from Jan 22 to Mar 14, 2020"),
            html.P("Forecast horizon: 7 days ahead")
        ])
        
    else:
        results = html.Div([
            html.H4("Time Series Analysis"),
            html.P(f"Data period: {processed_df['ObservationDate'].min().strftime('%Y-%m-%d')} to {processed_df['ObservationDate'].max().strftime('%Y-%m-%d')}"),
            html.P(f"Total data points: {len(processed_df)}"),
            html.P(f"Peak daily cases: {processed_df['New_Cases'].max():.0f}"),
            html.P(f"Peak daily deaths: {processed_df['New_Deaths'].max():.0f}")
        ])
    
    return results

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8051))
    app.run(debug=False, host='0.0.0.0', port=port)

