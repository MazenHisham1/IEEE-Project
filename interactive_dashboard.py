import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load and preprocess data
def load_data():
    df = pd.read_csv('covid.csv')
    df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
    df = df.fillna(0)
    
    # Group by date and country for time series analysis
    daily_global = df.groupby('ObservationDate').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum', 
        'Recovered': 'sum'
    }).reset_index()
    
    # Calculate daily new cases
    daily_global['New_Cases'] = daily_global['Confirmed'].diff().fillna(0)
    daily_global['New_Deaths'] = daily_global['Deaths'].diff().fillna(0)
    daily_global['New_Recovered'] = daily_global['Recovered'].diff().fillna(0)
    
    # Calculate 7-day moving averages
    daily_global['MA_7_Cases'] = daily_global['New_Cases'].rolling(window=7).mean()
    daily_global['MA_7_Deaths'] = daily_global['New_Deaths'].rolling(window=7).mean()
    
    return df, daily_global

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data
df, daily_global = load_data()

# Get unique countries for dropdown
countries = sorted(df['Country/Region'].unique())

# Define the layout
app.layout = html.Div([
    html.H1("COVID-19 Interactive Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Controls section
    html.Div([
        html.Div([
            html.Label("Select Country/Region:"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in countries],
                value='Mainland China',
                style={'marginBottom': 20}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Confirmed Cases', 'value': 'Confirmed'},
                    {'label': 'Deaths', 'value': 'Deaths'},
                    {'label': 'Recovered', 'value': 'Recovered'}
                ],
                value='Confirmed',
                style={'marginBottom': 20}
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    # Charts section
    html.Div([
        # Time series chart
        dcc.Graph(id='time-series-chart'),
        
        # Global trends chart
        dcc.Graph(id='global-trends-chart'),
        
        # Geographic distribution
        dcc.Graph(id='geo-chart'),
        
        # Moving averages chart
        dcc.Graph(id='moving-averages-chart')
    ])
])

# Callback for time series chart
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('country-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_time_series(selected_country, selected_metric):
    filtered_df = df[df['Country/Region'] == selected_country]
    country_daily = filtered_df.groupby('ObservationDate')[selected_metric].sum().reset_index()
    
    fig = px.line(country_daily, x='ObservationDate', y=selected_metric,
                  title=f'{selected_metric} Over Time - {selected_country}')
    fig.update_layout(xaxis_title="Date", yaxis_title=selected_metric)
    return fig

# Callback for global trends
@app.callback(
    Output('global-trends-chart', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_global_trends(selected_metric):
    fig = px.line(daily_global, x='ObservationDate', y=selected_metric,
                  title=f'Global {selected_metric} Trends')
    fig.update_layout(xaxis_title="Date", yaxis_title=selected_metric)
    return fig

# Callback for geographic distribution
@app.callback(
    Output('geo-chart', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_geo_chart(selected_metric):
    # Get latest data for each country
    latest_date = df['ObservationDate'].max()
    latest_data = df[df['ObservationDate'] == latest_date]
    country_totals = latest_data.groupby('Country/Region')[selected_metric].sum().reset_index()
    
    fig = px.bar(country_totals.head(20), x='Country/Region', y=selected_metric,
                 title=f'Top 20 Countries by {selected_metric}')
    fig.update_layout(xaxis_title="Country/Region", yaxis_title=selected_metric,
                      xaxis={'categoryorder': 'total descending'})
    return fig

# Callback for moving averages
@app.callback(
    Output('moving-averages-chart', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_moving_averages(selected_metric):
    if selected_metric == 'Confirmed':
        y_col = 'MA_7_Cases'
        title = '7-Day Moving Average of New Cases'
    elif selected_metric == 'Deaths':
        y_col = 'MA_7_Deaths'
        title = '7-Day Moving Average of New Deaths'
    else:
        # For recovered, use the raw data
        y_col = 'New_Recovered'
        title = 'New Recovered Cases'
    
    fig = px.line(daily_global, x='ObservationDate', y=y_col, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title="Count")
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)

