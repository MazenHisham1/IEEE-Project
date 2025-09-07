# app.py — Enhanced COVID-19 Dashboard with Parquet and Plotly-Resampler
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import numpy as np

# Try to import plotly-resampler (optional enhancement)
try:
    from plotly_resampler import FigureResampler, FigureWidgetResampler
    RESAMPLER_AVAILABLE = True
except ImportError:
    RESAMPLER_AVAILABLE = False
    st.warning("plotly-resampler not installed. Install with: pip install plotly-resampler")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🌍 COVID-19 Dashboard (Enhanced with Parquet & Resampler)")

# ------------------------------
# Helpers & Constants
# ------------------------------
DATA_DIR = "data"
PARQUET_DIR = "data_parquet"

# Names of the CSVs
CSV_FILES = {
    "covid_daily": "covid_19_data.csv",
    "confirmed": "time_series_covid_19_confirmed.csv",
    "deaths": "time_series_covid_19_deaths.csv",
    "recovered": "time_series_covid_19_recovered.csv",
}

# Corresponding Parquet files
PARQUET_FILES = {
    "covid_daily": "covid_19_data.parquet",
    "confirmed": "time_series_covid_19_confirmed.parquet",
    "deaths": "time_series_covid_19_deaths.parquet",
    "recovered": "time_series_covid_19_recovered.parquet",
    "processed_ts": "processed_timeseries.parquet",
    "snapshot": "latest_snapshot.parquet"
}

# ------------------------------
# Parquet Conversion Functions
# ------------------------------
@st.cache_data(show_spinner=False)
def convert_csv_to_parquet(force_convert=False):
    """Convert CSV files to Parquet format for faster loading"""
    Path(PARQUET_DIR).mkdir(exist_ok=True)
    
    conversion_log = []
    
    for key, csv_file in CSV_FILES.items():
        csv_path = Path(DATA_DIR) / csv_file
        parquet_path = Path(PARQUET_DIR) / PARQUET_FILES[key]
        
        # Check if conversion is needed
        if not parquet_path.exists() or force_convert:
            if csv_path.exists():
                start_time = time.time()
                df = pd.read_csv(csv_path)
                df.to_parquet(parquet_path, compression='snappy', engine='pyarrow')
                elapsed = time.time() - start_time
                
                csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
                parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
                compression_ratio = (1 - parquet_size/csv_size) * 100
                
                conversion_log.append({
                    'File': csv_file,
                    'CSV Size (MB)': f"{csv_size:.2f}",
                    'Parquet Size (MB)': f"{parquet_size:.2f}",
                    'Compression': f"{compression_ratio:.1f}%",
                    'Time (s)': f"{elapsed:.2f}"
                })
    
    return pd.DataFrame(conversion_log) if conversion_log else None

# ------------------------------
# Load + Preprocess with Parquet
# ------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_all_parquet():
    """Load data from Parquet files (much faster than CSV)"""
    start = time.time()
    
    # First ensure Parquet files exist
    convert_csv_to_parquet()
    
    # Check if processed files exist
    processed_ts_path = Path(PARQUET_DIR) / PARQUET_FILES["processed_ts"]
    snapshot_path = Path(PARQUET_DIR) / PARQUET_FILES["snapshot"]
    
    if processed_ts_path.exists() and snapshot_path.exists():
        # Load pre-processed data (super fast!)
        ts = pd.read_parquet(processed_ts_path)
        snapshot = pd.read_parquet(snapshot_path)
        covid_daily = pd.read_parquet(Path(PARQUET_DIR) / PARQUET_FILES["covid_daily"])
        latest_date = pd.to_datetime(ts["Date"].max())
    else:
        # Load raw Parquet files and process
        covid_daily = pd.read_parquet(Path(PARQUET_DIR) / PARQUET_FILES["covid_daily"])
        confirmed = pd.read_parquet(Path(PARQUET_DIR) / PARQUET_FILES["confirmed"])
        deaths = pd.read_parquet(Path(PARQUET_DIR) / PARQUET_FILES["deaths"])
        recovered = pd.read_parquet(Path(PARQUET_DIR) / PARQUET_FILES["recovered"])
        
        # Process data (same as before)
        def melt_and_group(ts_df, value_name):
            id_vars = [col for col in ["Province/State", "Country/Region", "Lat", "Long"] 
                      if col in ts_df.columns]
            value_vars = [c for c in ts_df.columns if c not in id_vars]
            m = ts_df.melt(id_vars=id_vars, value_vars=value_vars, 
                          var_name="Date", value_name=value_name)
            m["Date"] = pd.to_datetime(m["Date"], errors="coerce")
            grouped = m.groupby(["Country/Region", "Date"], as_index=False)[value_name].sum()
            return grouped
        
        confirmed_g = melt_and_group(confirmed, "Confirmed")
        deaths_g = melt_and_group(deaths, "Deaths")
        recovered_g = melt_and_group(recovered, "Recovered")
        
        # Merge
        ts = confirmed_g.merge(deaths_g, on=["Country/Region", "Date"], how="outer")
        ts = ts.merge(recovered_g, on=["Country/Region", "Date"], how="outer")
        
        # Fill missing values
        for c in ["Confirmed", "Deaths", "Recovered"]:
            if c in ts.columns:
                ts[c] = ts[c].fillna(0).astype(int)
        
        # Create snapshot
        latest_date = ts["Date"].max()
        snapshot = ts[ts["Date"] == latest_date].groupby(
            "Country/Region", as_index=False
        )[["Confirmed", "Deaths", "Recovered"]].sum()
        
        # Save processed data to Parquet for next time
        ts.to_parquet(processed_ts_path, compression='snappy')
        snapshot.to_parquet(snapshot_path, compression='snappy')
    
    elapsed = time.time() - start
    
    return {
        "covid_daily": covid_daily,
        "ts": ts,
        "snapshot": snapshot,
        "latest_date": latest_date,
        "load_time": elapsed
    }

# ------------------------------
# Data Loading Section
# ------------------------------
with st.spinner("Loading data..."):
    data_bundle = load_and_prepare_all_parquet()
    
covid_daily = data_bundle["covid_daily"]
ts = data_bundle["ts"]
snapshot = data_bundle["snapshot"]
latest_date = data_bundle["latest_date"]

# Show load time comparison
with st.expander("⚡ Performance Metrics"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Data Load Time", f"{data_bundle['load_time']:.2f}s")
        st.caption("Using Parquet format for optimal performance")
    
    with col2:
        conversion_df = convert_csv_to_parquet()
        if conversion_df is not None and not conversion_df.empty:
            st.caption("File Compression Stats:")
            st.dataframe(conversion_df, hide_index=True, use_container_width=True)

# ------------------------------
# Global KPIs
# ------------------------------
st.subheader("🌐 Global Overview")
k1, k2, k3, k4 = st.columns(4)
total_confirmed = int(snapshot["Confirmed"].sum())
total_deaths = int(snapshot["Deaths"].sum())
total_recovered = int(snapshot["Recovered"].sum())
mortality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0

k1.metric("Confirmed", f"{total_confirmed:,}")
k2.metric("Deaths", f"{total_deaths:,}")
k3.metric("Recovered", f"{total_recovered:,}")
k4.metric("Mortality Rate", f"{mortality_rate:.2f}%")
st.caption(f"Latest date in dataset: {latest_date.date()}")

# ------------------------------
# Sidebar / Filters
# ------------------------------
st.sidebar.header("🔧 Filters")

# Performance options
st.sidebar.subheader("Performance Settings")
use_resampler = st.sidebar.checkbox(
    "Use Plotly Resampler", 
    value=RESAMPLER_AVAILABLE,
    disabled=not RESAMPLER_AVAILABLE,
    help="Automatically downsample large datasets for smoother visualization"
)

if use_resampler and RESAMPLER_AVAILABLE:
    max_points = st.sidebar.slider(
        "Max points per trace",
        min_value=500,
        max_value=10000,
        value=2000,
        step=500,
        help="Lower values = faster rendering"
    )
else:
    max_points = None

# Country filter
all_countries = sorted(ts["Country/Region"].unique())
search = st.sidebar.text_input("Search country", value="", placeholder="Type to filter...")
if search:
    filtered_countries = [c for c in all_countries if search.lower() in c.lower()]
    if not filtered_countries:
        filtered_countries = all_countries
else:
    filtered_countries = all_countries

default_country = "US" if "US" in filtered_countries else filtered_countries[0]
country = st.sidebar.selectbox(
    "Select a Country", 
    options=filtered_countries,
    index=filtered_countries.index(default_country) if default_country in filtered_countries else 0
)

# Date range
country_ts = ts[ts["Country/Region"] == country]
min_date = country_ts["Date"].min()
max_date = country_ts["Date"].max()
date_range = st.sidebar.slider(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Moving average
show_ma = st.sidebar.checkbox("Show moving average", value=True)
ma_window = st.sidebar.number_input(
    "MA window (days)",
    min_value=1,
    max_value=30,
    value=7
) if show_ma else 7

# ------------------------------
# Choropleth Map
# ------------------------------
@st.cache_data(show_spinner=False)
def make_choropleth(snapshot_df):
    fig = px.choropleth(
        snapshot_df,
        locations="Country/Region",
        locationmode="country names",
        color="Confirmed",
        color_continuous_scale="Reds",
        title="Total Confirmed Cases by Country",
        hover_data=["Deaths", "Recovered"]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        geo=dict(showframe=False, showcoastlines=True)
    )
    return fig

st.subheader("🗺️ Global COVID-19 Map")
fig_map = make_choropleth(snapshot)
st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

# ------------------------------
# Country Trends with Resampler
# ------------------------------
st.subheader(f"📊 Trends for {country}")

@st.cache_data(show_spinner=False)
def build_trend_figure_with_resampler(ts_df, country_name, date_min, date_max, 
                                      show_ma_flag=True, ma_w=7, 
                                      use_resampler_flag=False, max_pts=2000):
    """Build trend figure with optional resampling for large datasets"""
    df = ts_df[ts_df["Country/Region"] == country_name].copy()
    mask = (df["Date"].dt.date >= date_min) & (df["Date"].dt.date <= date_max)
    df = df.loc[mask].sort_values("Date")
    
    if df.empty:
        return px.line(title=f"No data for {country_name} in selected range")
    
    # Calculate moving averages
    if show_ma_flag:
        df = df.set_index("Date").sort_index()
        df[["Confirmed_MA", "Deaths_MA", "Recovered_MA"]] = \
            df[["Confirmed", "Deaths", "Recovered"]].rolling(
                window=ma_w, min_periods=1
            ).mean()
        df = df.reset_index()
    
    # Create figure with or without resampler
    if use_resampler_flag and RESAMPLER_AVAILABLE:
        # Use FigureResampler for automatic downsampling
        fig = FigureResampler(go.Figure(), default_n_shown_samples=max_pts)
        
        # Add traces
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Confirmed"], 
                                name="Confirmed", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Deaths"], 
                                name="Deaths", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Recovered"], 
                                name="Recovered", line=dict(color='green')))
        
        if show_ma_flag:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Confirmed_MA"], 
                                    name="Confirmed (MA)", 
                                    line=dict(dash='dash', color='lightblue')))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Deaths_MA"], 
                                    name="Deaths (MA)", 
                                    line=dict(dash='dash', color='pink')))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Recovered_MA"], 
                                    name="Recovered (MA)", 
                                    line=dict(dash='dash', color='lightgreen')))
        
        fig.update_layout(
            title=f"COVID-19 Trends in {country_name} (Resampled)",
            xaxis_title="Date",
            yaxis_title="Count",
            hovermode='x unified'
        )
    else:
        # Standard Plotly figure
        fig = px.line(df, x="Date", y=["Confirmed", "Deaths", "Recovered"],
                     labels={"value": "Count", "variable": "Metric"},
                     title=f"COVID-19 Trends in {country_name}")
        
        if show_ma_flag:
            # Add MA traces
            for col, color in [("Confirmed_MA", "lightblue"), 
                              ("Deaths_MA", "pink"), 
                              ("Recovered_MA", "lightgreen")]:
                fig.add_scatter(x=df["Date"], y=df[col], name=col.replace("_", " "),
                              line=dict(dash='dash', color=color))
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=500
    )
    
    return fig

# Build and display trend figure
fig_trend = build_trend_figure_with_resampler(
    ts, country, date_range[0], date_range[1],
    show_ma, ma_window, use_resampler, max_points
)

if use_resampler and RESAMPLER_AVAILABLE:
    st.info("📌 Using Plotly Resampler: Zoom in to see more detail automatically loaded")
    
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------
# Country Statistics
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📌 Latest Stats for {country}")
    row = snapshot[snapshot["Country/Region"] == country]
    if not row.empty:
        latest = row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Confirmed", f"{int(latest['Confirmed']):,}")
        c2.metric("Deaths", f"{int(latest['Deaths']):,}")
        c3.metric("Recovered", f"{int(latest['Recovered']):,}")
        
        # Calculate additional metrics
        if latest['Confirmed'] > 0:
            death_rate = (latest['Deaths'] / latest['Confirmed']) * 100
            recovery_rate = (latest['Recovered'] / latest['Confirmed']) * 100
            st.caption(f"Death Rate: {death_rate:.2f}% | Recovery Rate: {recovery_rate:.2f}%")
    else:
        st.info("No data available for this country")

with col2:
    st.subheader("📈 Growth Analysis")
    # Calculate daily new cases
    country_data = ts[ts["Country/Region"] == country].sort_values("Date")
    if len(country_data) > 1:
        country_data["New_Cases"] = country_data["Confirmed"].diff()
        country_data["Growth_Rate"] = country_data["Confirmed"].pct_change() * 100
        
        # Last 7 days stats
        last_7_days = country_data.tail(7)
        avg_new_cases = last_7_days["New_Cases"].mean()
        avg_growth_rate = last_7_days["Growth_Rate"].mean()
        
        c1, c2 = st.columns(2)
        c1.metric("Avg New Cases (7d)", f"{int(avg_new_cases):,}")
        c2.metric("Avg Growth Rate (7d)", f"{avg_growth_rate:.2f}%")

# ------------------------------
# Data Export Section
# ------------------------------
st.subheader("💾 Data Export")
tab1, tab2, tab3 = st.tabs(["Download CSV", "Download Parquet", "View Raw Data"])

with tab1:
    filtered = ts[
        (ts["Country/Region"] == country) & 
        (ts["Date"].dt.date >= date_range[0]) & 
        (ts["Date"].dt.date <= date_range[1])
    ]
    if not filtered.empty:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_bytes,
            file_name=f"{country}_covid_data.csv",
            mime="text/csv"
        )

with tab2:
    if not filtered.empty:
        import io
        buffer = io.BytesIO()
        filtered.to_parquet(buffer, compression='snappy', engine='pyarrow')
        parquet_bytes = buffer.getvalue()
        st.download_button(
            "⬇️ Download as Parquet",
            data=parquet_bytes,
            file_name=f"{country}_covid_data.parquet",
            mime="application/octet-stream"
        )
        st.caption("💡 Parquet files are ~70% smaller and load 5-10x faster than CSV")

with tab3:
    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True
    )
    if len(filtered) > 100:
        st.caption(f"Showing first 100 of {len(filtered)} rows")

# ------------------------------
# Footer with instructions
# ------------------------------
with st.expander("🚀 Deployment Instructions"):
    st.markdown("""
    ### Deploying to Streamlit Community Cloud
    
    1. **Prepare your GitHub repository:**
       ```
       your-repo/
       ├── app.py (this file)
       ├── requirements.txt
       ├── data/
       │   ├── covid_19_data.csv
       │   ├── time_series_covid_19_confirmed.csv
       │   ├── time_series_covid_19_deaths.csv
       │   └── time_series_covid_19_recovered.csv
       └── .gitignore
       ```
    
    2. **Create requirements.txt:**
       ```
       streamlit>=1.28.0
       pandas>=2.0.0
       plotly>=5.17.0
       pyarrow>=14.0.0
       plotly-resampler>=0.9.0
       ```
    
    3. **Create .gitignore:**
       ```
       data_parquet/
       *.pyc
       __pycache__/
       .streamlit/
       ```
    
    4. **Push to GitHub:**
       ```bash
       git add .
       git commit -m "COVID dashboard with Parquet support"
       git push origin main
       ```
    
    5. **Deploy on Streamlit Cloud:**
       - Go to [share.streamlit.io](https://share.streamlit.io)
       - Click "New app"
       - Connect your GitHub repository
       - Select branch: main
       - Main file path: app.py
       - Click "Deploy"
    
    6. **Environment Variables (optional):**
       - In Streamlit Cloud settings, you can add secrets
       - Use for API keys if adding external data sources
    
    ### Performance Tips:
    - Parquet files load 5-10x faster than CSV
    - Use st.cache_data for expensive operations
    - Enable plotly-resampler for large datasets
    - Consider data pagination for very large tables
    """)

# Info about the enhancements
with st.sidebar.expander("ℹ️ About Enhancements"):
    st.markdown("""
    ### New Features:
    1. **Parquet Format**: Automatic conversion from CSV to Parquet for 70% file size reduction
    2. **Plotly Resampler**: Smart downsampling for smooth visualization of large datasets  
    3. **Performance Metrics**: Track load times and compression ratios
    4. **Enhanced Analytics**: Growth rates, moving averages, mortality metrics
    5. **Export Options**: Download data as CSV or Parquet
    6. **Deployment Ready**: Configured for Streamlit Community Cloud
    """)