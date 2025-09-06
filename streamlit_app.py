import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🌍 COVID-19 & World Data Dashboard")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    covid_daily = pd.read_csv("data/covid_19_data.csv")
    confirmed = pd.read_csv("data/time_series_covid_19_confirmed.csv")
    deaths = pd.read_csv("data/time_series_covid_19_deaths.csv")
    recovered = pd.read_csv("data/time_series_covid_19_recovered.csv")
    return covid_daily, confirmed, deaths, recovered

covid_daily, confirmed, deaths, recovered = load_data()

# ------------------------------
# Global KPIs
# ------------------------------
st.subheader("🌐 Global Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Confirmed", f"{covid_daily['Confirmed'].sum():,}")
col2.metric("Deaths", f"{covid_daily['Deaths'].sum():,}")
col3.metric("Recovered", f"{covid_daily['Recovered'].sum():,}")

# ------------------------------
# World Map
# ------------------------------
st.subheader("🗺️ Confirmed Cases by Country")
map_data = covid_daily.groupby("Country/Region")[["Confirmed"]].sum().reset_index()
fig_map = px.choropleth(
    map_data,
    locations="Country/Region",
    locationmode="country names",
    color="Confirmed",
    color_continuous_scale="Reds",
    title="Total Confirmed Cases"
)
st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------
# Country Drilldown
# ------------------------------
st.subheader("📊 Country Trends")

# Country Selector
all_countries = sorted(confirmed["Country/Region"].unique())
country = st.selectbox("Select a Country", all_countries, index=all_countries.index("US") if "US" in all_countries else 0)

# Function to melt time series
def melt_time_series(df, value_name):
    df_country = df[df["Country/Region"] == country].drop(columns=["Lat", "Long", "Country/Region"])
    df_country = df_country.sum(axis=0)  # aggregate over provinces
    df_country = df_country.reset_index()
    df_country.columns = ["Date", value_name]
    df_country["Date"] = pd.to_datetime(df_country["Date"], errors="coerce")
    df_country = df_country.dropna()
    return df_country

confirmed_ts = melt_time_series(confirmed, "Confirmed")
deaths_ts = melt_time_series(deaths, "Deaths")
recovered_ts = melt_time_series(recovered, "Recovered")

# Merge into one DataFrame
trend_df = confirmed_ts.merge(deaths_ts, on="Date").merge(recovered_ts, on="Date")

# Plot Trend
fig_trend = px.line(
    trend_df,
    x="Date",
    y=["Confirmed", "Deaths", "Recovered"],
    title=f"COVID-19 Trend in {country}"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ------------------------------
# Latest Numbers
# ------------------------------
st.subheader(f"📌 Latest Numbers in {country}")
latest = trend_df.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Confirmed", f"{latest['Confirmed']:,}")
col2.metric("Deaths", f"{latest['Deaths']:,}")
col3.metric("Recovered", f"{latest['Recovered']:,}")
