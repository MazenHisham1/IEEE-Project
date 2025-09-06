import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# -----------------------------------------------------------
# Dash App
# -----------------------------------------------------------
app = Dash(__name__)
app.title = "COVID-19 & World Data Dashboard"

# -----------------------------------------------------------
# Layout
# -----------------------------------------------------------
app.layout = html.Div([
    html.H1("🌍 COVID-19 & World Data Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs([
        # ----------------- Global Overview -------------------
        dcc.Tab(label="Global Overview", children=[
            html.Div([
                html.H2("Global KPIs"),
                dcc.Graph(
                    id="global_kpis",
                    figure=px.bar(
                        x=["Confirmed", "Deaths", "Recovered"],
                        y=[
                            covid_daily["Confirmed"].sum(),
                            covid_daily["Deaths"].sum(),
                            covid_daily["Recovered"].sum()
                        ],
                        labels={"x": "Metric", "y": "Total"},
                        title="Overall Global Totals"
                    )
                ),

                dcc.Graph(
                    id="world_map",
                    figure=px.choropleth(
                        covid_daily.groupby("Country/Region")[["Confirmed"]].sum().reset_index(),
                        locations="Country/Region",
                        locationmode="country names",
                        color="Confirmed",
                        title="Total Confirmed Cases by Country",
                        color_continuous_scale="Reds"
                    )
                ),

                dcc.Graph(
                    id="global_trend",
                    figure=px.line(
                        confirmed.drop(columns=["Lat","Long","Country/Region"])
                        .sum()
                        .reset_index(name="Confirmed"),
                        x="index", y="Confirmed",
                        title="Global Confirmed Cases Over Time"
                    )
                )
            ])
        ]),

        # ----------------- Country Insights -------------------
        dcc.Tab(label="Country Insights", children=[
            html.Div([
                html.H2("Country Drilldown"),
                dcc.Dropdown(
                    id="country_dropdown",
                    options=[{"label": c, "value": c} for c in covid_daily["Country/Region"].unique()],
                    value="US"
                ),
                dcc.Graph(id="country_trend"),
            ])
        ]),

        # ----------------- Socio-Economic -------------------
        dcc.Tab(label="Socio-Economic vs COVID", children=[
            html.Div([
                html.H2("Socio-Economic Impact"),
                dcc.Graph(
                    id="gdp_scatter",
                    figure=px.scatter(
                        Countries.merge(
                            covid_daily.groupby("Country/Region")[["Confirmed","Deaths"]].sum().reset_index(),
                            left_on="Country", right_on="Country/Region", how="inner"
                        ),
                        x="GDP ($ per capita)", y="Confirmed", size="Population",
                        color="Region",
                        hover_name="Country",
                        title="GDP vs Confirmed Cases"
                    )
                )
            ])
        ]),

        # ----------------- SARS vs COVID -------------------
        dcc.Tab(label="SARS 2003 vs COVID", children=[
            html.Div([
                html.H2("SARS 2003 vs COVID-19"),
                dcc.Graph(
                    id="sars_vs_covid",
                    figure=go.Figure()
                    .add_trace(go.Scatter(
                        x=Sars["Date"], y=Sars["Cumulative number of case(s)"],
                        mode="lines", name="SARS Cases"
                    ))
                    .add_trace(go.Scatter(
                        x=confirmed.drop(columns=["Lat","Long","Country/Region"]).sum().index,
                        y=confirmed.drop(columns=["Lat","Long","Country/Region"]).sum().values,
                        mode="lines", name="COVID Cases"
                    ))
                    .update_layout(title="SARS vs COVID Global Cases")
                )
            ])
        ]),

        # ----------------- Case Line List -------------------
        dcc.Tab(label="Case Line Lists", children=[
            html.Div([
                html.H2("Case Demographics"),
                dcc.Graph(
                    id="age_distribution",
                    figure=px.histogram(
                        line_list, x="age", nbins=30,
                        title="Age Distribution of Cases"
                    )
                ),
                dcc.Graph(
                    id="gender_distribution",
                    figure=px.histogram(
                        line_list, x="gender", color="gender",
                        title="Gender Distribution"
                    )
                )
            ])
        ]),
    ])
])

# -----------------------------------------------------------
# Callbacks
# -----------------------------------------------------------
@app.callback(
    Output("country_trend", "figure"),
    Input("country_dropdown", "value")
)
def update_country_trend(country):
    confirmed_c = confirmed[confirmed["Country/Region"] == country].drop(columns=["Lat","Long","Country/Region"]).sum()
    deaths_c = deaths[deaths["Country/Region"] == country].drop(columns=["Lat","Long","Country/Region"]).sum()
    recovered_c = recovered[recovered["Country/Region"] == country].drop(columns=["Lat","Long","Country/Region"]).sum()

    df = pd.DataFrame({
        "Date": confirmed_c.index,
        "Confirmed": confirmed_c.values,
        "Deaths": deaths_c.values,
        "Recovered": recovered_c.values
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Confirmed"], mode="lines", name="Confirmed"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Deaths"], mode="lines", name="Deaths"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Recovered"], mode="lines", name="Recovered"))
    fig.update_layout(title=f"COVID-19 Trend in {country}")

    return fig

# -----------------------------------------------------------
# Run (Notebook mode: switch to JupyterDash if you want inline)
# -----------------------------------------------------------
app.run(debug=True, use_reloader=False)
