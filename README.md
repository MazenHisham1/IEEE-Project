# COVID-19 Interactive Dashboard

A Streamlit-based interactive dashboard for COVID-19 time-series data, offering visual insights, forecasting capabilities, and easy deployment.

---

##  Live Demo

Try the live version here: [Dashboard Live Demo](https://covid-19-deploy.streamlit.app/) or [Dashboard Live Demo](https://covid-dashboard-production-12a2.up.railway.app/) 

---

##  Features

- 📊 **Global KPIs**: Quickly view total confirmed cases, deaths, and recoveries globally.
- 🌍 **Choropleth Map**: Interactive world map showing the latest confirmed cases by country.
-  **Country Trends**: Select a country and date range to visualize trends in confirmed cases, deaths, and recoveries—with optional moving averages.
-  **Downloadable CSV**: Export filtered time-series data for your selected country and date range.
-  **Parquet Support**: Automatically converts CSV datasets to Parquet format for faster loading and reduced file size.
-  **Plotly-Resampler Integration** (Experimental): Optionally use `plotly-resampler` for efficient visualization of large time-series data, with a reliable server-side fallback for downsampling when necessary.
-  **Full Deployment Ready**: Includes `Dockerfile`, `Procfile`, and support for deployment on platforms like Streamlit Community Cloud, Vercel, or Heroku.

---

##  Installation & Running Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/MazenHisham1/IEEE-Project.git
   cd IEEE-Project
