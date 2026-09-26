# FRAUD_ANALYZER

Hacker-themed Streamlit dashboard for exploring credit card fraud data and finding the features that predict fraud.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frauddashboard-rfkpx5rgjbznw3vzvomxq8.streamlit.app/)

**Live dashboard:** https://frauddashboard-rfkpx5rgjbznw3vzvomxq8.streamlit.app/

## Screenshots

### Overview
![Overview](screenshots/01_overview.png)

### Feature Explorer
![Feature Explorer](screenshots/02_feature_explorer.png)

### Predictors
![Predictors](screenshots/03_predictors.png)

### Segments
![Segments](screenshots/04_segments.png)

### Data Table
![Data Table](screenshots/05_data_table.png)

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tabs: Overview · Feature Explorer · Predictors (correlation / mutual info / RandomForest) · Segments (fraud-rate heatmaps, risky segments) · Data table with sort & CSV export.
