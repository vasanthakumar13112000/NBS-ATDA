
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("UK Crime Data Dashboard")

# Load cleaned data
df = pd.read_csv("cleaned_crime_data.csv")

# User filter
crime_filter = st.selectbox("Select Crime Type", df['crime_type'].dropna().unique())
filtered = df[df['crime_type'] == crime_filter]

# Map of crimes
st.subheader("Crime Locations")
st.map(filtered[['latitude', 'longitude']].dropna())

# Monthly trend
df['month'] = pd.to_datetime(df['month'], errors='coerce')
monthly = df[df['crime_type'] == crime_filter].groupby(df['month'].dt.to_period("M")).size()
st.line_chart(monthly)
