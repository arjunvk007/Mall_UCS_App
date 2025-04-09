import streamlit as st
from utils.load_data import load_data
from utils.visuals import plot_heatmap

def show():
    df = load_data()
    st.subheader("🔍 Correlation Heatmap")
    plot_heatmap(df)
