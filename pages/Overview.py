import streamlit as st
from utils.load_data import load_data

def show():
    st.title("📋 Dataset Overview")

    df = load_data()
    
    st.subheader("🔍 First 5 Records")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    st.subheader("🧮 Dataset Info")
    st.write(f"Number of Rows: {df.shape[0]}")
    st.write(f"Number of Columns: {df.shape[1]}")
