import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

def plot_heatmap(df):
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(fig)

def plot_clusters(df, labels):
    df['Cluster'] = labels
    fig = px.scatter(df, x="Annual_Income", y="Spending_Score", color=df['Cluster'].astype(str),
                     title="Customer Segmentation", labels={"color": "Cluster"})
    st.plotly_chart(fig, use_container_width=True)
