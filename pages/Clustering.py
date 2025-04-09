import streamlit as st
from utils.load_data import load_data
from utils.cluster_model import run_kmeans
from utils.visuals import plot_clusters

def show():
    df = load_data()
    st.subheader("🤖 KMeans Clustering")

    clusters = st.slider("Select number of clusters", 2, 10, 5)
    model, labels = run_kmeans(df, clusters)

    st.markdown("### 📌 Cluster Centers")
    st.write(model.cluster_centers_)

    plot_clusters(df, labels)
