import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🧠 Customer Segmentation", layout="wide")
st.title("🧠 Customer Behavior Dashboard using Unsupervised Learning")

st.markdown("""
This interactive dashboard allows you to explore customer behavior using **unsupervised learning**.
Upload your customer dataset, apply **K-Means clustering**, and analyze the segments.
""")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("📂 Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    if st.checkbox("Show summary statistics"):
        st.write(df.describe())

    # Step 2: Select features for clustering
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("🧬 Select features for clustering", numeric_columns, default=numeric_columns)

    if selected_features:
        X = df[selected_features]
        X_scaled = StandardScaler().fit_transform(X)

        # Step 3: Choose number of clusters
        k = st.slider("🔢 Select number of clusters (K)", 2, 10, 3)

        # Step 4: Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        st.subheader("📌 Clustered Data")
        st.dataframe(df.head())

        # Step 5: Visualization
        st.subheader("📈 Cluster Visualization")

        if len(selected_features) >= 2:
            x_axis = st.selectbox("X-axis", selected_features, index=0)
            y_axis = st.selectbox("Y-axis", selected_features, index=1)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette='tab10')
            st.pyplot(plt.gcf())
        else:
            st.warning("Please select at least two features for visualization.")
else:
    st.info("Please upload a dataset to begin.")

