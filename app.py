import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit page config
st.set_page_config(page_title="🧠 Customer Segmentation", layout="wide")
st.title("🧠 Customer Behavior Dashboard using Unsupervised Learning")

st.markdown("""
This interactive dashboard allows you to explore customer behavior using **unsupervised learning**.
Upload your customer dataset, apply **K-Means clustering**, analyze the segments, and predict cluster for new input.
""")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("📂 Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    if st.checkbox("Show summary statistics"):
        st.write(df.describe())


        # Step 6: Input for prediction
        st.subheader("🧮 Predict Cluster for New Customer Input")
        input_data = []
        for feature in selected_features:
            value = st.number_input(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()))
            input_data.append(value)

        if st.button("Predict Cluster"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            predicted_cluster = kmeans.predict(input_scaled)[0]
            st.success(f"🌟 The predicted cluster for the input is: **Cluster {predicted_cluster}**")

            # Optional: Show where the input lands on the cluster plot
            if len(selected_features) >= 2:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette='tab10')
                plt.scatter(
                    input_data[selected_features.index(x_axis)],
                    input_data[selected_features.index(y_axis)],
                    color='black', label='New Input', s=100, marker='X'
                )
                plt.legend()
                st.pyplot(plt.gcf())
else:
    st.info("Please upload a dataset to begin.")


    # Step 2: Select features for clustering
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("🧬 Select features for clustering", numeric_columns, default=numeric_columns)

    if selected_features:
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 3: Choose number of clusters
        k = st.slider("🔢 Select number of clusters (K)", 2, 10, 3)

        # Step 4: Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters

        st.subheader("📌 Clustered Data")
        st.dataframe(df.head())
