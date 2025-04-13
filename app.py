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
A predefined customer dataset is loaded. You can apply **K-Means clustering**, analyze segments, and predict the cluster for new inputs.
""")

# Load data from file
df = pd.read_csv("data.csv")  # Make sure 'data.csv' is in the same folder or provide full path

    # Step 5: Input for prediction
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
    st.warning("Please select features to continue.")

    # Step 4: Cluster Visualization
    st.subheader("📈 Cluster Visualization")
    if len(selected_features) >= 2:
        x_axis = st.selectbox("X-axis", selected_features, index=0)
        y_axis = st.selectbox("Y-axis", selected_features, index=1)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Cluster', palette='tab10')
        st.pyplot(plt.gcf())
    else:
        st.warning("Please select at least two features for visualization.")
