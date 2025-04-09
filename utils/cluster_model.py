from sklearn.cluster import KMeans

def run_kmeans(df, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(df[["Annual_Income", "Spending_Score"]])
    return model, labels
