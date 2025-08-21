from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

def clustering(df):
    # Unsupervised Learning: DBSCAN e KMeans
    to_cluster = df.drop('G3', axis=1)

    k_means = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps = 0.7, min_samples=10)

    labels_dbscan = dbscan.fit_predict(to_cluster)
    labels_kmeans = k_means.fit_predict(to_cluster)

    # Analisi 2D
    plt.figure(figsize=(12,5))

    # KMeans
    plt.subplot(1,2,1)
    plt.scatter(to_cluster['PC1'], to_cluster['PC2'], c=labels_kmeans, cmap='viridis', s=50)
    plt.title("KMeans Clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # DBSCAN
    plt.subplot(1,2,2)
    plt.scatter(to_cluster['PC1'], to_cluster['PC2'], c=labels_dbscan, cmap='viridis', s=50)
    plt.title("DBSCAN Clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.show()

    return labels_kmeans, labels_dbscan

