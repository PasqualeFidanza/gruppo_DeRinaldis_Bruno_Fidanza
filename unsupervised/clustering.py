from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

def clustering(df):
    # Unsupervised Learning: DBSCAN e KMeans
    to_cluster = df.drop('G3', axis=1)

    k_means = KMeans(n_clusters=2, random_state=42)
    dbscan = DBSCAN(eps=0.7, min_samples=4)

    labels_dbscan = dbscan.fit_predict(to_cluster)
    labels_kmeans = k_means.fit_predict(to_cluster)

    fig = plt.figure(figsize=(14,6))

    # KMeans
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(to_cluster['PC1'], to_cluster['PC2'], to_cluster['PC3'], 
                c=labels_kmeans, cmap='viridis', s=50)
    
    # Plot dei centroidi
    centroids = k_means.cluster_centers_
    ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                c='red', marker='*', s=300, label='Centroidi')
    
    ax1.set_title("KMeans Clustering")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()

    # DBSCAN
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(to_cluster['PC1'], to_cluster['PC2'], to_cluster['PC3'], 
                c=labels_dbscan, cmap='viridis', s=50)
    ax2.set_title("DBSCAN Clustering")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")

    plt.show()

    return labels_kmeans, labels_dbscan
