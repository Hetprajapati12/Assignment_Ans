
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class Clustering:
    """
    A class to perform clustering on geospatial data using different methods.
    """

    def __init__(self, file_path):
        """
        Initialize the Clustering class with a dataset path.
        Args:
            file_path (str): Path to the CSV file containing geospatial data.
        """
        self.file_path = 'C:/Users/hetpr/OneDrive/Desktop/Assignment_Ans/ML Assignment Dataset.csv'
        self.data = None

    def load_and_preprocess(self):
        """
        Load and preprocess the geospatial dataset.
        Returns:
            pd.DataFrame: Preprocessed DataFrame with longitude and latitude columns.
        """
        self.data = pd.read_csv(self.file_path)
        self.data[['Longitude', 'Latitude']] = self.data['Longitude;Latitude'].str.split(';', expand=True)
        self.data['Longitude'] = pd.to_numeric(self.data['Longitude'], errors='coerce')
        self.data['Latitude'] = pd.to_numeric(self.data['Latitude'], errors='coerce')
        self.data.dropna(inplace=True)

    def kmeans_clustering(self, n_clusters=4):
        """
        Apply K-Means clustering and calculate Silhouette Score.
        Args:
            n_clusters (int): Number of clusters.
        Returns:
            float: Silhouette Score for K-Means clustering.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['KMeans_Cluster'] = kmeans.fit_predict(self.data[['Longitude', 'Latitude']])
        return silhouette_score(self.data[['Longitude', 'Latitude']], self.data['KMeans_Cluster'])

    def dbscan_clustering(self, eps=0.5, min_samples=10):
        """
        Apply DBSCAN clustering and calculate Silhouette Score (excluding noise).
        Args:
            eps (float): Maximum distance between samples to be considered in the same cluster.
            min_samples (int): Minimum samples in a neighborhood for a point to be considered a core point.
        Returns:
            float: Silhouette Score for DBSCAN clustering or -1 if not applicable.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.data['DBSCAN_Cluster'] = dbscan.fit_predict(self.data[['Longitude', 'Latitude']])
        if len(set(self.data['DBSCAN_Cluster'])) > 1:
            return silhouette_score(
                self.data[['Longitude', 'Latitude']][self.data['DBSCAN_Cluster'] != -1],
                self.data['DBSCAN_Cluster'][self.data['DBSCAN_Cluster'] != -1]
            )
        return -1

    def agglomerative_clustering(self, n_clusters=4):
        """
        Apply Agglomerative Clustering and calculate Silhouette Score.
        Args:
            n_clusters (int): Number of clusters.
        Returns:
            float: Silhouette Score for Agglomerative clustering.
        """
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.data['Agglomerative_Cluster'] = agglo.fit_predict(self.data[['Longitude', 'Latitude']])
        return silhouette_score(self.data[['Longitude', 'Latitude']], self.data['Agglomerative_Cluster'])

    def gmm_clustering(self, n_components=4):
        """
        Apply Gaussian Mixture Model (GMM) clustering and calculate Silhouette Score.
        Args:
            n_components (int): Number of components (clusters).
        Returns:
            float: Silhouette Score for GMM clustering.
        """
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.data['GMM_Cluster'] = gmm.fit_predict(self.data[['Longitude', 'Latitude']])
        return silhouette_score(self.data[['Longitude', 'Latitude']], self.data['GMM_Cluster'])

    def visualize_clusters(self, cluster_column, title):
        """
        Visualize clusters on a scatter plot.
        Args:
            cluster_column (str): Column containing cluster labels.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data['Longitude'], self.data['Latitude'], c=self.data[cluster_column], cmap='tab10', s=10)
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='Cluster')
        plt.show()
    
    def visualize_cluster_counts(self, cluster_column, title):
        """
        Visualize the number of points in each cluster using a bar plot.
        Args:
            cluster_column (str): Column containing cluster labels.
            title (str): Title for the bar plot.
        """
        cluster_counts = self.data[cluster_column].value_counts()
        plt.figure(figsize=(12, 6))
        cluster_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Points')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to perform clustering and choose the best method.
    """
    file_path = 'ML_Assignment_Dataset.csv'
    clustering = Clustering(file_path)
    clustering.load_and_preprocess()

    # Apply and evaluate clustering methods
    kmeans_score = clustering.kmeans_clustering(n_clusters=4)
    dbscan_score = clustering.dbscan_clustering(eps=0.5, min_samples=10)
    agglo_score = clustering.agglomerative_clustering(n_clusters=4)
    gmm_score = clustering.gmm_clustering(n_components=4)

    # Print scores for comparison
    print(f"K-Means Silhouette Score: {kmeans_score:.2f}")
    print(f"DBSCAN Silhouette Score: {dbscan_score:.2f}")
    print(f"Agglomerative Clustering Silhouette Score: {agglo_score:.2f}")
    print(f"GMM Silhouette Score: {gmm_score:.2f}")

    # Choose the best clustering method
    scores = {
        'K-Means': kmeans_score,
        'DBSCAN': dbscan_score,
        'Agglomerative': agglo_score,
        'GMM': gmm_score
    }
    best_method = max(scores, key=scores.get)
    print(f"Best clustering method: {best_method}")

    # Visualize the best clustering result
    clustering.visualize_clusters(cluster_column=f"{best_method.replace(' ', '_')}_Cluster", title=f"Best Clustering Method: {best_method}")
    clustering.visualize_cluster_counts(cluster_column=f"{best_method.replace(' ', '_')}_Cluster",title=f"Barplot for Best Clustering Method: {best_method}")
if __name__ == '__main__':
    main()
