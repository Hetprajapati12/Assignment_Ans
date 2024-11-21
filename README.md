# Clustering Geospatial Data

## Project Overview
This project demonstrates clustering geospatial data using multiple methods, including:
1. **K-Means**
2. **DBSCAN**
3. **Agglomerative Clustering**
4. **Gaussian Mixture Model (GMM)**

The dataset comprises latitude and longitude values representing geographic locations. The goal is to cluster these points into meaningful groups, evaluate the performance of each clustering method, and visualize the results using various types of plots.

---

## Approach
1. **Preprocessing**:
   - The dataset was preprocessed to split the `Longitude;Latitude` column into separate `Longitude` and `Latitude` columns.
   - Missing or invalid values were removed to ensure clean data for clustering.

2. **Clustering**:
   - **K-Means**:
     - Partitions data into \(k\) clusters by minimizing the within-cluster sum of squares.
     - Centroids were visualized for each cluster.
   - **DBSCAN**:
     - A density-based clustering algorithm capable of identifying noise points.
   - **Agglomerative Clustering**:
     - Hierarchical clustering method using Ward's linkage to minimize variance.
   - **Gaussian Mixture Model (GMM)**:
     - Fits a probabilistic model of Gaussian distributions to the data.

3. **Evaluation**:
   - The **Silhouette Score** was used to evaluate clustering performance.
     - Silhouette Score measures how well data points fit within their clusters compared to other clusters.
     - Scores range from -1 (poor clustering) to 1 (optimal clustering).

4. **Visualization**:
   - Scatter plots and Bar plots were created for each clustering method to visualize the cluster assignments.

5. **Best Method Selection**:
   - Based on the Silhouette Score, the best clustering method was selected and visualized with additional insights.

---

## Assumptions
1. The dataset is accurate, with no significant geospatial outliers.
2. Geospatial coordinates are evenly distributed, allowing clustering methods to perform meaningfully.
3. The number of clusters for K-Means, Agglomerative Clustering, and GMM is pre-defined as 4.

---

## Hurdles
1. **DBSCAN Noise Points**:
   - DBSCAN frequently labeled points as noise due to varying density, reducing cluster count.
   - The visualization excluded these noise points for clarity.
   
2. **Cluster Size Variance**:
   - Unevenly distributed data caused clustering methods like K-Means to assign incorrect clusters for sparse areas.
   
3. **Plot Overlaps**:
   - Scatter plot labels and KDE plots were overlapping due to dense clusters. Resolved by adjusting figure size and rotation.

4. **Parameter Sensitivity**:
   - DBSCAN’s performance heavily depended on tuning the `eps` and `min_samples` parameters.

---

## Solution
1. **Parameter Tuning**:
   - DBSCAN parameters were tuned manually to improve clustering quality.
2. **Dynamic Visualization**:
   - Created scatter plots, KDE heatmaps, and bar plots for an intuitive understanding of clusters.
3. **Error Handling**:
   - Ensured valid clusters were visualized and excluded noise points for density-based clustering.
4. **Conditional Visualization**:
   - Centroid visualization was added for K-Means clustering.

---

## Results
- **Silhouette Scores**:
  - **K-Means**: 0.52
  - **DBSCAN**: 0.57
  - **Agglomerative Clustering**: 0.51
  - **GMM**: 0.49

- **Best Method**: 
  - **DBSCAN** achieved the highest Silhouette Score (0.57) and was selected as the best clustering method.

---

## How to Run
1. Clone this repository and navigate to the project folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
