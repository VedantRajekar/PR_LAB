# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# 2. Load Dataset and Preprocess
wine_data = load_wine()

df = pd.DataFrame(
    data=wine_data.data,
    columns=wine_data.feature_names
)

ground_truth = wine_data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# 3. Gaussian Mixture Model (EM Algorithm)
gmm = GaussianMixture(n_components=3, random_state=42)

gmm_labels = gmm.fit_predict(X_scaled)

gmm_silhouette = silhouette_score(X_scaled, gmm_labels)
gmm_ari = adjusted_rand_score(ground_truth, gmm_labels)

print("GMM Silhouette Score:", gmm_silhouette)
print("GMM Adjusted Rand Index:", gmm_ari)


# 4. K-Means Clustering
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_ari = adjusted_rand_score(ground_truth, kmeans_labels)

print("K-Means Silhouette Score:", kmeans_silhouette)
print("K-Means Adjusted Rand Index:", kmeans_ari)