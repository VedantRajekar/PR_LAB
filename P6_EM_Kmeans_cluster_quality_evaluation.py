# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# 2. Load Dataset and Preprocess
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

y_true = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# 3. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# 4. EM Clustering (Gaussian Mixture)
gmm = GaussianMixture(n_components=3, random_state=42)
y_gmm = gmm.fit_predict(X_scaled)


# 5. Evaluation Metrics

# K-Means Evaluation
print("K-Means Silhouette Score:", silhouette_score(X_scaled, y_kmeans))
print("K-Means Adjusted Rand Index:", adjusted_rand_score(y_true, y_kmeans))
print("K-Means Davies-Bouldin Score:", davies_bouldin_score(X_scaled, y_kmeans))

# GMM Evaluation
print("GMM Silhouette Score:", silhouette_score(X_scaled, y_gmm))
print("GMM Adjusted Rand Index:", adjusted_rand_score(y_true, y_gmm))
print("GMM Davies-Bouldin Score:", davies_bouldin_score(X_scaled, y_gmm))