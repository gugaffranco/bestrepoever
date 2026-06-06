import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def remove_outliers(X_scaled, contamination=0.01, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_mask = iso.fit_predict(X_scaled) == 1
    X_scaled_clean = X_scaled[outlier_mask]
    print(f"Removed {(~outlier_mask).sum()} outliers, keeping {outlier_mask.sum()} customers")
    return X_scaled_clean, outlier_mask

def pca_data(X_scaled_clean, n_components=13, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled_clean)
    X_pca = pd.DataFrame(X_pca, index=X_scaled_clean.index)
    return X_pca, pca

def fit_kmeans(X_pca, n_clusters=5, random_state=42, n_init=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans_labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, kmeans_labels)
    print(f"K-Means silhouette={score:.3f} clusters={n_clusters}")
    return kmeans, kmeans_labels

def assign_all_labels(X_scaled, outlier_mask, kmeans, pca, kmeans_labels):
    X_pca_outliers = pca.transform(X_scaled[~outlier_mask])
    outlier_labels = kmeans.predict(X_pca_outliers)
    all_labels = np.empty(len(X_scaled), dtype=int)
    all_labels[outlier_mask] = kmeans_labels
    all_labels[~outlier_mask] = outlier_labels
    return all_labels

def save_cluster_assignments(X_scaled, all_labels, path='cluster_assignments.csv'):
    result = pd.DataFrame({'customer_id': X_scaled.index, 'cluster': all_labels})
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} customers to {path}")