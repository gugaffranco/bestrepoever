import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from preprocessing_final import load_data


def remove_outliers(X_scaled, contamination=0.01, random_state=42):
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_mask = iso.fit_predict(X_scaled) == 1
    X_scaled_clean = X_scaled[outlier_mask]
    print(f"\nRemoved {(~outlier_mask).sum()} outliers, keeping {outlier_mask.sum()} customers ")
    return X_scaled_clean, outlier_mask

def fit_kmeans(X_scaled_clean, n_clusters=5, random_state=42, n_init=10):
    print('Starting KMeans')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    kmeans_labels = kmeans.fit_predict(X_scaled_clean)
    score = silhouette_score(X_scaled_clean, kmeans_labels)
    print(f"\nK-Means silhouette={score:.3f} clusters={n_clusters}\n")
    return kmeans, kmeans_labels

def assign_all_labels(X_scaled, outlier_mask, kmeans, kmeans_labels):
    X_scaled_outliers = X_scaled[~outlier_mask]
    outlier_labels = kmeans.predict(X_scaled_outliers)
    all_labels = np.empty(len(X_scaled), dtype=int)
    all_labels[outlier_mask] = kmeans_labels
    all_labels[~outlier_mask] = outlier_labels
    print('All labels assigned')
    return all_labels

def save_cluster_assignments(X_scaled, all_labels, path='cluster_assignments.csv'):
    result = pd.DataFrame({'customer_id': X_scaled.index, 'cluster': all_labels})
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} customers to {path}\n")

if __name__ == '__main__':
    X_scaled=load_data('X_scaled.csv')
    X_scaled_clean, outlier_mask=remove_outliers(X_scaled)
    kmeans, kmeans_labels= fit_kmeans(X_scaled_clean)
    all_labels=assign_all_labels(X_scaled, outlier_mask, kmeans, kmeans_labels)
    save_cluster_assignments(X_scaled, all_labels)