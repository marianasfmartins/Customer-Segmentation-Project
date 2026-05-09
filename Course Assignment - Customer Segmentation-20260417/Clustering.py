import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage



# --- 1. Find optimal k for K-Means ---
def find_optimal_k(df, k_range=range(2, 11)):

    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df, labels))
        print(f"k={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette: {silhouette_score(df, labels):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, marker='o')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(k_range, silhouette_scores, marker='o', color='orange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    return inertias, silhouette_scores


# --- 2. Fit final K-Means model ---
def fit_kmeans(df, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)

    print(f"Silhouette Score: {silhouette_score(df, labels):.4f}")
    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    return kmeans, labels


# --- 3. Hierarchical Clustering ---
def fit_hierarchical(df, n_clusters, method='ward'):

    # Dendrogram (on a sample for performance)
    sample = df[:2000] if len(df) > 2000 else df
    linked = linkage(sample, method=method)

    plt.figure(figsize=(14, 6))
    dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    # Fit model
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = hierarchical.fit_predict(df)

    print(f"Silhouette Score: {silhouette_score(df, labels):.4f}")
    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    return hierarchical, labels


# --- 4. DBSCAN ---
def find_optimal_eps(df, n_neighbors=5):

    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(df)
    distances, _ = neighbors.kneighbors(df)
    distances = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.title('K-Distance Graph (find elbow for eps)')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{n_neighbors}-NN Distance')
    plt.tight_layout()
    plt.show()


def fit_dbscan(df, eps, min_samples=5):

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise} ({(n_noise / len(labels) * 100):.2f}%)")

    if n_clusters > 1:
        mask = labels != -1
        print(f"Silhouette Score (excluding noise): {silhouette_score(df[mask], labels[mask]):.4f}")

    print(f"Cluster sizes:\n{pd.Series(labels).value_counts().sort_index()}")

    return dbscan, labels


# --- 5. Compare all models ---
def compare_models(df, labels_dict):

    print("\n--- Model Comparison (Silhouette Scores) ---")
    results = {}

    for model_name, labels in labels_dict.items():
        mask = labels != -1
        score = silhouette_score(df[mask], labels[mask])
        results[model_name] = score
        print(f"{model_name}: {score:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nBest model: {best_model} with silhouette score {results[best_model]:.4f}")

    return results



