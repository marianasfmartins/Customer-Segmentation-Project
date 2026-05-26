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

    # Create cluster profile with cluster numbers as columns and variables as rows
    numeric_df = df.select_dtypes(include=[np.number])
    cluster_profile = numeric_df.groupby(labels).mean().T
    cluster_profile.columns = [f"cluster_{int(c)}" for c in cluster_profile.columns]

    return cluster_profile


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



# --- 5. Self-Organizing Map (SOM) ---
def _initialize_som_weights(data, map_shape, random_state):
    rng = np.random.default_rng(random_state)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return rng.uniform(min_vals, max_vals, size=(map_shape[0], map_shape[1], data.shape[1]))


def _find_bmu(weights, sample):
    diff = weights - sample
    dist_sq = np.sum(diff**2, axis=2)
    return np.unravel_index(np.argmin(dist_sq), dist_sq.shape)


def _som_learning_rate(initial_lr, iteration, n_iterations):
    return initial_lr * np.exp(-iteration / n_iterations)


def _som_radius(initial_sigma, iteration, time_constant):
    return initial_sigma * np.exp(-iteration / time_constant)


def plot_som_u_matrix(weights):
    """Plot the SOM U-Matrix showing average neighbor distances."""
    m, n, dim = weights.shape
    u_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i - 1, j])
            if i < m - 1:
                neighbors.append(weights[i + 1, j])
            if j > 0:
                neighbors.append(weights[i, j - 1])
            if j < n - 1:
                neighbors.append(weights[i, j + 1])
            if neighbors:
                distances = np.linalg.norm(weights[i, j] - np.vstack(neighbors), axis=1)
                u_matrix[i, j] = distances.mean()

    plt.figure(figsize=(8, 6))
    plt.imshow(u_matrix, cmap='viridis', origin='lower')
    plt.colorbar(label='Average distance to neighbors')
    plt.title('SOM U-Matrix')
    plt.xlabel('Map x')
    plt.ylabel('Map y')
    plt.tight_layout()
    plt.show()

    return u_matrix


def fit_som(df, map_shape=(10, 10), n_iterations=1000, learning_rate=0.5, sigma=None, random_state=42, plot_u_matrix=False):
    """Train a Self-Organizing Map and return BMU assignments for each sample."""
    data = df.values.astype(float)

    if sigma is None:
        sigma = max(map_shape) / 2.0

    weights = _initialize_som_weights(data, map_shape, random_state)
    time_constant = n_iterations / np.log(sigma) if sigma > 1 else n_iterations

    grid_x = np.arange(map_shape[0])[:, np.newaxis]
    grid_y = np.arange(map_shape[1])[np.newaxis, :]

    for iteration in range(n_iterations):
        lr = _som_learning_rate(learning_rate, iteration, n_iterations)
        radius = _som_radius(sigma, iteration, time_constant)
        radius_sq = radius ** 2 if radius > 0 else 1.0

        for sample in data:
            bmu_i, bmu_j = _find_bmu(weights, sample)
            distance_sq = (grid_x - bmu_i) ** 2 + (grid_y - bmu_j) ** 2
            neighborhood = np.exp(-distance_sq / (2 * radius_sq))
            influence = neighborhood[..., np.newaxis]
            weights += lr * influence * (sample - weights)

    bmu_indices = np.array([_find_bmu(weights, x) for x in data])
    labels = bmu_indices[:, 0] * map_shape[1] + bmu_indices[:, 1]

    if plot_u_matrix:
        plot_som_u_matrix(weights)

    print(f"Trained SOM with map shape {map_shape} and {n_iterations} iterations.")
    print(f"Unique BMU clusters: {len(np.unique(labels))}")

    return {
        'weights': weights,
        'labels': labels,
        'bmu_indices': bmu_indices,
        'map_shape': map_shape,
    }


def get_som_cluster_labels(weights, n_clusters, random_state=42):
    """Cluster SOM neurons into named groups and return the neuron cluster map."""
    neurons = weights.reshape(-1, weights.shape[2])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    neuron_labels = kmeans.fit_predict(neurons)
    return neuron_labels.reshape(weights.shape[0], weights.shape[1])


def assign_som_clusters(weights, df, n_clusters, random_state=42):
    """Assign each data sample to a SOM neuron cluster label."""
    neuron_labels = get_som_cluster_labels(weights, n_clusters, random_state)
    data = df.values.astype(float)
    bmu_indices = np.array([_find_bmu(weights, x) for x in data])
    labels = np.array([neuron_labels[i, j] for i, j in bmu_indices])
    return labels, neuron_labels


# --- 6. Compare all models ---
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



