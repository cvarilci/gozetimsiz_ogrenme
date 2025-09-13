# clustering_country_analysis.py
# Extended clustering pipeline for 29-country dataset
# Includes: PCA selection, Hierarchical (Agglomerative) clustering, HDBSCAN,
# internal validation (Silhouette, Davies-Bouldin), dendrogram, and world maps (Plotly choropleth)

# Requirements:
# pip install pandas numpy matplotlib seaborn scikit-learn plotly hdbscan scipy

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------
# 1) Load data (same CSV used earlier)
# --------------------------
DF_PATH = "29-country_data.csv"
raw = pd.read_csv(DF_PATH)
print("Loaded rows:", raw.shape[0])

# Keep a copy of country names for maps
countries = raw['country'].copy()

# Drop non-numeric column and scale (you used MinMax earlier; we'll reuse that)
X = raw.drop(columns=['country']).copy()
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --------------------------
# 2) PCA: choose number of components that explain >= 90% variance (or at least 2)
# --------------------------
pca = PCA()
pca_full = pca.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

n_components = int(np.searchsorted(cumvar, 0.90) + 1)  # first index where cumulative >= .90
n_components = max(2, n_components)
print(f"Chosen n_components = {n_components} (cumulative variance = {cumvar[n_components-1]:.3f})")

pca = PCA(n_components=n_components, random_state=42)
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=[f'PC{i+1}' for i in range(n_components)])

# Quick scree plot
plt.figure(figsize=(8,4))
plt.plot(range(1, len(cumvar)+1), cumvar, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA cumulative explained variance')
plt.grid(True)
plt.show()

# --------------------------
# 3) Hierarchical clustering (Agglomerative) exploration
#    - plot dendrogram for visual inspection
#    - evaluate clusterings for 2..6 clusters with different linkage methods
# --------------------------
# Dendrogram using Ward linkage on PCA reduced space
Z = linkage(X_pca, method='ward')
plt.figure(figsize=(12, 4))
dendrogram(Z, labels=countries.values, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.tight_layout()
plt.show()

linkages = ['ward', 'complete', 'average']
hier_results = {}
for link in linkages:
    hier_results[link] = {}
    for k in range(2,7):
        model = AgglomerativeClustering(n_clusters=k, linkage=link)
        labels = model.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        db = davies_bouldin_score(X_pca, labels)
        hier_results[link][k] = {'labels': labels, 'silhouette': sil, 'db': db}
        print(f"Linkage={link:8} k={k} silhouette={sil:.4f} db={db:.4f}")

# Pick the best hierarchical result by silhouette across linkages
best_link, best_k = None, None
best_sil = -1
for link in hier_results:
    for k in hier_results[link]:
        if hier_results[link][k]['silhouette'] > best_sil:
            best_sil = hier_results[link][k]['silhouette']
            best_link = link
            best_k = k
print(f"Best hierarchical: linkage={best_link}, k={best_k}, silhouette={best_sil:.4f}")

hier_labels = hier_results[best_link][best_k]['labels']

# --------------------------
# 4) HDBSCAN clustering
#    - try a small sweep of min_cluster_size and select best by silhouette on non-noise points
# --------------------------
hdb_results = {}
for mcs in [3, 4, 5, 6, 8]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric='euclidean')
    labels = clusterer.fit_predict(X_pca)
    # HDBSCAN returns -1 for noise; silhouette cannot handle single cluster or all -1
    mask = labels != -1
    if len(np.unique(labels[mask])) <= 1 or mask.sum() == 0:
        sil = -1
        db = np.inf
    else:
        sil = silhouette_score(X_pca[mask], labels[mask])
        db = davies_bouldin_score(X_pca[mask], labels[mask])
    hdb_results[mcs] = {'labels': labels, 'silhouette': sil, 'db': db, 'n_noise': (labels==-1).sum(), 'n_clusters': len(set(labels)) - ( -1 in labels )}
    print(f"HDBSCAN mcs={mcs} silhouette={sil:.4f} db={db:.4f} n_clusters={hdb_results[mcs]['n_clusters']} n_noise={hdb_results[mcs]['n_noise']}")

# choose best by silhouette
best_mcs = max(hdb_results.keys(), key=lambda k: hdb_results[k]['silhouette'])
print(f"Best HDBSCAN min_cluster_size = {best_mcs}")
hdb_labels = hdb_results[best_mcs]['labels']

# --------------------------
# 5) KMeans benchmark (you already ran KMeans on PCA earlier) - we compute again on PCA-reduced space
# --------------------------
km = KMeans(n_clusters=3, random_state=42)
km_labels = km.fit_predict(X_pca)
sil = silhouette_score(X_pca, km_labels)
db = davies_bouldin_score(X_pca, km_labels)
print(f"KMeans with k=3 silhouette={sil:.4f} db={db:.4f}")

# --------------------------
# 6) Summarize and attach cluster labels to a DataFrame for mapping
# --------------------------
out = pd.DataFrame({
    'country': countries.values,
    'hier_cluster': hier_labels,
    'kmeans_cluster': km_labels,
    'hdbscan_cluster': hdb_labels
})

# For nicer map categories, we'll map label integers to readable categories per algorithm separately.
# We'll create a helper to map integer cluster ids into strings like 'Cluster 0', 'Cluster 1', ...

def label_to_category(series, prefix='Cluster'):
    unique = sorted([x for x in np.unique(series) if x != -1])
    mapping = {lab: f"{prefix} {i}" for i, lab in enumerate(unique)}
    # keep noise as 'Noise' if present
    def mapper(x):
        if x == -1:
            return 'Noise'
        return mapping.get(x, f"{prefix} {x}")
    return series.map(mapper)

out['hier_cat'] = label_to_category(out['hier_cluster'], prefix='Hier')
out['kmeans_cat'] = label_to_category(out['kmeans_cluster'], prefix='KMeans')
out['hdbscan_cat'] = label_to_category(out['hdbscan_cluster'], prefix='HDBSCAN')

print(out.head())

# --------------------------
# 7) Plot choropleth maps for each algorithm using Plotly
# --------------------------
# Helper that builds and shows/saves a choropleth

def plot_choropleth(df_map, locations_col, cat_col, title, filename=None):
    fig = px.choropleth(df_map, locations=locations_col, locationmode='country names', color=cat_col, title=title)
    fig.update_geos(fitbounds='locations', visible=False)
    if filename:
        fig.write_html(filename)
        print(f"Saved map to {filename}")
    fig.show()

# Plot maps (these will open in your notebook / browser)
plot_choropleth(out, 'country', 'hier_cat', f'Hierarchical Clusters (link={best_link}, k={best_k})', filename='map_hierarchical.html')
plot_choropleth(out, 'country', 'kmeans_cat', f'KMeans (k=3)', filename='map_kmeans.html')
plot_choropleth(out, 'country', 'hdbscan_cat', f'HDBSCAN (min_cluster_size={best_mcs})', filename='map_hdbscan.html')

# --------------------------
# 8) Quick per-cluster summary statistics (on original features) to interpret clusters
# --------------------------
# Merge cluster labels back to original scaled and original data
full = raw.copy()
full = full.merge(out[['country','hier_cat','kmeans_cat','hdbscan_cat']], on='country')

for alg in ['hier_cat','kmeans_cat','hdbscan_cat']:
    print('\nSummary for', alg)
    
    # sadece sayısal kolonları al
    numeric_cols = full.select_dtypes(include=[np.number]).columns
    
    display = full.groupby(alg)[numeric_cols].mean().T
    print(display)

# --------------------------
# 9) Save results to CSV for later inspection
# --------------------------
out.to_csv('cluster_labels_by_algorithm.csv', index=False)
print('Wrote cluster_labels_by_algorithm.csv')

# End of script
print('Done.')
