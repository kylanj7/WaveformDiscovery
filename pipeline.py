import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import time
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

# Check for GPU availability
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    HAS_GPU = True
    print("GPU acceleration enabled with RAPIDS libraries")
except ImportError:
    HAS_GPU = False
    from sklearn.decomposition import IncrementalPCA
    from sklearn.manifold import TSNE
    print("GPU libraries not available, using CPU processing")

print("Starting large-scale waveform analysis with GPU acceleration...")
start_time = time.time()

#  Waveform function
def generate_waveform(params):
    f_sine, f_square, f_triangle, f_sawtooth = params
    
    # Use numpy arrays for CPU processing
    X = np.linspace(0, 1, 1000)
    Y = np.linspace(0, 1, 1000)
    
    x1 = np.sin(2 * np.pi * f_sine * X)
    y1 = np.sign(np.sin(2 * np.pi * f_square * Y))
    x2 = 2/np.pi * np.arcsin(np.sin(2 * np.pi * f_triangle * X))
    y2 = 2 * (f_sawtooth * Y - np.floor(0.5 + f_sawtooth * Y))
    
    return np.sin(y1 * y2) * np.cos(x1 * x2) * (y1 - y2)

# GPU-optimized batch feature extraction
def extract_features_batch(waveforms):
    """Extract features for a batch of waveforms, optimized for GPU if available"""
    features_list = []
    
    if HAS_GPU:
        # Move to GPU
        waveforms_gpu = cp.array(waveforms, dtype=cp.float32)
        
        # Basic statistics (vectorized across batch)
        means = cp.mean(waveforms_gpu, axis=1)
        stds = cp.std(waveforms_gpu, axis=1)
        maxs = cp.max(waveforms_gpu, axis=1)
        mins = cp.min(waveforms_gpu, axis=1)
        abs_means = cp.mean(cp.abs(waveforms_gpu), axis=1)
        ranges = maxs - mins
        energies = cp.sum(waveforms_gpu**2, axis=1)
        
        # Move results back to CPU
        means = cp.asnumpy(means)
        stds = cp.asnumpy(stds)
        ranges = cp.asnumpy(ranges)
        abs_means = cp.asnumpy(abs_means)
        energies = cp.asnumpy(energies)
        
        # Process each waveform for more complex features
        for i, wf in enumerate(waveforms):
            features = {}
            features['mean'] = means[i]
            features['std'] = stds[i]
            features['range'] = ranges[i]
            features['abs_mean'] = abs_means[i]
            features['energy'] = energies[i]
            
            # These operations are more efficiently done on CPU
            features['zero_crossings'] = np.sum(np.diff(np.signbit(wf)))
            features['peak_count'] = len(signal.find_peaks(wf[:500])[0])
            
            # Statistical moments
            wf_mean = means[i]
            wf_var = stds[i]**2
            wf_centered = wf - wf_mean
            features['skewness'] = np.mean(wf_centered**3) / (wf_var**(3/2)) if wf_var > 0 else 0
            features['kurtosis'] = np.mean(wf_centered**4) / (wf_var**2) if wf_var > 0 else 0
            
            # Complexity (total variation)
            features['complexity'] = np.sum(np.abs(np.diff(wf)))
            
            # Frequency domain
            fft_vals = np.abs(fft(wf))[:len(wf)//2]
            fft_max = np.max(fft_vals)
            fft_vals = fft_vals / fft_max if fft_max > 0 else fft_vals
            
            # Frequency bands
            num_bands = 10
            band_size = len(fft_vals) // num_bands
            for j in range(num_bands):
                start = j * band_size
                end = (j + 1) * band_size if j < num_bands - 1 else len(fft_vals)
                features[f'band_{j}'] = np.sum(fft_vals[start:end])
            
            # Spectral features
            freqs = np.fft.fftfreq(len(wf))[:len(wf)//2]
            dom_freq_idx = np.argmax(fft_vals)
            features['dominant_freq'] = freqs[dom_freq_idx] if dom_freq_idx < len(freqs) else 0
            
            spectral_sum = np.sum(fft_vals)
            if spectral_sum > 0:
                features['spectral_centroid'] = np.sum(freqs * fft_vals) / spectral_sum
            else:
                features['spectral_centroid'] = 0
                
            features_list.append(features)
    else:
        # CPU version - process each waveform individually
        for wf in waveforms:
            features = {}
            
            # Basic time features
            features['mean'] = np.mean(wf)
            features['std'] = np.std(wf)
            features['range'] = np.max(wf) - np.min(wf)
            features['abs_mean'] = np.mean(np.abs(wf))
            features['energy'] = np.sum(wf**2)
            features['zero_crossings'] = np.sum(np.diff(np.signbit(wf)))
            features['peak_count'] = len(signal.find_peaks(wf[:500])[0])
            
            # Statistical moments
            wf_mean = features['mean']
            wf_var = features['std']**2
            wf_centered = wf - wf_mean
            features['skewness'] = np.mean(wf_centered**3) / (wf_var**(3/2)) if wf_var > 0 else 0
            features['kurtosis'] = np.mean(wf_centered**4) / (wf_var**2) if wf_var > 0 else 0
            
            # Complexity
            features['complexity'] = np.sum(np.abs(np.diff(wf)))
            
            # Frequency domain
            fft_vals = np.abs(fft(wf))[:len(wf)//2]
            fft_max = np.max(fft_vals)
            fft_vals = fft_vals / fft_max if fft_max > 0 else fft_vals
            
            # Frequency bands
            num_bands = 10
            band_size = len(fft_vals) // num_bands
            for j in range(num_bands):
                start = j * band_size
                end = (j + 1) * band_size if j < num_bands - 1 else len(fft_vals)
                features[f'band_{j}'] = np.sum(fft_vals[start:end])
            
            # Spectral features
            freqs = np.fft.fftfreq(len(wf))[:len(wf)//2]
            dom_freq_idx = np.argmax(fft_vals)
            features['dominant_freq'] = freqs[dom_freq_idx] if dom_freq_idx < len(freqs) else 0
            
            spectral_sum = np.sum(fft_vals)
            if spectral_sum > 0:
                features['spectral_centroid'] = np.sum(freqs * fft_vals) / spectral_sum
            else:
                features['spectral_centroid'] = 0
                
            features_list.append(features)
            
    return features_list

# Create output directory
output_dir = "cuda_waveform_analysis"
os.makedirs(output_dir, exist_ok=True)

# Generate diverse waveform dataset with wide parameter ranges
print("Generating diverse waveform dataset...")
num_samples = 12000
parameter_ranges = [
    (100, 25000),      # f_sine
    (0.05, 100),       # f_square
    (500, 50000),      # f_triangle
    (1000, 100000)     # f_sawtooth
]

# Use Latin Hypercube Sampling for better parameter space coverage
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=4, seed=42)
sample_points = sampler.random(n=num_samples)

# Scale the samples to the parameter ranges
all_params = []
for point in sample_points:
    scaled_params = [
        low + (high - low) * value 
        for (low, high), value in zip(parameter_ranges, point)
    ]
    all_params.append(scaled_params)

# Generate waveforms and extract features in batches
all_features = []
batch_size = 500 if HAS_GPU else 200  # Larger batches for GPU

# Process in batches with progress bar
for batch_start in tqdm(range(0, num_samples, batch_size), desc="Generating waveforms"):
    batch_end = min(batch_start + batch_size, num_samples)
    batch_params = all_params[batch_start:batch_end]
    
    # Generate waveforms (still on CPU as it's efficient enough)
    batch_waveforms = [generate_waveform(params) for params in batch_params]
    
    # Extract features (GPU-optimized if available)
    batch_features = extract_features_batch(batch_waveforms)
    
    all_features.extend(batch_features)
    
    # Save a few examples from each batch
    for i in range(0, min(5, len(batch_waveforms)), max(1, len(batch_waveforms)//5)):
        plt.figure(figsize=(10, 4))
        plt.plot(batch_waveforms[i][:200])
        plt.title(f"Waveform {batch_start + i}: f_sine={batch_params[i][0]:.0f}, f_square={batch_params[i][1]:.2f}, " +
                  f"f_triangle={batch_params[i][2]:.0f}, f_sawtooth={batch_params[i][3]:.0f}")
        plt.close()
    
    # Clear batch data to save memory
    del batch_waveforms
    gc.collect()

# Convert to DataFrame
print("\nPreprocessing features...")
df = pd.DataFrame(all_features)
param_df = pd.DataFrame(all_params, columns=['f_sine', 'f_square', 'f_triangle', 'f_sawtooth'])
df = pd.concat([df, param_df], axis=1)

# Save raw data
df.to_csv(os.path.join(output_dir, "all_waveform_features.csv"), index=False)

# Standardize features
feature_columns = [col for col in df.columns if col not in ['f_sine', 'f_square', 'f_triangle', 'f_sawtooth']]
X = df[feature_columns].values

# Apply dimensionality reduction and clustering
print("Applying dimensionality reduction and clustering...")
n_components = min(30, X.shape[1])
n_clusters = 128

if HAS_GPU:
    # Use GPU for dimensionality reduction
    X_gpu = cp.array(X, dtype=cp.float32)
    
    # Standardize on GPU
    X_mean = cp.mean(X_gpu, axis=0)
    X_std = cp.std(X_gpu, axis=0)
    X_gpu = (X_gpu - X_mean) / (X_std + 1e-8)  # Add small epsilon to avoid division by zero
    
    # PCA on GPU
    pca_gpu = cuPCA(n_components=n_components)
    X_pca_gpu = pca_gpu.fit_transform(X_gpu)
    
    # K-means on GPU
    kmeans_gpu = cuKMeans(n_clusters=n_clusters, random_state=42, max_iter=300)
    kmeans_gpu.fit(X_pca_gpu)
    
    # Get results back to CPU
    X_pca = cp.asnumpy(X_pca_gpu)
    kmeans_labels = cp.asnumpy(kmeans_gpu.labels_)
    cluster_centers = cp.asnumpy(kmeans_gpu.cluster_centers_)
    
    # Clean GPU memory
    del X_gpu, X_pca_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    # For t-SNE visualization (subsample to save memory)
    sample_indices = np.random.choice(range(num_samples), min(5000, num_samples), replace=False)
    X_pca_sample = X_pca[sample_indices]
    X_pca_gpu_sample = cp.array(X_pca_sample, dtype=cp.float32)
    
    # t-SNE on GPU
    tsne_gpu = cuTSNE(n_components=2, perplexity=50, n_iter=1000, random_state=42)
    X_tsne_gpu = tsne_gpu.fit_transform(X_pca_gpu_sample)
    X_tsne = cp.asnumpy(X_tsne_gpu)
    sample_labels = kmeans_labels[sample_indices]
    
    # Clean GPU memory again
    del X_pca_gpu_sample, X_tsne_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
else:
    # CPU fallback for standardization, PCA, and K-means
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA on CPU with incremental approach to save memory
    ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    X_pca = ipca.fit_transform(X_scaled)
    
    # K-means on CPU with mini-batches
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, max_iter=300)
    kmeans_labels = kmeans.fit_predict(X_pca)
    cluster_centers = kmeans.cluster_centers_
    
    # t-SNE on CPU (subsample to save computation)
    sample_indices = np.random.choice(range(num_samples), min(5000, num_samples), replace=False)
    tsne = TSNE(n_components=2, perplexity=50, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_pca[sample_indices])
    sample_labels = kmeans_labels[sample_indices]

# Save clustering results
np.save(os.path.join(output_dir, "X_pca.npy"), X_pca)
np.save(os.path.join(output_dir, "kmeans_labels.npy"), kmeans_labels)
np.save(os.path.join(output_dir, "cluster_centers.npy"), cluster_centers)

# Add cluster labels to dataframe
df['cluster'] = kmeans_labels
df[['cluster', 'f_sine', 'f_square', 'f_triangle', 'f_sawtooth']].to_csv(
    os.path.join(output_dir, "cluster_parameters.csv"), index=False)
	# Create output directory if it doesn't exist
output_dir = os.path.join("cuda_waveform_analysis", "plots")
os.makedirs(output_dir, exist_ok=True)

# Create t-SNE cluster visualization
plt.figure(figsize=(12, 10))
# We're only using K-means, no DBSCAN in the GPU version
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=sample_labels, palette='viridis', s=50, alpha=0.7)
plt.title(f'K-Means Clustering (k={n_clusters})')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig(os.path.join(output_dir, 'kmeans_clusters_tsne.png'))
plt.close()

# Parameter distributions across clusters
for param in ['f_sine', 'f_square', 'f_triangle', 'f_sawtooth']:
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='cluster', y=param, data=df)  # Note: using 'cluster' instead of 'kmeans_cluster'
    plt.title(f'Distribution of {param} Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(param)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_{param}_distribution.png'))
    plt.close()

# Feature importance visualization
# Need to calculate feature importance first since we're using GPU K-means
print("Calculating feature importance...")

# Extract features for feature importance analysis
feature_columns = [col for col in df.columns if col not in 
                  ['f_sine', 'f_square', 'f_triangle', 'f_sawtooth', 'cluster']]

# Train a Random Forest to get feature importance
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = df[feature_columns].values
y = df['cluster'].values

# Train model to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top Features for Distinguishing Waveform Clusters')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Visualize representative waveforms from each cluster
# Determine grid size based on number of clusters
if n_clusters <= 8:
    n_rows, n_cols = 4, 2
elif n_clusters <= 16:
    n_rows, n_cols = 4, 4
elif n_clusters <= 32:
    n_rows, n_cols = 8, 4
else:
    # For 128 clusters, we'll create multiple figures
    n_rows, n_cols = 16, 8
    # Create multiple figures, 16 clusters per figure
    clusters_per_figure = 16
    num_figures = (n_clusters + clusters_per_figure - 1) // clusters_per_figure
    
    for fig_num in range(num_figures):
        plt.figure(figsize=(20, 15))
        start_cluster = fig_num * clusters_per_figure
        end_cluster = min(start_cluster + clusters_per_figure, n_clusters)
        
        for i, cluster_id in enumerate(range(start_cluster, end_cluster)):
            # Find the waveform closest to cluster center
            cluster_indices = np.where(kmeans_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue  # Skip empty clusters
                
            distances = np.linalg.norm(X_pca[cluster_indices] - cluster_centers[cluster_id], axis=1)
            representative_idx = cluster_indices[np.argmin(distances)]
            
            # Generate the representative waveform
            representative_params = all_params[representative_idx]
            representative_waveform = generate_waveform(representative_params)
            
            plt.subplot(4, 4, i+1)  # 4x4 grid for 16 clusters per figure
            plt.plot(representative_waveform[:200])  # Plot first 200 samples for visibility
            plt.title(f'Cluster {cluster_id} Waveform\n' + 
                     f'f_sine={representative_params[0]:.0f}, f_square={representative_params[1]:.1f}\n' +
                     f'f_triangle={representative_params[2]:.0f}, f_sawtooth={representative_params[3]:.0f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_representative_waveforms_{fig_num+1}.png'))
        plt.close()

    # Also create a figure with the top 16 largest clusters
    cluster_sizes = np.bincount(kmeans_labels)
    largest_clusters = np.argsort(-cluster_sizes)[:16]  # Get indices of 16 largest clusters
    
    plt.figure(figsize=(20, 15))
    for i, cluster_id in enumerate(largest_clusters):
        # Find representative waveform
        cluster_indices = np.where(kmeans_labels == cluster_id)[0]
        distances = np.linalg.norm(X_pca[cluster_indices] - cluster_centers[cluster_id], axis=1)
        representative_idx = cluster_indices[np.argmin(distances)]
        
        # Generate waveform
        representative_params = all_params[representative_idx]
        representative_waveform = generate_waveform(representative_params)
        
        plt.subplot(4, 4, i+1)
        plt.plot(representative_waveform[:200])  # Plot first 200 samples for visibility
        plt.title(f'Cluster {cluster_id} Waveform (Size: {cluster_sizes[cluster_id]})\n' + 
                 f'f_sine={representative_params[0]:.0f}, f_square={representative_params[1]:.1f}\n' +
                 f'f_triangle={representative_params[2]:.0f}, f_sawtooth={representative_params[3]:.0f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'largest_clusters_waveforms.png'))
    plt.close()

# Parameter relationship visualization
# We'll sample a subset of points for this visualization to avoid overcrowding
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42)

plt.figure(figsize=(12, 10))
sns.scatterplot(x='f_sine', y='f_triangle', hue='cluster', data=df_sample, 
                palette='viridis', s=60, alpha=0.7)
plt.title('Relationship Between f_sine and f_triangle Parameters by Cluster')
plt.savefig(os.path.join(output_dir, 'parameter_relationship.png'))
plt.close()

# Additional: Create a correlation matrix for parameters and cluster assignment
# This helps understand which parameters most influence cluster assignment
plt.figure(figsize=(10, 8))
param_cluster_df = df[['f_sine', 'f_square', 'f_triangle', 'f_sawtooth', 'cluster']]
correlation = param_cluster_df.corr().abs()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Between Parameters and Cluster Assignment')
plt.savefig(os.path.join(output_dir, 'parameter_cluster_correlation.png'))
plt.close()

print(f"All visualizations saved to {output_dir}/")
