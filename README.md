# CUDA-Accelerated Waveform Analysis

## Overview
This project implements large-scale waveform generation, feature extraction, and clustering with GPU acceleration using CUDA via the RAPIDS suite (cuDF, cuML, cuPy). The system automatically falls back to CPU processing when GPU resources are unavailable.

## Features
- **GPU Acceleration**: Utilizes NVIDIA RAPIDS libraries for accelerated data processing
- **Automatic Fallback**: Gracefully reverts to CPU processing when GPU is unavailable
- **Waveform Generation**: Creates synthetic waveforms with complex parameter combinations
- **Feature Extraction**: Computes time and frequency domain features from waveforms
- **Dimensionality Reduction**: Implements PCA for feature compression
- **Clustering**: Groups similar waveforms using K-means
- **Visualization**: Generates comprehensive plots to analyze clusters and feature importance

## Requirements
- Python 3.6+
- NumPy, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn
- TQDM (for progress bars)
- Optional GPU dependencies:
  - NVIDIA GPU with CUDA support
  - RAPIDS suite (cuDF, cuML, cuPy)

## Installation
```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm

# GPU acceleration (optional)
pip install cudf-cuda11 cuml-cuda11 cupy-cuda11
# Adjust the CUDA version as needed for your system
```

## Usage
Simply run the main script:
```bash
python pipeline.py
```

The script will:
1. Check for GPU availability and inform the user
2. Generate a diverse dataset of synthetic waveforms
3. Extract features from each waveform
4. Perform dimensionality reduction and clustering
5. Generate visualizations and analysis
6. Save results to the `cuda_waveform_analysis` directory

## Output Structure
```
cuda_waveform_analysis/
├── all_waveform_features.csv   # Raw feature data
├── X_pca.npy                   # PCA-transformed features
├── kmeans_labels.npy           # Cluster assignments
├── cluster_centers.npy         # Cluster centroids
├── cluster_parameters.csv      # Mapping of clusters to parameters
└── plots/
    ├── kmeans_clusters_tsne.png           # t-SNE visualization of clusters
    ├── cluster_*_distribution.png         # Parameter distributions by cluster
    ├── feature_importance.png             # Top features for cluster separation
    ├── cluster_representative_waveforms_*.png  # Example waveforms from each cluster
    ├── largest_clusters_waveforms.png     # Waveforms from most common clusters
    ├── parameter_relationship.png         # Relationship between parameters
    └── parameter_cluster_correlation.png  # Correlation heatmap
```

## Implementation Details

### Waveform Generation
The system generates complex waveforms by combining sine, square, triangle, and sawtooth waves with various frequency parameters. Latin Hypercube Sampling ensures efficient parameter space coverage.

Cluster Samples:
<img width="1189" height="990" alt="cluster_016_samples_5x1" src="https://github.com/user-attachments/assets/14dbca7b-1a79-4e59-b80a-7ea2bf79c595" />

### Feature Extraction

For each waveform, the system extracts:
- Basic statistics (mean, standard deviation, range)
- Signal characteristics (zero crossings, peak count)
- Statistical moments (skewness, kurtosis)
- Complexity measures (total variation)
- Frequency domain features (spectral bands, dominant frequency)

### Dimensionality Reduction and Clustering
The high-dimensional feature space is reduced using PCA, then clustered using K-means to identify similar waveform patterns. t-SNE is used for visualization purposes.
<img width="1294" height="989" alt="tsne_overview" src="https://github.com/user-attachments/assets/6bf3b6bc-9903-40ca-bde2-95a509efb462" />

### Real Physical Replication of EMFs using Frequency Multiplyer

Taking the dimensions given from the discovery model, we can replicate these waveforms in a tanglible way.
![SsfnEx4y](https://github.com/user-attachments/assets/54d759d9-e5c6-4f15-aa82-5374d7bb521d)

### GPU Acceleration
When available, the system uses:
- cuPy for accelerated array operations
- cuML for PCA, t-SNE and K-means
- Batch processing to optimize memory usage

## Performance Considerations
- Memory usage scales with the number of waveforms and features
- Batch processing helps manage memory constraints
- GPU acceleration provides significant speedup for large datasets
- The code automatically adjusts batch sizes based on available resources

## Extending the Project
- Add new waveform generation functions
- Implement additional feature extraction methods
- Experiment with different clustering algorithms
- Export identified clusters for use in audio synthesis or signal processing applications
