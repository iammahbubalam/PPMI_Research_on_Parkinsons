"""
Segmentation Comparison Script for DaTscan Image Slices
This script compares 5 different clustering algorithms for segmentation:
1. K-Means
2. Gaussian Mixture Model (GMM)
3. Mini Batch K-Means (Memory efficient)
4. DBSCAN
5. Mean Shift

Benchmarking metric:
- Execution Time (seconds)
"""

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_clustering_metrics(pixels, labels, sample_size=10000):
    """
    Calculate clustering quality metrics efficiently using sampling
    
    Parameters:
    -----------
    pixels : ndarray
        The pixel data
    labels : ndarray
        Cluster labels
    sample_size : int
        Number of pixels to sample for metric calculation (to avoid memory issues)
    
    Returns:
    --------
    dict : Dictionary containing the metrics
    """
    metrics = {}
    
    try:
        # Get unique labels
        unique_labels = np.unique(labels)
        
        # Need at least 2 clusters
        if len(unique_labels) < 2:
            metrics['silhouette_score'] = -1
            metrics['davies_bouldin_index'] = np.inf
            metrics['calinski_harabasz_score'] = 0
            return metrics
        
        # Sample pixels if dataset is too large
        if len(pixels) > sample_size:
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels_sample = pixels[indices]
            labels_sample = labels[indices]
        else:
            pixels_sample = pixels
            labels_sample = labels
        
        # Calculate metrics
        metrics['silhouette_score'] = silhouette_score(pixels_sample, labels_sample)
        metrics['davies_bouldin_index'] = davies_bouldin_score(pixels_sample, labels_sample)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(pixels_sample, labels_sample)
        
    except Exception as e:
        # If any error occurs, return default values
        metrics['silhouette_score'] = -1
        metrics['davies_bouldin_index'] = np.inf
        metrics['calinski_harabasz_score'] = 0
    
    return metrics


def kmeans_segmentation(image_path, num_clusters=3, threshold=None):
    """
    K-Means clustering segmentation - Original function
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply threshold if provided 
    if threshold is not None:
        img_array[img_array < threshold] = 0
    
    # Flatten the image array to a 2D array of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Start timing
    start_time = time.time()
    
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    
    # Fit KMeans model to the pixels
    kmeans.fit(pixels)
    
    # Get the labels and cluster centers
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # End timing
    execution_time = time.time() - start_time
    
    # Calculate clustering quality metrics
    quality_metrics = calculate_clustering_metrics(pixels, labels)
    
    # Find the brightest cluster
    brightest_cluster_center = np.argmax(np.sum(cluster_centers, axis=1))
    
    # Create modified cluster centers (only keep brightest)
    modified_cluster_centers = np.zeros_like(cluster_centers)
    modified_cluster_centers[brightest_cluster_center] = cluster_centers[brightest_cluster_center]
    
    # Create segmented image
    segmented_img_array = modified_cluster_centers[labels].reshape(img_array.shape).astype(np.uint8)
    compressed_img = Image.fromarray(segmented_img_array)
    
    # Combine metrics
    metrics = {
        'execution_time': execution_time,
        **quality_metrics
    }
    
    return compressed_img, metrics
 
def gmm_segmentation(image_path, num_clusters=3, threshold=None):
    """
    Gaussian Mixture Model (GMM) segmentation - works exactly like K-Means
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply threshold if provided 
    if threshold is not None:
        img_array[img_array < threshold] = 0
    
    # Flatten the image array to a 2D array of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Start timing
    start_time = time.time()
    
    # Initialize GMM model
    gmm = GaussianMixture(n_components=num_clusters, random_state=42, covariance_type='full')
    
    # Fit GMM and predict labels
    labels = gmm.fit_predict(pixels)
    cluster_centers = gmm.means_
    
    # End timing
    execution_time = time.time() - start_time
    
    # Calculate clustering quality metrics
    quality_metrics = calculate_clustering_metrics(pixels, labels)
    
    # Find the brightest cluster
    brightest_cluster_center = np.argmax(np.sum(cluster_centers, axis=1))
    
    # Create modified cluster centers (only keep brightest)
    modified_cluster_centers = np.zeros_like(cluster_centers)
    modified_cluster_centers[brightest_cluster_center] = cluster_centers[brightest_cluster_center]
    
    # Create segmented image
    segmented_img_array = modified_cluster_centers[labels].reshape(img_array.shape).astype(np.uint8)
    compressed_img = Image.fromarray(segmented_img_array)
    
    # Combine metrics
    metrics = {
        'execution_time': execution_time,
        **quality_metrics
    }
    
    return compressed_img, metrics


def minibatch_kmeans_segmentation(image_path, num_clusters=3, threshold=None):
    """
    Mini Batch K-Means clustering - Memory efficient version of K-Means
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply threshold if provided 
    if threshold is not None:
        img_array[img_array < threshold] = 0
    
    # Flatten the image array to a 2D array of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Start timing
    start_time = time.time()
    
    # Initialize MiniBatchKMeans
    minibatch_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1000)
    
    # Fit and predict
    labels = minibatch_kmeans.fit_predict(pixels)
    cluster_centers = minibatch_kmeans.cluster_centers_
    
    # End timing
    execution_time = time.time() - start_time
    
    # Calculate clustering quality metrics
    quality_metrics = calculate_clustering_metrics(pixels, labels)
    
    # Find the brightest cluster
    brightest_cluster_center = np.argmax(np.sum(cluster_centers, axis=1))
    
    # Create modified cluster centers (only keep brightest)
    modified_cluster_centers = np.zeros_like(cluster_centers)
    modified_cluster_centers[brightest_cluster_center] = cluster_centers[brightest_cluster_center]
    
    # Create segmented image
    segmented_img_array = modified_cluster_centers[labels].reshape(img_array.shape).astype(np.uint8)
    compressed_img = Image.fromarray(segmented_img_array)
    
    # Combine metrics
    metrics = {
        'execution_time': execution_time,
        **quality_metrics
    }
    
    return compressed_img, metrics


def dbscan_segmentation(image_path, eps=10, min_samples=50, threshold=None):
    """
    DBSCAN clustering - works exactly like K-Means
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply threshold if provided 
    if threshold is not None:
        img_array[img_array < threshold] = 0
    
    # Flatten the image array to a 2D array of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Start timing
    start_time = time.time()
    
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Fit and predict
    labels = dbscan.fit_predict(pixels)
    
    # Get unique labels (excluding noise if present)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
    
    # Calculate cluster centers for non-noise points
    if len(unique_labels) > 0:
        cluster_centers = np.array([pixels[labels == label].mean(axis=0) for label in unique_labels])
    else:
        cluster_centers = np.array([[0, 0, 0]])
    
    # End timing
    execution_time = time.time() - start_time
    
    # Calculate clustering quality metrics (only for non-noise points)
    valid_mask = labels != -1
    if np.sum(valid_mask) > 0 and len(unique_labels) > 1:
        quality_metrics = calculate_clustering_metrics(pixels[valid_mask], labels[valid_mask])
    else:
        quality_metrics = {
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0
        }
    
    # Find the brightest cluster
    if len(cluster_centers) > 0:
        brightest_cluster_center = np.argmax(np.sum(cluster_centers, axis=1))
        brightest_label = unique_labels[brightest_cluster_center]
    else:
        brightest_label = -1
    
    # Create modified cluster centers (only keep brightest)
    modified_cluster_centers = np.zeros((len(unique_labels), 3))
    if len(unique_labels) > 0:
        modified_cluster_centers[brightest_cluster_center] = cluster_centers[brightest_cluster_center]
    
    # Map old labels to new indices
    label_map = {old_label: idx for idx, old_label in enumerate(unique_labels)}
    label_map[-1] = 0  # Noise points map to 0
    
    # Remap labels
    remapped_labels = np.array([label_map[l] if l in label_map else 0 for l in labels])
    
    # Create segmented image
    segmented_img_array = modified_cluster_centers[remapped_labels].reshape(img_array.shape).astype(np.uint8)
    compressed_img = Image.fromarray(segmented_img_array)
    
    # Combine metrics
    metrics = {
        'execution_time': execution_time,
        **quality_metrics
    }
    
    return compressed_img, metrics


def meanshift_segmentation(image_path, bandwidth=30, threshold=None):
    """
    Mean Shift clustering - works exactly like K-Means
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Apply threshold if provided 
    if threshold is not None:
        img_array[img_array < threshold] = 0
    
    # Flatten the image array to a 2D array of pixels
    pixels = img_array.reshape((-1, 3))
    
    # Start timing
    start_time = time.time()
    
    # Initialize Mean Shift
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    
    # Fit and predict
    labels = meanshift.fit_predict(pixels)
    cluster_centers = meanshift.cluster_centers_
    
    # End timing
    execution_time = time.time() - start_time
    
    # Calculate clustering quality metrics
    quality_metrics = calculate_clustering_metrics(pixels, labels)
    
    # Find the brightest cluster
    brightest_cluster_center = np.argmax(np.sum(cluster_centers, axis=1))
    
    # Create modified cluster centers (only keep brightest)
    modified_cluster_centers = np.zeros_like(cluster_centers)
    modified_cluster_centers[brightest_cluster_center] = cluster_centers[brightest_cluster_center]
    
    # Create segmented image
    segmented_img_array = modified_cluster_centers[labels].reshape(img_array.shape).astype(np.uint8)
    compressed_img = Image.fromarray(segmented_img_array)
    
    # Combine metrics
    metrics = {
        'execution_time': execution_time,
        **quality_metrics
    }
    
    return compressed_img, metrics




def benchmark_all_algorithms(image_path, num_clusters=3, threshold=None, 
                             dbscan_eps=10, dbscan_min_samples=50, 
                             meanshift_bandwidth=30):
    """
    Run all 5 segmentation algorithms on a single image and compare results
    """
    results = {}
    
    # K-Means
    try:
        seg_img, metrics = kmeans_segmentation(image_path, num_clusters, threshold)
        results['KMeans'] = {'image': seg_img, **metrics}
    except Exception as e:
        print(f"  K-Means failed: {e}")
        results['KMeans'] = {
            'execution_time': 0, 
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0,
            'error': str(e)
        }
    
    # GMM
    try:
        seg_img, metrics = gmm_segmentation(image_path, num_clusters, threshold)
        results['GMM'] = {'image': seg_img, **metrics}
    except Exception as e:
        print(f"  GMM failed: {e}")
        results['GMM'] = {
            'execution_time': 0,
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0,
            'error': str(e)
        }
    
    # MiniBatch KMeans
    try:
        seg_img, metrics = minibatch_kmeans_segmentation(image_path, num_clusters, threshold)
        results['MiniBatchKMeans'] = {'image': seg_img, **metrics}
    except Exception as e:
        print(f"  MiniBatchKMeans failed: {e}")
        results['MiniBatchKMeans'] = {
            'execution_time': 0,
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0,
            'error': str(e)
        }
    
    # DBSCAN
    try:
        seg_img, metrics = dbscan_segmentation(image_path, dbscan_eps, dbscan_min_samples, threshold)
        results['DBSCAN'] = {'image': seg_img, **metrics}
    except Exception as e:
        print(f"  DBSCAN failed: {e}")
        results['DBSCAN'] = {
            'execution_time': 0,
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0,
            'error': str(e)
        }
    
    # Mean Shift
    try:
        seg_img, metrics = meanshift_segmentation(image_path, meanshift_bandwidth, threshold)
        results['MeanShift'] = {'image': seg_img, **metrics}
    except Exception as e:
        print(f"  Mean Shift failed: {e}")
        results['MeanShift'] = {
            'execution_time': 0,
            'silhouette_score': -1,
            'davies_bouldin_index': np.inf,
            'calinski_harabasz_score': 0,
            'error': str(e)
        }
    
    return results


def process_random_samples(input_dir, num_samples=100, output_dir=None, 
                          num_clusters=3, threshold=None,
                          dbscan_eps=10, dbscan_min_samples=50,
                          meanshift_bandwidth=30):
    """
    Process random samples - Run each algorithm on all samples sequentially
    """
    # Get all image files
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Randomly sample ONCE
    if len(image_files) > num_samples:
        np.random.seed(42)
        sampled_files = np.random.choice(image_files, num_samples, replace=False)
    else:
        sampled_files = image_files
        print(f"Using all {len(sampled_files)} images (less than requested {num_samples})")
    
    print(f"\nSelected {len(sampled_files)} random samples")
    print("="*80)
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for algo in ['KMeans', 'GMM', 'MiniBatchKMeans', 'DBSCAN', 'MeanShift']:
            (output_path / algo).mkdir(exist_ok=True)
    
    # Dictionary to store all results by image
    results_by_image = {}
    
    # Initialize results structure
    for image_file in sampled_files:
        results_by_image[image_file.name] = {'image_path': str(image_file)}
    
    # Define algorithms
    algorithms = [
        ('KMeans', kmeans_segmentation, {'num_clusters': num_clusters, 'threshold': threshold}),
        ('GMM', gmm_segmentation, {'num_clusters': num_clusters, 'threshold': threshold}),
        ('MiniBatchKMeans', minibatch_kmeans_segmentation, {'num_clusters': num_clusters, 'threshold': threshold}),
        ('DBSCAN', dbscan_segmentation, {'eps': dbscan_eps, 'min_samples': dbscan_min_samples, 'threshold': threshold}),
        ('MeanShift', meanshift_segmentation, {'bandwidth': meanshift_bandwidth, 'threshold': threshold})
    ]
    
    # Run each algorithm on all samples
    for algo_name, algo_func, params in algorithms:
        print(f"\n{'='*80}")
        print(f"Running {algo_name} on all {len(sampled_files)} samples...")
        print(f"{'='*80}")
        
        for idx, image_file in enumerate(sampled_files):
            print(f"[{idx+1}/{len(sampled_files)}] {algo_name}: {image_file.name}", end=' ... ')
            
            try:
                # Run the algorithm
                seg_img, metrics = algo_func(str(image_file), **params)
                
                # Save segmented image if output directory specified
                if output_dir:
                    output_file = output_path / algo_name / f"{image_file.stem}_{algo_name}.png"
                    seg_img.save(output_file)
                
                # Store results
                results_by_image[image_file.name][f"{algo_name}_execution_time"] = metrics.get('execution_time', 0)
                results_by_image[image_file.name][f"{algo_name}_silhouette_score"] = metrics.get('silhouette_score', -1)
                results_by_image[image_file.name][f"{algo_name}_davies_bouldin_index"] = metrics.get('davies_bouldin_index', np.inf)
                results_by_image[image_file.name][f"{algo_name}_calinski_harabasz_score"] = metrics.get('calinski_harabasz_score', 0)
                
                print(f"✓ ({metrics.get('execution_time', 0):.3f}s)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                results_by_image[image_file.name][f"{algo_name}_execution_time"] = 0
                results_by_image[image_file.name][f"{algo_name}_silhouette_score"] = -1
                results_by_image[image_file.name][f"{algo_name}_davies_bouldin_index"] = np.inf
                results_by_image[image_file.name][f"{algo_name}_calinski_harabasz_score"] = 0
                results_by_image[image_file.name][f"{algo_name}_error"] = str(e)
    
    # Convert to list of dictionaries for DataFrame
    detailed_results = []
    for image_name, data in results_by_image.items():
        result_row = {'image_name': image_name}
        result_row.update({k: v for k, v in data.items() if k != 'image_path'})
        detailed_results.append(result_row)
    
    # Create dataframe
    df = pd.DataFrame(detailed_results)
    
    # Calculate summary statistics
    algorithms = ['KMeans', 'GMM', 'MiniBatchKMeans', 'DBSCAN', 'MeanShift']
    metrics = ['execution_time', 'silhouette_score', 'davies_bouldin_index', 'calinski_harabasz_score']
    
    summary_data = []
    for algo in algorithms:
        algo_summary = {'Algorithm': algo}
        for metric in metrics:
            col_name = f"{algo}_{metric}"
            if col_name in df.columns:
                if metric == 'davies_bouldin_index':
                    # Lower is better, filter out inf values
                    valid_values = df[col_name].replace([np.inf, -np.inf], np.nan).dropna()
                elif metric == 'silhouette_score':
                    # Filter out -1 values (errors)
                    valid_values = df[col_name][df[col_name] > -1]
                else:
                    valid_values = df[col_name][df[col_name] > 0]
                
                if len(valid_values) > 0:
                    algo_summary[f"{metric}_mean"] = valid_values.mean()
                    algo_summary[f"{metric}_std"] = valid_values.std()
                    algo_summary[f"{metric}_median"] = valid_values.median()
                    algo_summary[f"{metric}_min"] = valid_values.min()
                    algo_summary[f"{metric}_max"] = valid_values.max()
                else:
                    algo_summary[f"{metric}_mean"] = 0 if metric != 'davies_bouldin_index' else np.inf
                    algo_summary[f"{metric}_std"] = 0
                    algo_summary[f"{metric}_median"] = 0
                    algo_summary[f"{metric}_min"] = 0
                    algo_summary[f"{metric}_max"] = 0
        
        # Count successful runs
        exec_time_col = f"{algo}_execution_time"
        if exec_time_col in df.columns:
            algo_summary['successful_runs'] = len(df[exec_time_col][df[exec_time_col] > 0])
        else:
            algo_summary['successful_runs'] = 0
            
        summary_data.append(algo_summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df, df


def create_comparison_table(summary_df):
    """
    Create a simple comparison table for publication
    Returns a clean DataFrame
    """
    table_data = []
    
    for _, row in summary_df.iterrows():
        table_row = {
            'Algorithm': row['Algorithm'],
            'Execution_Time_Mean': round(row.get('execution_time_mean', 0), 3),
            'Execution_Time_Std': round(row.get('execution_time_std', 0), 3),
            'Silhouette_Score_Mean': round(row.get('silhouette_score_mean', 0), 3),
            'Silhouette_Score_Std': round(row.get('silhouette_score_std', 0), 3),
            'Davies_Bouldin_Mean': round(row.get('davies_bouldin_index_mean', 0), 3) if row.get('davies_bouldin_index_mean', np.inf) != np.inf else 'N/A',
            'Davies_Bouldin_Std': round(row.get('davies_bouldin_index_std', 0), 3) if row.get('davies_bouldin_index_mean', np.inf) != np.inf else 'N/A',
            'Calinski_Harabasz_Mean': round(row.get('calinski_harabasz_score_mean', 0), 1),
            'Calinski_Harabasz_Std': round(row.get('calinski_harabasz_score_std', 0), 1),
            'Success_Rate': f"{row.get('successful_runs', 0)}/100"
        }
        table_data.append(table_row)
    
    comparison_table = pd.DataFrame(table_data)
    return comparison_table


def print_comparison_report(summary_df):
    """
    Create and save comparison table - simple output only
    """
    # Create simple comparison table
    comparison_table = create_comparison_table(summary_df)
    print("\nComparison table created successfully")
    return comparison_table


def convert_to_binary(img):
    """
    Convert segmented image to binary (black background, white segmented region)
    """
    img_array = np.array(img)
    # If the image has any non-zero values, make them white (255)
    binary = np.zeros_like(img_array)
    # Check if any channel has non-zero values
    mask = np.any(img_array > 0, axis=-1)
    binary[mask] = 255
    return binary


def visualize_segmentation_comparison(image_paths, num_samples=3, output_dir=None,
                                     num_clusters=3, threshold=None,
                                     dbscan_eps=10, dbscan_min_samples=50,
                                     meanshift_bandwidth=30):
    """
    Visualize segmentation comparison - Creates ONE image per slice
    Each image shows: Original + 5 algorithm results (6 images total)
    Segmented regions are WHITE on BLACK background
    For 3 slices, creates 3 separate images
     
    Parameters:
    -----------
    image_paths : list or str
        List of image paths or directory path
    num_samples : int
        Number of samples to visualize (if directory path provided)
    output_dir : str
        Directory to save the visualizations (optional)
    """
    # Get image paths
    if isinstance(image_paths, str):
        # It's a directory
        input_path = Path(image_paths)
        all_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        if len(all_files) > num_samples:
            np.random.seed(42)
            image_paths = np.random.choice(all_files, num_samples, replace=False)
        else:
            image_paths = all_files[:num_samples]
    
    num_samples = len(image_paths)
    algorithms = ['Original', 'KMeans', 'GMM', 'MiniBatchKMeans', 'DBSCAN', 'MeanShift']
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Process each slice separately - ONE IMAGE PER SLICE
    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing slice {idx+1}/{num_samples}: {Path(image_path).name}")
        
        # Load original image
        original_img = Image.open(image_path)
        
        # Run all segmentation algorithms
        try:
            kmeans_img, _ = kmeans_segmentation(str(image_path), num_clusters, threshold)
        except Exception as e:
            print(f"  KMeans failed: {e}")
            kmeans_img = Image.fromarray(np.zeros_like(np.array(original_img)))
        
        try:
            gmm_img, _ = gmm_segmentation(str(image_path), num_clusters, threshold)
        except Exception as e:
            print(f"  GMM failed: {e}")
            gmm_img = Image.fromarray(np.zeros_like(np.array(original_img)))
        
        try:
            minibatch_img, _ = minibatch_kmeans_segmentation(str(image_path), num_clusters, threshold)
        except Exception as e:
            print(f"  MiniBatchKMeans failed: {e}")
            minibatch_img = Image.fromarray(np.zeros_like(np.array(original_img)))
        
        try:
            dbscan_img, _ = dbscan_segmentation(str(image_path), dbscan_eps, dbscan_min_samples, threshold)
        except Exception as e:
            print(f"  DBSCAN failed: {e}")
            dbscan_img = Image.fromarray(np.zeros_like(np.array(original_img)))
        
        try:
            meanshift_img, _ = meanshift_segmentation(str(image_path), meanshift_bandwidth, threshold)
        except Exception as e:
            print(f"  MeanShift failed: {e}")
            meanshift_img = Image.fromarray(np.zeros_like(np.array(original_img)))
        
        # Prepare images list: Original + 5 segmented (as binary)
        # The segmentation functions already return only the brightest cluster
        # Just need to convert them to grayscale properly
        images = [
            original_img,  # Keep original as-is
            kmeans_img,    # Already has only brightest cluster
            gmm_img,
            minibatch_img,
            dbscan_img,
            meanshift_img
        ]
        
        # Create ONE figure for this slice with 6 images (original + 5 algorithms)
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        
        # Plot all images
        for col, (img, title) in enumerate(zip(images, algorithms)):
            if col == 0:
                # Show original image in grayscale
                axes[col].imshow(img, cmap='gray')
            else:
                # Show segmented images - brightest cluster is white, rest is black
                img_array = np.array(img)
                # Convert to grayscale if RGB
                if len(img_array.shape) == 3:
                    img_gray = np.mean(img_array, axis=2)
                else:
                    img_gray = img_array
                axes[col].imshow(img_gray, cmap='gray', vmin=0, vmax=255)
            
            axes[col].set_title(title, fontsize=12, fontweight='bold')
            axes[col].axis('off')
        
        # Add slice name as super title
        fig.suptitle(f"Slice: {Path(image_path).name}", fontsize=10, y=0.98)
        
        plt.tight_layout()
        
        # Save individual image for this slice
        if output_dir:
            output_file = output_path / f"comparison_slice_{idx+1}_{Path(image_path).stem}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            saved_files.append(str(output_file))
            print(f"  ✓ Saved: {output_file.name}")
        
        plt.close(fig)
    
    if saved_files:
        print(f"\n{'='*80}")
        print(f"Created {len(saved_files)} separate images (one per slice):")
        for f in saved_files:
            print(f"  - {f}")
        print(f"{'='*80}")
    
    return saved_files
    
    return fig


if __name__ == "__main__":
    import sys
    
    # Configuration
    INPUT_DIR = "/home/m8m/Projects/PPMI_Research_on_Parkinsons/src/review/slices_output"
    OUTPUT_DIR = "/home/m8m/Projects/PPMI_Research_on_Parkinsons/src/review/segmentation_comparison"
    NUM_CLUSTERS = 7  # CRITICAL: Use 3 clusters for speed and medical imaging standard
    
    print("="*80)
    print("COMPLETE SEGMENTATION ANALYSIS: Benchmark + Visualization")
    print("="*80)
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Number of Clusters: {NUM_CLUSTERS}")
    print(f"Benchmark Samples: 100")
    print(f"Visualization Samples: 3")
    print("="*80)
    print("\n")
    
    # # STEP 1: Run benchmark on 100 random samples
    # print("STEP 1/2: Running benchmark on 100 samples...")
    # print("="*80)
    # summary_df, detailed_df = process_random_samples(
    #     input_dir=INPUT_DIR,
    #     num_samples=100,
    #     output_dir=OUTPUT_DIR,
    #     num_clusters=NUM_CLUSTERS,
    #     threshold=None,
    #     dbscan_eps=10,
    #     dbscan_min_samples=50,
    #     meanshift_bandwidth=30
    # )
    
    # # Create comparison table
    # comparison_table = print_comparison_report(summary_df)
    
    # # Save all results
    # comparison_table.to_csv(f"{OUTPUT_DIR}/comparison_table.csv", index=False)
    # summary_df.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
    # detailed_df.to_csv(f"{OUTPUT_DIR}/detailed_results.csv", index=False)
    
    # print(f"\n✓ Benchmark complete! Results saved to {OUTPUT_DIR}")
    # print(f"  - comparison_table.csv (USE THIS FOR YOUR PAPER)")
    # print(f"  - summary_statistics.csv")
    # print(f"  - detailed_results.csv")
    
    # STEP 2: Create visualizations
    print("\n" + "="*80)
    print("STEP 2/2: Creating visualizations (3 separate images, one per slice)...")
    print("="*80)
    saved_images = visualize_segmentation_comparison(
        image_paths=INPUT_DIR,
        num_samples=10,
        output_dir=OUTPUT_DIR,
        num_clusters=NUM_CLUSTERS,
        threshold=None,
        dbscan_eps=10,
        dbscan_min_samples=50,
        meanshift_bandwidth=30
    )
    
    # Final summary
    print("\n" + "="*80)
    print("ALL DONE! ✓")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR}/comparison_table.csv - Main results for paper")
    print(f"  2. {OUTPUT_DIR}/summary_statistics.csv - Detailed statistics")
    print(f"  3. {OUTPUT_DIR}/detailed_results.csv - Per-image results")
    print(f"  4. {OUTPUT_DIR}/comparison_slice_*.png - 3 visualization images (one per slice)")
    print(f"  5. {OUTPUT_DIR}/[Algorithm]/*.png - All segmented images")
    print("="*80)
