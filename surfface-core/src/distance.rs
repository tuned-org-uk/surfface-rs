// surfface-core/src/distance.rs
//! Distance metrics for centroid and feature comparisons
//!
//! Implements:
//! - Bhattacharyya distance (diagonal Gaussian) [file:3]
//! - Euclidean and squared Euclidean distances
//! - Affinity conversions for graph edge weights
//!
//! The Bhattacharyya distance measures statistical overlap between
//! Gaussian distributions and is used for:
//! - MST edge weighting (Stage B1) [file:4]
//! - Feature-space Laplacian construction (Stage C) [file:2]

use burn::prelude::*;

/// Diagonal Gaussian Bhattacharyya distance [file:3]
///
/// Computes the Bhattacharyya distance between two Gaussian distributions
/// with diagonal covariance matrices:
///
/// D_B = Σ_k [ 0.25 * (μᵢᵏ - μⱼᵏ)² / (σᵢᵏ² + σⱼᵏ²)
///           + 0.25 * ln((σᵢᵏ² + σⱼᵏ²) / (2√(σᵢᵏ²σⱼᵏ²))) ]
///
/// Where:
/// - μᵢ, μⱼ: mean vectors
/// - σᵢ², σⱼ²: variance vectors (diagonal covariance)
///
/// Returns a scalar distance (sum over all dimensions)
pub fn bhattacharyya_diagonal<B: Backend>(
    mean_i: Tensor<B, 1>,
    var_i: Tensor<B, 1>,
    mean_j: Tensor<B, 1>,
    var_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let eps = 1e-10; // Numerical stability

    // Regularize variances to prevent division by zero
    let var_i_reg = var_i.clamp_min(eps);
    let var_j_reg = var_j.clamp_min(eps);

    // Variance sum: σᵢ² + σⱼ²
    let sigma_sum = var_i_reg.clone() + var_j_reg.clone();

    // Variance product: σᵢ² * σⱼ²
    let sigma_prod = var_i_reg * var_j_reg;

    // Mean difference: (μᵢ - μⱼ)²
    let mean_diff = mean_i - mean_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    // Mahalanobis term: 0.25 * (μᵢ - μⱼ)² / (σᵢ² + σⱼ²)
    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);

    // Log-determinant term: 0.25 * ln((σᵢ² + σⱼ²) / (2√(σᵢ²σⱼ²)))
    // Simplified: 0.25 * ln(σ_sum) - 0.25 * ln(2) - 0.25 * 0.5 * ln(σ_prod)
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .clamp_min(eps)
        .log()
        .mul_scalar(0.25);

    // Sum over feature dimensions
    (mahalanobis + log_term).sum()
}

/// Bhattacharyya distance for slices (CPU-friendly version)
///
/// Used in MST construction where we need to compute distances
/// between pairs of centroids efficiently without full tensor operations.
///
/// # Arguments
/// * `mean_i` - Mean vector for distribution i (length F)
/// * `var_i` - Variance vector for distribution i (length F)
/// * `mean_j` - Mean vector for distribution j (length F)
/// * `var_j` - Variance vector for distribution j (length F)
///
/// # Returns
/// Scalar Bhattacharyya distance
pub fn bhattacharyya_distance_diagonal(
    mean_i: &[f32],
    var_i: &[f32],
    mean_j: &[f32],
    var_j: &[f32],
) -> f32 {
    assert_eq!(mean_i.len(), mean_j.len());
    assert_eq!(var_i.len(), var_j.len());
    assert_eq!(mean_i.len(), var_i.len());

    let eps = 1e-10f32;
    let mut distance = 0.0f32;

    for k in 0..mean_i.len() {
        let sigma_i = var_i[k].max(eps);
        let sigma_j = var_j[k].max(eps);
        let sigma_sum = sigma_i + sigma_j;
        let sigma_prod = sigma_i * sigma_j;

        // Mahalanobis term
        let mean_diff = mean_i[k] - mean_j[k];
        let mahalanobis = 0.25 * (mean_diff * mean_diff) / sigma_sum;

        // Log-determinant term
        let log_term = 0.25 * ((sigma_sum / (2.0 * sigma_prod.sqrt())).max(eps)).ln();

        distance += mahalanobis + log_term;
    }

    distance
}

/// Affinity (for graph edge weights): w = exp(-D_B) [file:3]
///
/// Converts Bhattacharyya distance to an affinity weight:
/// - Distance 0 → Affinity 1 (identical distributions)
/// - Distance ∞ → Affinity 0 (no overlap)
pub fn bhattacharyya_affinity<B: Backend>(
    mean_i: Tensor<B, 1>,
    var_i: Tensor<B, 1>,
    mean_j: Tensor<B, 1>,
    var_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let distance = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j);
    distance.neg().exp()
}

/// Batch Bhattacharyya for pairwise feature distances (F x F) [file:2]
///
/// Computes all pairwise Bhattacharyya distances between features
/// in feature-space (transposed centroids).
///
/// # Arguments
/// * `features` - Feature vectors [F, C] where each row is a feature's
///                values across C centroids
/// * `variances` - Variance vectors [F, C] for each feature
///
/// # Returns
/// Distance matrix [F, F] where entry (i, j) is D_B(feature_i, feature_j)
///
/// # Performance
/// - Memory: O(F² + FC)
/// - Time: O(F²C)
///
/// For large F (e.g., 100K), use sparse k-NN approximation instead [file:2]
pub fn bhattacharyya_pairwise<B: Backend>(
    features: Tensor<B, 2>,
    variances: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [f, c] = features.dims();
    let eps = 1e-10;

    // Regularize variances
    let variances_reg = variances.clamp_min(eps);

    // Expand for broadcasting: [F, 1, C] and [1, F, C]
    let features_i = features.clone().reshape([f, 1, c]);
    let features_j = features.clone().reshape([1, f, c]);
    let var_i = variances_reg.clone().reshape([f, 1, c]);
    let var_j = variances_reg.clone().reshape([1, f, c]);

    // Vectorized computation [F, F, C]
    let sigma_sum = var_i.clone() + var_j.clone();
    let sigma_prod = var_i * var_j;

    let mean_diff = features_i - features_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    // Mahalanobis term [F, F, C]
    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);

    // Log-determinant term [F, F, C]
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .clamp_min(eps)
        .log()
        .mul_scalar(0.25);

    // Sum over centroid dimension (C) to get [F, F]
    (mahalanobis + log_term).sum_dim(2).squeeze()
}

/// Euclidean L2 distance between two vectors
pub fn euclidean_distance<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = vec_i - vec_j;
    diff.powf_scalar(2.0).sum().sqrt()
}

/// Squared Euclidean distance (avoids sqrt for speed)
pub fn squared_euclidean_distance<B: Backend>(
    vec_i: Tensor<B, 1>,
    vec_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let diff = vec_i - vec_j;
    diff.powf_scalar(2.0).sum()
}

/// Euclidean distance for slices (CPU-friendly)
pub fn euclidean_distance_slice(vec_i: &[f32], vec_j: &[f32]) -> f32 {
    assert_eq!(vec_i.len(), vec_j.len());
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Squared Euclidean distance for slices
pub fn squared_euclidean_distance_slice(vec_i: &[f32], vec_j: &[f32]) -> f32 {
    assert_eq!(vec_i.len(), vec_j.len());
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// Cosine similarity: cos(θ) = (a·b) / (||a|| ||b||)
pub fn cosine_similarity<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let dot_product = (vec_i.clone() * vec_j.clone()).sum();
    let norm_i = vec_i.powf_scalar(2.0).sum().sqrt();
    let norm_j = vec_j.powf_scalar(2.0).sum().sqrt();

    dot_product / (norm_i * norm_j).clamp_min(1e-10)
}

/// Cosine distance: 1 - cos(θ)
pub fn cosine_distance<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let sim = cosine_similarity(vec_i, vec_j);
    Tensor::ones_like(&sim) - sim
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AutoBackend;
    type TestBackend = AutoBackend;

    #[test]
    fn test_bhattacharyya_identical_distributions() {
        let device = Default::default();

        let mean = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let var = Tensor::<TestBackend, 1>::from_floats([0.5, 0.5, 0.5], &device);

        let distance = bhattacharyya_diagonal(mean.clone(), var.clone(), mean, var);

        let dist_val: f32 = distance.into_scalar();
        assert!(
            dist_val < 1e-6,
            "Distance between identical distributions should be ~0, got {}",
            dist_val
        );
    }

    #[test]
    fn test_bhattacharyya_different_means() {
        let device = Default::default();

        let mean_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
        let mean_j = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
        let var = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);

        let distance = bhattacharyya_diagonal(mean_i, var.clone(), mean_j, var);

        let dist_val: f32 = distance.into_scalar();
        assert!(dist_val > 0.0, "Distance should be positive");
    }

    #[test]
    fn test_bhattacharyya_slice_vs_tensor() {
        crate::init();
        let device = Default::default();

        let mean_i_vec = vec![1.0, 2.0, 3.0];
        let mean_j_vec = vec![1.5, 2.5, 3.5];
        let var_i_vec = vec![0.5, 0.5, 0.5];
        let var_j_vec = vec![0.6, 0.6, 0.6];

        // Slice version
        let dist_slice =
            bhattacharyya_distance_diagonal(&mean_i_vec, &var_i_vec, &mean_j_vec, &var_j_vec);

        // Tensor version
        let mean_i = Tensor::<TestBackend, 1>::from_floats(mean_i_vec.as_slice(), &device);
        let mean_j = Tensor::<TestBackend, 1>::from_floats(mean_j_vec.as_slice(), &device);
        let var_i = Tensor::<TestBackend, 1>::from_floats(var_i_vec.as_slice(), &device);
        let var_j = Tensor::<TestBackend, 1>::from_floats(var_j_vec.as_slice(), &device);

        let dist_tensor: f32 = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j).into_scalar();

        assert!(
            (dist_slice - dist_tensor).abs() < 1e-5,
            "Slice and tensor versions should match: {} vs {}",
            dist_slice,
            dist_tensor
        );
    }

    #[test]
    fn test_bhattacharyya_affinity() {
        let device = Default::default();

        let mean_i = Tensor::<TestBackend, 1>::zeros([3], &device);
        let mean_j = Tensor::<TestBackend, 1>::zeros([3], &device);
        let var = Tensor::<TestBackend, 1>::ones([3], &device);

        let affinity = bhattacharyya_affinity(mean_i, var.clone(), mean_j, var);
        let aff_val: f32 = affinity.into_scalar();

        assert!(
            aff_val > 0.99,
            "Affinity between identical distributions should be ~1, got {}",
            aff_val
        );
    }

    #[test]
    fn test_bhattacharyya_pairwise() {
        let device = Default::default();

        // 4 features, 3 centroids
        let features = Tensor::<TestBackend, 2>::random(
            [4, 3],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let variances = Tensor::<TestBackend, 2>::ones([4, 3], &device).mul_scalar(0.5);

        let distances = bhattacharyya_pairwise(features, variances);

        assert_eq!(distances.dims(), [4, 4]);

        // Diagonal should be ~0 (self-distance)
        let diag_data = distances.to_data();
        let diag_vec: Vec<f32> = diag_data.to_vec().unwrap();
        for i in 0..4 {
            let self_dist = diag_vec[i * 4 + i];
            assert!(
                self_dist < 1e-5,
                "Self-distance should be ~0, got {}",
                self_dist
            );
        }
    }

    #[test]
    fn test_euclidean_distances() {
        let device = Default::default();

        let vec_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
        let vec_j = Tensor::<TestBackend, 1>::from_floats([3.0, 4.0], &device);

        let dist: f32 = euclidean_distance(vec_i, vec_j).into_scalar();
        assert!(
            (dist - 5.0).abs() < 1e-5,
            "3-4-5 triangle: expected 5.0, got {}",
            dist
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let device = Default::default();

        // Parallel vectors
        let vec_i = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
        let vec_j = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0], &device);

        let sim: f32 = cosine_similarity(vec_i, vec_j).into_scalar();
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Parallel vectors should have cos=1, got {}",
            sim
        );

        // Orthogonal vectors
        let vec_a = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
        let vec_b = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0], &device);

        let sim_ortho: f32 = cosine_similarity(vec_a, vec_b).into_scalar();
        assert!(
            sim_ortho.abs() < 1e-5,
            "Orthogonal vectors should have cos=0, got {}",
            sim_ortho
        );
    }

    #[test]
    fn test_numerical_stability() {
        let device = Default::default();

        // Very small variances (near-zero)
        let mean_i = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);
        let mean_j = Tensor::<TestBackend, 1>::from_floats([1.1, 2.1], &device);
        let var_i = Tensor::<TestBackend, 1>::from_floats([1e-12, 1e-12], &device);
        let var_j = Tensor::<TestBackend, 1>::from_floats([1e-12, 1e-12], &device);

        let distance = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j);
        let dist_val: f32 = distance.into_scalar();

        assert!(
            dist_val.is_finite(),
            "Distance should be finite even with tiny variances"
        );
    }
}
