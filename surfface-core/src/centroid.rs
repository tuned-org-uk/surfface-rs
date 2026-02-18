//! Basic step of the pipeline: Spot centroids using Kalman Clustering

use burn::prelude::*;
use kalman_clustering::KalmanClusterer;

/// Batch centroid state for Surfface pipeline [file:2]
/// Stores aggregated results from kalman_clustering::Clusterer
pub struct CentroidState<B: Backend> {
    pub means: Tensor<B, 2>,       // [C, F]
    pub variances: Tensor<B, 2>,   // [C, F] - Diagonal covariance
    pub counts: Tensor<B, 1, Int>, // [C]
}

impl<B: Backend> CentroidState<B> {
    /// Initialize from kalman_clustering::Clusterer results
    pub fn from_clusterer(
        centroids: Tensor<B, 2>,         // Output from clusterer.centroids()
        variances: Tensor<B, 2>,         // Output from clusterer.variances()
        assignments: &Tensor<B, 1, Int>, // Output from clusterer.assignments()
    ) -> Self {
        let [c, _] = centroids.dims();

        // Compute counts per centroid
        let mut counts_vec = vec![0i32; c];
        let assignments: Vec<i32> = assignments.to_data().convert::<i32>().to_vec().unwrap();

        for &c_id in &assignments {
            if c_id >= 0 && (c_id as usize) < c {
                counts_vec[c_id as usize] += 1;
            }
        }

        let counts = Tensor::<B, 1, Int>::from_ints(counts_vec.as_slice(), &centroids.device());

        Self {
            means: centroids,
            variances,
            counts,
        }
    }

    /// Initialize from raw tensors (for compatibility/testing)
    pub fn from_clustering(
        centroids: Tensor<B, 2>,
        counts: Tensor<B, 1, Int>,
        initial_variance: f32,
    ) -> Self {
        let [c, f] = centroids.dims();
        let variances = Tensor::ones([c, f], &centroids.device()).mul_scalar(initial_variance);

        Self {
            means: centroids,
            variances,
            counts,
        }
    }

    /// Create from KalmanClusterer (convenience wrapper)
    pub fn from_kalman_clusterer(clusterer: &KalmanClusterer<B>, device: &B::Device) -> Self {
        // Export centroids as Vec<Vec<f32>>
        let centroids_vec = clusterer.export_centroids();
        let c = centroids_vec.len();

        if c == 0 {
            panic!("Cannot create CentroidState from empty clusterer");
        }

        let f = centroids_vec[0].len();

        // Flatten to create tensor [C, F]
        let centroids_flat: Vec<f32> = centroids_vec.into_iter().flatten().collect();

        // Create tensor with explicit shape [C, F]
        let centroids = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(centroids_flat, burn::tensor::Shape::new([c, f])),
            device,
        );

        // Get assignments
        let assignments_vec = clusterer.assignments.clone();
        let assignments = Tensor::<B, 1, Int>::from_ints(
            assignments_vec
                .iter()
                .map(|&x| x.unwrap() as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        // Extract variances from centroids field
        let variances_data: Vec<f32> = clusterer
            .centroids
            .iter()
            .flat_map(|cent| cent.variance_to_vec())
            .collect();

        let variances = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(variances_data, burn::tensor::Shape::new([c, f])),
            device,
        );

        Self::from_clusterer(centroids, variances, &assignments)
    }

    /// Thickness proxy for MST weighting [file:4]
    /// Returns mean variance per centroid (trace(P)/F)
    pub fn get_thickness(&self) -> Tensor<B, 1> {
        self.variances.clone().mean_dim(1).squeeze()
    }

    /// Transpose to feature-space for Laplacian [file:2]
    pub fn to_feature_nodes(&self) -> Tensor<B, 2> {
        self.means.clone().transpose()
    }

    /// Get feature variances for Bhattacharyya distance [file:3]
    pub fn get_feature_variances(&self) -> Tensor<B, 2> {
        self.variances.clone().transpose()
    }

    /// Regularize variances to prevent numerical issues [file:3]
    pub fn regularize_variances(&mut self, eps: f32, min_var: f32, max_var: f32) {
        self.variances = (self.variances.clone() + eps).clamp(min_var, max_var);
    }

    pub fn num_centroids(&self) -> usize {
        self.means.dims()[0]
    }

    pub fn feature_dim(&self) -> usize {
        self.means.dims()[1]
    }
}
