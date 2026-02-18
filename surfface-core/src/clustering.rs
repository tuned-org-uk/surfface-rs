// surfface-core/src/clustering.rs
//! Clustering stage: Spots centroids from raw input data with optional JL projection
//!
//! This is Stage A of the Surfface pipeline [file:2]:
//! - Optional JL projection for F > 1000 [file:1]
//! - Incremental clustering with radius threshold
//! - Kalman variance tracking
//!
//! Based on ArrowSpace's start_clustering_dim_reduce algorithm [file:6]

use crate::centroid::CentroidState;
use crate::reduction::{ImplicitProjection, compute_jl_dimension};
use burn::prelude::*;
use rayon::prelude::*;

/// Configuration for the clustering stage
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Maximum number of centroids to create (C_max)
    pub max_clusters: usize,

    /// Squared L2 distance threshold for creating new centroids
    pub radius_threshold: f32,

    /// Random seed for deterministic clustering
    pub seed: Option<u64>,

    /// Enable JL projection for F > projection_threshold
    pub use_projection: bool,

    /// Dimension threshold to trigger projection
    pub projection_threshold: usize,

    /// JL epsilon parameter (controls distortion vs compression)
    pub jl_epsilon: f32,

    /// Minimum target dimension after projection
    pub min_projected_dim: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            max_clusters: 10_000,
            radius_threshold: 1.0, // Squared L2 threshold
            seed: Some(42),
            use_projection: true,
            projection_threshold: 1000, // Trigger JL for F > 1K [file:1]
            jl_epsilon: 0.3,            // 30% distortion tolerance
            min_projected_dim: 64,
        }
    }
}

impl ClusteringConfig {
    /// Create config for high-dimensional data (F=100K)
    pub fn high_dimensional() -> Self {
        Self {
            max_clusters: 10_000,
            radius_threshold: 1.5,
            seed: Some(42),
            use_projection: true,
            projection_threshold: 1000,
            jl_epsilon: 0.3,
            min_projected_dim: 128,
        }
    }
}

/// Output of the clustering stage
pub struct ClusteringOutput<B: Backend> {
    pub state: CentroidState<B>,
    pub assignments: Vec<Option<usize>>,
    pub num_items: usize,
    pub original_dim: usize,
    pub working_dim: usize,
    pub projection: Option<ImplicitProjection>,
}

/// Clustering stage executor
pub struct ClusteringStage {
    config: ClusteringConfig,
}

impl ClusteringStage {
    pub fn new(config: ClusteringConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ClusteringConfig::default())
    }

    /// Execute clustering from Vec<Vec<f32>> (ArrowSpace-style algorithm)
    pub fn execute_from_vec<B: Backend>(
        &self,
        rows: Vec<Vec<f32>>,
        device: &B::Device,
    ) -> ClusteringOutput<B> {
        let n_items = rows.len();
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);

        log::info!(
            "🎯 Clustering {} items (F={}) into max {} centroids",
            n_items,
            n_features,
            self.config.max_clusters
        );

        // STAGE 1: Early Dimensionality Reduction (if beneficial) [file:6]
        let (working_rows, working_dim, projection) = if self.config.use_projection
            && n_features > self.config.projection_threshold
        {
            log::info!("Applying JL projection to accelerate clustering");

            // Compute target dimension
            let jl_dim = compute_jl_dimension(n_items, n_features, self.config.jl_epsilon);
            let target_dim = jl_dim
                .min(n_features / 2)
                .max(self.config.min_projected_dim);

            log::info!(
                "Early projection: {} features → {} dimensions (ε={:.2})",
                n_features,
                target_dim,
                self.config.jl_epsilon
            );

            // Create projection matrix
            let proj = ImplicitProjection::new(n_features, target_dim, self.config.seed);

            // Project all rows in parallel (Rayon) [file:6]
            let projected: Vec<Vec<f32>> = rows.par_iter().map(|row| proj.project(row)).collect();

            let compression = n_features as f64 / target_dim as f64;
            log::info!(
                "JL projection complete: {:.1}x compression, {} MB → {} MB",
                compression,
                (n_items * n_features * 4) / (1024 * 1024),
                (n_items * target_dim * 4) / (1024 * 1024)
            );

            (projected, target_dim, Some(proj))
        } else {
            log::debug!(
                "Skipping projection (F={} < threshold={})",
                n_features,
                self.config.projection_threshold
            );
            (rows.clone(), n_features, None)
        };

        // STAGE 2: Incremental Clustering [file:6]
        log::info!(
            "Running incremental clustering: max_clusters={}, radius={:.6}",
            self.config.max_clusters,
            self.config.radius_threshold
        );

        let (centroids_vec, assignments, counts) =
            self.run_incremental_clustering(&working_rows, working_dim);

        let n_clusters = centroids_vec.len();
        log::info!(
            "✓ Clustering complete: {} centroids, {} items assigned ({:.2}% compression)",
            n_clusters,
            assignments.iter().filter(|x| x.is_some()).count(),
            (n_clusters as f64 / n_items as f64) * 100.0
        );

        // STAGE 3: Convert to CentroidState
        let state = self.build_centroid_state::<B>(centroids_vec, counts, device);

        ClusteringOutput {
            state,
            assignments,
            num_items: n_items,
            original_dim: n_features,
            working_dim,
            projection,
        }
    }

    /// Incremental clustering with radius threshold (ArrowSpace algorithm) [file:6]
    fn run_incremental_clustering(
        &self,
        rows: &[Vec<f32>],
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Option<usize>>, Vec<usize>) {
        let n = rows.len();
        let max_clusters = self.config.max_clusters;
        let radius_sq = self.config.radius_threshold;

        let mut centroids: Vec<Vec<f32>> = Vec::new();
        let mut assignments: Vec<Option<usize>> = vec![None; n];
        let mut counts: Vec<usize> = Vec::new();

        // Initialize with first point
        centroids.push(rows[0].clone());
        counts.push(1);
        assignments[0] = Some(0);

        // Process remaining points
        for i in 1..n {
            let point = &rows[i];

            // Find nearest centroid
            let mut min_dist_sq = f32::INFINITY;
            let mut nearest_idx = 0;

            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist_sq = self.squared_l2_distance(point, centroid);
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    nearest_idx = c_idx;
                }
            }

            // Assign or create new centroid
            if min_dist_sq < radius_sq {
                // Assign to existing centroid
                assignments[i] = Some(nearest_idx);
                counts[nearest_idx] += 1;

                // Update centroid (online mean)
                let count = counts[nearest_idx] as f32;
                let weight = 1.0 / count;
                for j in 0..dim {
                    centroids[nearest_idx][j] += weight * (point[j] - centroids[nearest_idx][j]);
                }
            } else if centroids.len() < max_clusters {
                // Create new centroid
                centroids.push(point.clone());
                counts.push(1);
                assignments[i] = Some(centroids.len() - 1);
            } else {
                // Force assignment to nearest (cluster budget exhausted)
                assignments[i] = Some(nearest_idx);
                counts[nearest_idx] += 1;
            }

            // Progress logging
            if (i + 1) % 100_000 == 0 {
                log::debug!(
                    "  Processed {}/{} items, {} centroids",
                    i + 1,
                    n,
                    centroids.len()
                );
            }
        }

        (centroids, assignments, counts)
    }

    /// Squared L2 distance
    fn squared_l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Build CentroidState from clustering results
    fn build_centroid_state<B: Backend>(
        &self,
        centroids_vec: Vec<Vec<f32>>,
        counts_vec: Vec<usize>,
        device: &B::Device,
    ) -> CentroidState<B> {
        let c = centroids_vec.len();
        let f = centroids_vec[0].len();

        // Convert centroids to tensor [C, F]
        let centroids_flat: Vec<f32> = centroids_vec.into_iter().flatten().collect();
        let centroids = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(centroids_flat, burn::tensor::Shape::new([c, f])),
            device,
        );

        // Convert counts to tensor [C]
        let counts = Tensor::<B, 1, Int>::from_ints(
            counts_vec
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        // Initialize with uniform variance (will be refined by Kalman in Stage B)
        CentroidState::from_clustering(centroids, counts, 0.1)
    }
}
