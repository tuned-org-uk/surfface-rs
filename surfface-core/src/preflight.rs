// surfface-core/src/preflight.rs
//! Preflight stage: Prepare centroids for stable graph construction
//!
//! This is Stage B0 of the Surfface pipeline [file:2]:
//! - Conditional magnitude normalization (if variance > 10×)
//! - Zero-variance feature detection
//! - Variance regularization for Bhattacharyya stability
//!
//! Critical for preventing eigenvalue computation failures [file:1]

use crate::centroid::CentroidState;
use burn::prelude::*;

/// Configuration for the preflight stage
#[derive(Debug, Clone)]
pub struct PreflightConfig {
    /// Magnitude variance threshold to trigger normalization (default: 10.0)
    /// If max_norm / min_norm > threshold, normalize centroids
    pub magnitude_threshold: f32,

    /// Minimum variance for features (default: 1e-6)
    /// Features with variance below this are flagged
    pub min_variance: f32,

    /// Variance regularization epsilon (default: 1e-4) [file:3]
    pub variance_epsilon: f32,

    /// Variance clamp range (default: [1e-4, 100.0])
    pub variance_min: f32,
    pub variance_max: f32,

    /// Remove zero-variance features instead of regularizing
    pub remove_zero_variance: bool,
}

impl Default for PreflightConfig {
    fn default() -> Self {
        Self {
            magnitude_threshold: 10.0,
            min_variance: 1e-6,
            variance_epsilon: 1e-4,
            variance_min: 1e-4,
            variance_max: 100.0,
            remove_zero_variance: false,
        }
    }
}

impl PreflightConfig {
    /// Conservative config for high-dimensional data
    pub fn conservative() -> Self {
        Self {
            magnitude_threshold: 5.0, // More aggressive normalization
            min_variance: 1e-5,
            variance_epsilon: 1e-3,
            variance_min: 1e-3,
            variance_max: 50.0,
            remove_zero_variance: false,
        }
    }

    /// Strict config that removes problematic features
    pub fn strict() -> Self {
        Self {
            magnitude_threshold: 10.0,
            min_variance: 1e-4,
            variance_epsilon: 1e-4,
            variance_min: 1e-4,
            variance_max: 100.0,
            remove_zero_variance: true,
        }
    }
}

/// Statistics about magnitude distribution
#[derive(Debug, Clone)]
pub struct MagnitudeStats {
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    pub variance_ratio: f32, // max/min
}

impl MagnitudeStats {
    /// Check if normalization is needed based on variance ratio
    pub fn needs_normalization(&self, threshold: f32) -> bool {
        self.variance_ratio > threshold
    }

    /// Pretty-print statistics
    pub fn summary(&self) -> String {
        format!(
            "min={:.4}, max={:.4}, mean={:.4}, ratio={:.2}×",
            self.min_norm, self.max_norm, self.mean_norm, self.variance_ratio
        )
    }
}

/// Output of the preflight stage
pub struct PreflightOutput<B: Backend> {
    /// Centroids prepared for graph construction [C, F]
    /// (normalized if magnitude variance exceeded threshold)
    pub normalized_centroids: Tensor<B, 2>,

    /// Original centroids for interpretation [C, F]
    pub original_centroids: Tensor<B, 2>,

    /// Regularized variances for Bhattacharyya [C, F]
    pub variances: Tensor<B, 2>,

    /// Was normalization applied?
    pub was_normalized: bool,

    /// Magnitude statistics
    pub magnitude_stats: MagnitudeStats,

    /// Indices of zero-variance features (if detected)
    pub zero_variance_features: Vec<usize>,

    /// Updated CentroidState
    pub state: CentroidState<B>,
}

impl<B: Backend> PreflightOutput<B> {
    /// Get a summary of the preflight results
    pub fn summary(&self) -> String {
        format!(
            "Preflight: normalized={}, zero_var={}, magnitude_ratio={:.2}×",
            self.was_normalized,
            self.zero_variance_features.len(),
            self.magnitude_stats.variance_ratio
        )
    }
}

/// Preflight stage executor
pub struct PreflightStage {
    config: PreflightConfig,
}

impl PreflightStage {
    pub fn new(config: PreflightConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(PreflightConfig::default())
    }

    /// Execute preflight checks and transformations
    pub fn execute<B: Backend>(&self, mut state: CentroidState<B>) -> PreflightOutput<B> {
        let [c, f] = state.means.dims();

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE B0: PREFLIGHT                                  ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!("🔍 Analyzing {} centroids × {} features", c, f);

        // STEP 1: Analyze magnitude distribution
        log::debug!("Step 1/4: Computing magnitude statistics...");
        let magnitude_stats = self.compute_magnitude_stats(&state.means);

        log::info!("  📊 Magnitude distribution: {}", magnitude_stats.summary());

        if magnitude_stats.variance_ratio > self.config.magnitude_threshold {
            log::warn!(
                "  ⚠️  High magnitude variance detected: {:.2}× > threshold {:.2}×",
                magnitude_stats.variance_ratio,
                self.config.magnitude_threshold
            );
        } else {
            log::info!(
                "  ✓ Magnitude variance within bounds: {:.2}× ≤ {:.2}×",
                magnitude_stats.variance_ratio,
                self.config.magnitude_threshold
            );
        }

        // STEP 2: Conditional normalization
        log::debug!("Step 2/4: Checking normalization requirements...");
        let (normalized_centroids, was_normalized) =
            if magnitude_stats.needs_normalization(self.config.magnitude_threshold) {
                log::warn!("  🔧 Applying L2 normalization to prevent eigenvalue collapse");
                let normalized = self.normalize_l2(&state.means);

                // Verify normalization
                let post_stats = self.compute_magnitude_stats(&normalized);
                log::info!(
                    "  ✓ Normalization complete: ratio {:.2}× → {:.2}×",
                    magnitude_stats.variance_ratio,
                    post_stats.variance_ratio
                );

                (normalized, true)
            } else {
                log::info!("  ✓ Skipping normalization (not needed)");
                (state.means.clone(), false)
            };

        // STEP 3: Detect zero-variance features
        log::debug!("Step 3/4: Detecting zero-variance features...");
        let zero_variance_features = self.detect_zero_variance(&state.variances);

        if !zero_variance_features.is_empty() {
            log::warn!(
                "  ⚠️  Found {} zero-variance features (var < {:.2e})",
                zero_variance_features.len(),
                self.config.min_variance
            );

            if zero_variance_features.len() <= 20 {
                log::debug!(
                    "  Zero-variance feature indices: {:?}",
                    zero_variance_features
                );
            } else {
                log::debug!(
                    "  Zero-variance feature indices: {:?} ... ({} more)",
                    &zero_variance_features[..20],
                    zero_variance_features.len() - 20
                );
            }

            if self.config.remove_zero_variance {
                log::warn!("  🗑️  Removal requested but not implemented (TODO)");
            } else {
                log::info!("  ℹ️  Will regularize in next step");
            }
        } else {
            log::info!("  ✓ No zero-variance features detected");
        }

        // STEP 4: Regularize variances for Bhattacharyya stability [file:3]
        log::debug!("Step 4/4: Regularizing variances...");

        // Compute variance stats before regularization
        let var_data_before = state.variances.clone().to_data();
        let var_vec_before: Vec<f32> = var_data_before.to_vec().unwrap();
        let min_var_before = var_vec_before.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_var_before = var_vec_before
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        state.regularize_variances(
            self.config.variance_epsilon,
            self.config.variance_min,
            self.config.variance_max,
        );

        // Compute variance stats after regularization
        let var_data_after = state.variances.clone().to_data();
        let var_vec_after: Vec<f32> = var_data_after.to_vec().unwrap();
        let min_var_after = var_vec_after.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_var_after = var_vec_after
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        log::info!(
            "  ✓ Variances regularized: [{:.2e}, {:.2e}] → [{:.2e}, {:.2e}]",
            min_var_before,
            max_var_before,
            min_var_after,
            max_var_after
        );

        // Final summary
        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  PREFLIGHT COMPLETE                                   ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!(
            "  • Normalized: {}",
            if was_normalized { "YES" } else { "NO" }
        );
        log::info!(
            "  • Zero-variance features: {}",
            zero_variance_features.len()
        );
        log::info!(
            "  • Magnitude ratio: {:.2}×",
            magnitude_stats.variance_ratio
        );
        log::info!(
            "  • Variance range: [{:.2e}, {:.2e}]",
            min_var_after,
            max_var_after
        );

        PreflightOutput {
            normalized_centroids: normalized_centroids.clone(),
            original_centroids: state.means.clone(),
            variances: state.variances.clone(),
            was_normalized,
            magnitude_stats,
            zero_variance_features,
            state,
        }
    }

    /// Compute magnitude statistics for centroids
    pub(crate) fn compute_magnitude_stats<B: Backend>(
        &self,
        centroids: &Tensor<B, 2>,
    ) -> MagnitudeStats {
        // Compute L2 norms for each centroid
        let norms: Tensor<B, 1> = centroids
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .squeeze();

        let norms_data = norms.to_data();
        let norms_vec: Vec<f32> = norms_data.to_vec().unwrap();

        let min_norm = norms_vec
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .max(1e-10); // Prevent zero
        let max_norm = norms_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_norm = norms_vec.iter().sum::<f32>() / norms_vec.len() as f32;
        let variance_ratio = max_norm / min_norm;

        MagnitudeStats {
            min_norm,
            max_norm,
            mean_norm,
            variance_ratio,
        }
    }

    /// L2 normalization: scale each centroid to unit norm
    pub(crate) fn normalize_l2<B: Backend>(&self, centroids: &Tensor<B, 2>) -> Tensor<B, 2> {
        // Compute norms [C, 1]
        let norms = centroids
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-10); // Prevent division by zero

        // Broadcast and divide
        centroids.clone() / norms
    }

    /// Detect features with near-zero variance across centroids
    fn detect_zero_variance<B: Backend>(&self, variances: &Tensor<B, 2>) -> Vec<usize> {
        let [_c, _f] = variances.dims();

        // Compute mean variance per feature (average across centroids)
        let feature_variances: Tensor<B, 1> = variances.clone().mean_dim(0).squeeze();

        let var_data = feature_variances.to_data();
        let var_vec: Vec<f32> = var_data.to_vec().unwrap();

        // Find features below threshold
        var_vec
            .iter()
            .enumerate()
            .filter(|(_, v)| **v < self.config.min_variance)
            .map(|(i, _)| i)
            .collect()
    }
}
