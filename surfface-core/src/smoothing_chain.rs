// surfface-core/src/kalman.rs
//! Kalman smoothing stage: Regularize centroids along MST order
//!
//! This is Stage B2 of the Surfface pipeline:
//! - Takes MST 1D ordering from Stage B1
//! - Applies Rauch-Tung-Striebel (RTS) smoother along centroid sequence
//! - Produces smoothed centroid means and variances
//! - Reduces noise while preserving manifold structure
//!
//! Mathematical framework:
//! - State space model: x_t = F x_{t-1} + w_t (process model)
//! - Observation model: y_t = H x_t + v_t (observation = raw centroids)
//! - Forward pass: Kalman filter
//! - Backward pass: RTS smoothing equations
//!
//! References:
//! - Rauch, Tung, Striebel (1965): "Maximum likelihood estimates of linear dynamic systems"
//! - Särkkä (2013): "Bayesian Filtering and Smoothing"

use crate::centroid::CentroidState;
use crate::mst::MSTOutput;
use burn::prelude::*;

/// Configuration for Kalman smoothing
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// Process noise covariance Q (controls smoothness)
    /// Higher Q → more responsive to observations
    /// Lower Q → smoother trajectory
    pub process_noise: f32,

    /// Observation noise covariance R (controls trust in observations)
    /// Higher R → trust predictions more
    /// Lower R → trust observations more
    pub observation_noise: f32,

    /// State transition model type
    pub transition_model: TransitionModel,

    /// Minimum variance floor (numerical stability)
    pub variance_floor: f32,

    /// Maximum variance ceiling (prevent explosion)
    pub variance_ceiling: f32,
}

/// State transition model for Kalman filter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionModel {
    /// Identity: x_t = x_{t-1} + w_t (random walk)
    /// Simple, no momentum, suitable for stationary features
    Identity,

    /// Damped: x_t = α x_{t-1} + w_t (exponential smoothing)
    /// α ∈ (0, 1) controls damping strength
    Damped(f32),

    /// Trunk-aware: Lower process noise along trunk edges
    /// Uses MST trunk annotation to reduce smoothing on main path
    TrunkAware { trunk_factor: f32 },
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.01,    // Low process noise (smooth)
            observation_noise: 0.1, // Higher observation noise (trust prior)
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }
}

impl KalmanConfig {
    /// Conservative smoothing (trust observations more)
    pub fn conservative() -> Self {
        Self {
            process_noise: 0.1,
            observation_noise: 0.01,
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }

    /// Aggressive smoothing (smooth heavily)
    pub fn aggressive() -> Self {
        Self {
            process_noise: 0.001,
            observation_noise: 1.0,
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }

    /// Trunk-aware smoothing (preserve trunk structure)
    pub fn trunk_aware(trunk_factor: f32) -> Self {
        Self {
            process_noise: 0.01,
            observation_noise: 0.1,
            transition_model: TransitionModel::TrunkAware { trunk_factor },
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }
}

/// Output of Kalman smoothing stage
pub struct KalmanOutput<B: Backend> {
    /// Smoothed centroid means [C, F]
    pub smoothed_means: Tensor<B, 2>,

    /// Smoothed centroid variances [C, F]
    pub smoothed_variances: Tensor<B, 2>,

    /// Centroid counts (preserved from input state) [C]
    pub counts: Tensor<B, 1, Int>,

    /// Filtered means (from forward pass, for diagnostics) [C, F]
    pub filtered_means: Tensor<B, 2>,

    /// Filtered variances (from forward pass) [C, F]
    pub filtered_variances: Tensor<B, 2>,

    /// Smoothing gain per step (diagnostics)
    pub smoothing_gains: Vec<f32>,

    /// Mean variance reduction (smoothed vs raw)
    pub variance_reduction: f32,
}

impl<B: Backend> KalmanOutput<B> {
    pub fn summary(&self) -> String {
        format!(
            "Kalman: variance_reduction={:.2}%, gains_mean={:.4}",
            self.variance_reduction * 100.0,
            self.smoothing_gains.iter().sum::<f32>() / self.smoothing_gains.len() as f32
        )
    }

    /// Create smoothed CentroidState
    pub fn to_centroid_state(&self) -> CentroidState<B> {
        CentroidState {
            means: self.smoothed_means.clone(),
            variances: self.smoothed_variances.clone(),
            counts: self.counts.clone(), // ← Preserve counts
        }
    }
}

/// Kalman smoothing stage executor
pub struct KalmanStage {
    config: KalmanConfig,
}

impl KalmanStage {
    pub fn new(config: KalmanConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(KalmanConfig::default())
    }

    /// Execute Kalman smoothing
    pub fn execute<B: Backend>(
        &self,
        state: &CentroidState<B>,
        mst_output: &MSTOutput,
    ) -> KalmanOutput<B> {
        let device = state.means.device();
        let c = state.num_centroids();
        let f = state.feature_dim();

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE B2: KALMAN SMOOTHING                           ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!("📊 Smoothing {} centroids (F={}) along MST order", c, f);
        log::info!(
            "  • Process noise Q={:.4}, Observation noise R={:.4}",
            self.config.process_noise,
            self.config.observation_noise
        );

        // Extract data to CPU for sequential processing
        let means_data = state.means.to_data();
        let means_vec: Vec<f32> = means_data.to_vec().unwrap();

        let variances_data = state.variances.to_data();
        let variances_vec: Vec<f32> = variances_data.to_vec().unwrap();

        let order = &mst_output.centroid_order;

        // FORWARD PASS: Kalman filter
        log::debug!("Step 1/2: Forward pass (Kalman filter)...");
        let (filtered_means, filtered_vars, predicted_vars) =
            self.forward_pass(&means_vec, &variances_vec, order, f);

        // BACKWARD PASS: RTS smoother
        log::debug!("Step 2/2: Backward pass (RTS smoothing)...");
        let (smoothed_means, smoothed_vars, gains) =
            self.backward_pass(&filtered_means, &filtered_vars, &predicted_vars, order, f);

        // Compute variance reduction
        let raw_var_mean: f32 = variances_vec.iter().sum::<f32>() / variances_vec.len() as f32;
        let smoothed_var_mean: f32 = smoothed_vars.iter().sum::<f32>() / smoothed_vars.len() as f32;
        let variance_reduction = (raw_var_mean - smoothed_var_mean) / raw_var_mean;

        log::info!("  ✓ Smoothing complete");
        log::info!(
            "    • Variance reduction: {:.2}%",
            variance_reduction * 100.0
        );
        log::info!(
            "    • Mean smoothing gain: {:.4}",
            gains.iter().sum::<f32>() / gains.len() as f32
        );

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  KALMAN SMOOTHING COMPLETE                            ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");

        // Convert back to tensors with explicit type annotations
        let smoothed_means_tensor =
            Tensor::<B, 2>::from_floats(smoothed_means.as_slice(), &device).reshape([c, f]);

        let smoothed_vars_tensor =
            Tensor::<B, 2>::from_floats(smoothed_vars.as_slice(), &device).reshape([c, f]);

        let filtered_means_tensor =
            Tensor::<B, 2>::from_floats(filtered_means.as_slice(), &device).reshape([c, f]);

        let filtered_vars_tensor =
            Tensor::<B, 2>::from_floats(filtered_vars.as_slice(), &device).reshape([c, f]);

        KalmanOutput {
            smoothed_means: smoothed_means_tensor,
            smoothed_variances: smoothed_vars_tensor,
            counts: state.counts.clone(), // ← Preserve counts from input
            filtered_means: filtered_means_tensor,
            filtered_variances: filtered_vars_tensor,
            smoothing_gains: gains,
            variance_reduction,
        }
    }

    /// Forward pass: Kalman filter along MST order
    /// Returns: (filtered_means, filtered_variances, predicted_variances)
    fn forward_pass(
        &self,
        means: &[f32],
        variances: &[f32],
        order: &[usize],
        f: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = order.len();
        let mut filtered_means = vec![0.0; c * f];
        let mut filtered_vars = vec![0.0; c * f];
        let mut predicted_vars = vec![0.0; c * f];

        // Initialize with first centroid (in MST order)
        let first_idx = order[0];
        for feat in 0..f {
            filtered_means[feat] = means[first_idx * f + feat];
            filtered_vars[feat] = variances[first_idx * f + feat]
                .clamp(self.config.variance_floor, self.config.variance_ceiling);
        }

        // Kalman filter: process observations sequentially along MST order
        for t in 1..c {
            let curr_idx = order[t];
            let prev_t = t - 1;

            for feat in 0..f {
                // PREDICTION STEP
                // x_{t|t-1} = F * x_{t-1|t-1}
                let x_pred = match self.config.transition_model {
                    TransitionModel::Identity => filtered_means[prev_t * f + feat],
                    TransitionModel::Damped(alpha) => alpha * filtered_means[prev_t * f + feat],
                    TransitionModel::TrunkAware { .. } => {
                        // TODO: Use trunk annotation
                        filtered_means[prev_t * f + feat]
                    }
                };

                // P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
                let p_pred = filtered_vars[prev_t * f + feat] + self.config.process_noise;
                predicted_vars[t * f + feat] = p_pred;

                // UPDATE STEP
                // Observation: y_t
                let y_obs = means[curr_idx * f + feat];
                let r_obs = variances[curr_idx * f + feat]
                    .clamp(self.config.variance_floor, self.config.variance_ceiling)
                    + self.config.observation_noise;

                // Innovation covariance: S = H * P_{t|t-1} * H^T + R
                let s = p_pred + r_obs;

                // Kalman gain: K = P_{t|t-1} * H^T * S^{-1}
                let k = p_pred / s;

                // Updated state: x_{t|t} = x_{t|t-1} + K * (y_t - H * x_{t|t-1})
                let x_filt = x_pred + k * (y_obs - x_pred);
                filtered_means[t * f + feat] = x_filt;

                // Updated covariance: P_{t|t} = (I - K * H) * P_{t|t-1}
                let p_filt = (1.0 - k) * p_pred;
                filtered_vars[t * f + feat] =
                    p_filt.clamp(self.config.variance_floor, self.config.variance_ceiling);
            }
        }

        (filtered_means, filtered_vars, predicted_vars)
    }

    /// Backward pass: RTS smoother equations
    /// Returns: (smoothed_means, smoothed_variances, smoothing_gains)
    fn backward_pass(
        &self,
        filtered_means: &[f32],
        filtered_vars: &[f32],
        predicted_vars: &[f32],
        order: &[usize],
        f: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = order.len();
        let mut smoothed_means = filtered_means.to_vec();
        let mut smoothed_vars = filtered_vars.to_vec();
        let mut gains = Vec::new();

        // Backward pass: t = T-1, T-2, ..., 0
        for t in (0..c - 1).rev() {
            let next_t = t + 1;
            let mut gain_sum = 0.0;

            for feat in 0..f {
                // RTS smoothing gain: C_t = P_{t|t} * F^T * P_{t+1|t}^{-1}
                let p_filt = filtered_vars[t * f + feat];
                let p_pred_next = predicted_vars[next_t * f + feat];

                let c_gain = if p_pred_next > self.config.variance_floor {
                    p_filt / p_pred_next
                } else {
                    0.0
                };

                gain_sum += c_gain;

                // Smoothed mean: x_{t|T} = x_{t|t} + C_t * (x_{t+1|T} - x_{t+1|t})
                let x_filt = filtered_means[t * f + feat];
                let x_smooth_next = smoothed_means[next_t * f + feat];
                let x_pred_next = match self.config.transition_model {
                    TransitionModel::Identity => filtered_means[t * f + feat],
                    TransitionModel::Damped(alpha) => alpha * filtered_means[t * f + feat],
                    TransitionModel::TrunkAware { .. } => filtered_means[t * f + feat],
                };

                smoothed_means[t * f + feat] = x_filt + c_gain * (x_smooth_next - x_pred_next);

                // Smoothed variance: P_{t|T} = P_{t|t} + C_t * (P_{t+1|T} - P_{t+1|t}) * C_t^T
                let p_smooth_next = smoothed_vars[next_t * f + feat];
                smoothed_vars[t * f + feat] =
                    p_filt + c_gain * (p_smooth_next - p_pred_next) * c_gain;
                smoothed_vars[t * f + feat] = smoothed_vars[t * f + feat]
                    .clamp(self.config.variance_floor, self.config.variance_ceiling);
            }

            gains.push(gain_sum / f as f32);
        }

        (smoothed_means, smoothed_vars, gains)
    }
}
