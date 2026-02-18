//! Stage A': Johnson-Lindenstrauss dimension reduction (optional preprocessing).
//!
//! Pipeline position: Raw [N, F] → Projected [N, R] where R ≪ F.
//!
//! This stage is optional and triggered when F is very large (e.g., F > 1000).
//! It uses a seed-based implicit random projection matrix to avoid storing
//! the full [R, F] dense matrix in memory.
//!
//! Key design decisions:
//! - **Implicit projection**: The Gaussian random matrix is regenerated on-the-fly
//!   from a fixed seed, ensuring reproducibility without memory overhead.
//! - **Deterministic**: Same seed → same projection → reproducible pipelines.
//! - **JL theorem**: For ε-distortion, target dimension R ≥ 8 ln(N) / ε².
//!   Reference: Achlioptas (2003), "Database-friendly random projections".
//!
//! ArrowSpace compatibility:
//! - ArrowSpace stores `projectionmatrix: Option<ImplicitProjection>` in the
//!   ArrowSpace struct and applies it to query vectors at search time.
//! - Surfface promotes this to a first-class Stage A' with explicit artifacts.

use burn::prelude::*;
use log::{debug, info};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for JL projection stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionConfig {
    /// Distortion parameter ε. Smaller ε → higher target dimension.
    /// Typical range: 0.1–0.5.
    pub epsilon: f32,

    /// Minimum original dimension to trigger projection.
    /// If F < threshold, skip projection entirely.
    pub min_dim_threshold: usize,

    /// Fixed seed for reproducible projections.
    pub seed: Option<u64>,

    /// If true, use GPU-accelerated projection (Burn tensors).
    /// If false, use CPU-only row-by-row projection.
    pub use_gpu: bool,

    /// Maximum target dimension (safety cap).
    /// Prevents pathological JL bounds from creating huge projections.
    pub max_target_dim: usize,
}

impl Default for ReductionConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.3,
            min_dim_threshold: 512,
            seed: Some(42),
            use_gpu: false,
            max_target_dim: 2048,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Implicit Projection Matrix
// ─────────────────────────────────────────────────────────────────────────────

/// JL projection matrix (implicit, seed-based).
///
/// Does not store the [R, F] matrix explicitly. Instead, it regenerates
/// the Gaussian random values on-the-fly using a fixed seed.
///
/// Memory cost: O(1) constants only.
/// Compute cost: O(N·F·R) for projecting N rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplicitProjection {
    pub original_dim: usize,
    pub target_dim: usize,
    pub seed: u64,
}

impl ImplicitProjection {
    /// Create a new implicit projection.
    ///
    /// # Arguments
    /// * `original_dim` - F, the original feature dimension
    /// * `target_dim` - R, the reduced dimension (must be ≤ F)
    /// * `seed` - RNG seed for reproducibility
    pub fn new(original_dim: usize, target_dim: usize, seed: Option<u64>) -> Self {
        assert!(
            target_dim <= original_dim,
            "Target dimension {} cannot exceed original dimension {}",
            target_dim,
            original_dim
        );

        let seed = seed.unwrap_or(42);
        debug!(
            "Creating ImplicitProjection: {} → {} (seed={})",
            original_dim, target_dim, seed
        );

        Self {
            original_dim,
            target_dim,
            seed,
        }
    }

    /// Project a single row: x [F] → y [R].
    ///
    /// This is the CPU-only, row-by-row projection used by ArrowSpace.
    /// For batch projection, use `project_batch_cpu` or `project_batch_gpu`.
    ///
    /// # Math
    /// y = (1/√R) · Φ · x, where Φ ~ N(0, 1)^{R×F}
    ///
    /// # Performance
    /// O(F·R) per row. For N rows, total O(N·F·R).
    pub fn project(&self, row: &[f32]) -> Vec<f32> {
        assert_eq!(
            row.len(),
            self.original_dim,
            "Row length {} does not match original dimension {}",
            row.len(),
            self.original_dim
        );

        let scale = 1.0 / (self.target_dim as f32).sqrt();
        let mut result = vec![0.0f32; self.target_dim];

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        for j in 0..self.target_dim {
            let mut sum = 0.0f32;
            for i in 0..self.original_dim {
                let rand_val: f32 = rng.sample(StandardNormal);
                sum += row[i] * rand_val;
            }
            result[j] = sum * scale;
        }

        result
    }

    /// Project a batch of rows in parallel (CPU-only, Rayon).
    ///
    /// # Arguments
    /// * `rows` - Vec of [F]-dimensional rows
    ///
    /// # Returns
    /// Vec of [R]-dimensional projected rows
    ///
    /// # Performance
    /// O(N·F·R), parallelized over N using Rayon.
    pub fn project_batch_cpu(&self, rows: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use rayon::prelude::*;

        rows.par_iter().map(|row| self.project(row)).collect()
    }

    /// Project a batch of rows using GPU (Burn tensors).
    ///
    /// # Arguments
    /// * `data_flat` - Flattened [N·F] row-major data
    /// * `n` - Number of rows
    ///
    /// # Returns
    /// Flattened [N·R] row-major projected data
    ///
    /// # GPU Strategy
    /// 1. Upload data [N, F] to device
    /// 2. Generate Φ [R, F] on device (seed-based)
    /// 3. Compute Y = X · Φᵀ (batched matmul)
    /// 4. Download [N, R] result
    ///
    /// This is faster than CPU for large N or when data is already on GPU.
    pub fn project_batch_gpu<B: Backend>(
        &self,
        data_flat: &[f32],
        n: usize,
        device: &B::Device,
    ) -> Vec<f32> {
        let f = self.original_dim;
        let r = self.target_dim;

        debug!("GPU projection: [{}×{}] → [{}×{}]", n, f, n, r);

        // 1. Upload data [N, F]
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(data_flat.to_vec(), Shape::new([n, f])),
            device,
        );

        // 2. Generate random projection matrix Φ [R, F]
        let phi_flat = self.generate_phi_flat();
        let phi = Tensor::<B, 2>::from_data(TensorData::new(phi_flat, Shape::new([r, f])), device);

        // 3. Y = X · Φᵀ = [N, F] · [F, R] = [N, R]
        let y = x.matmul(phi.transpose());

        // 4. Download to CPU
        y.to_data().to_vec().unwrap()
    }

    /// Generate the full [R×F] projection matrix as a flat vector.
    ///
    /// This is used internally by `project_batch_gpu`.
    /// For large F, this can be memory-intensive (R·F·4 bytes).
    fn generate_phi_flat(&self) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let scale = 1.0 / (self.target_dim as f32).sqrt();

        (0..self.target_dim * self.original_dim)
            .map(|_| rng.sample::<f32, _>(StandardNormal) * scale)
            .collect()
    }

    /// Apply projection to query vectors at search time (ArrowSpace compat).
    ///
    /// This is the entry point used by the search stage to ensure
    /// query vectors match the index dimensionality.
    pub fn project_query(&self, query: &[f32]) -> Vec<f32> {
        self.project(query)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JL Dimension Calculation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the target dimension R for JL projection.
///
/// Uses the JL lemma bound: R ≥ 8 ln(N) / ε² to preserve pairwise
/// distances with probability 1 - 1/N.
///
/// # Arguments
/// * `n_points` - Number of data points N
/// * `original_dim` - Original dimension F
/// * `epsilon` - Distortion parameter (typical: 0.1–0.5)
///
/// # Returns
/// Target dimension R, clamped to [32, F].
///
/// # Example
/// ```
/// use surfface_core::reduction::compute_jl_dimension;
/// let r = compute_jl_dimension(10_000, 2048, 0.3);
/// assert!(r >= 32 && r <= 2048);
/// ```
pub fn compute_jl_dimension(n_points: usize, original_dim: usize, epsilon: f32) -> usize {
    if original_dim < 32 {
        return original_dim;
    }

    let log_n = (n_points as f32).ln();
    let eps_sq = epsilon.powi(2);
    let jl_bound = (8.0 * log_n / eps_sq).ceil() as usize;

    let target = jl_bound.clamp(32, original_dim);

    debug!(
        "JL dimension: N={}, F={}, ε={:.2} → R={} (bound={})",
        n_points, original_dim, epsilon, target, jl_bound
    );

    target
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage A' Output
// ─────────────────────────────────────────────────────────────────────────────

/// Output of Stage A': dimension reduction.
#[derive(Debug, Clone)]
pub struct ReductionOutput {
    /// Projected data [N×R] flattened row-major
    pub projected_data: Vec<f32>,

    /// Number of rows N
    pub n_items: usize,

    /// Reduced dimension R
    pub reduced_dim: usize,

    /// Original dimension F
    pub original_dim: usize,

    /// Projection matrix (for applying to query vectors)
    pub projection: ImplicitProjection,

    /// Compression ratio: F / R
    pub compression_ratio: f32,
}

impl ReductionOutput {
    pub fn summary(&self) -> String {
        format!(
            "ReductionOutput: N={}, F={} → R={}, compression={:.2}x",
            self.n_items, self.original_dim, self.reduced_dim, self.compression_ratio
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage A' Executor
// ─────────────────────────────────────────────────────────────────────────────

/// Stage A' executor: JL dimension reduction.
pub struct ReductionStage {
    pub config: ReductionConfig,
}

impl ReductionStage {
    pub fn new(config: ReductionConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ReductionConfig::default())
    }

    /// Execute Stage A'.
    ///
    /// # Arguments
    /// * `data_flat` - Flattened [N·F] row-major input data
    /// * `n_items` - Number of rows N
    /// * `n_features` - Original dimension F
    ///
    /// # Returns
    /// `ReductionOutput` with projected data and projection matrix.
    ///
    /// If F < `min_dim_threshold`, returns identity projection (no reduction).
    pub fn execute(&self, data_flat: &[f32], n_items: usize, n_features: usize) -> ReductionOutput {
        info!("╔═══════════════════════════════════════════════════════╗");
        info!("║  STAGE A': JOHNSON-LINDENSTRAUSS REDUCTION           ║");
        info!("╚═══════════════════════════════════════════════════════╝");
        info!("  Input: N={}, F={}", n_items, n_features);

        // Check if reduction is needed
        if n_features < self.config.min_dim_threshold {
            info!(
                "  ⊗ Skipping: F={} < threshold={}",
                n_features, self.config.min_dim_threshold
            );
            return self.identity_projection(data_flat, n_items, n_features);
        }

        // Compute target dimension
        let target_dim = compute_jl_dimension(n_items, n_features, self.config.epsilon)
            .min(self.config.max_target_dim);

        if target_dim >= n_features {
            info!("  ⊗ Skipping: R={} ≥ F={}", target_dim, n_features);
            return self.identity_projection(data_flat, n_items, n_features);
        }

        info!(
            "  → Projecting to R={} (ε={:.2})",
            target_dim, self.config.epsilon
        );

        // Create projection matrix
        let projection = ImplicitProjection::new(n_features, target_dim, self.config.seed);

        // Project data
        info!("  🖥️  CPU projection (parallel)");
        let projected_data = self.project_cpu(data_flat, n_features, &projection);

        let compression = n_features as f32 / target_dim as f32;

        info!("  ✓ Reduction complete: {:.2}x compression", compression);
        info!("╔═══════════════════════════════════════════════════════╗");
        info!("║  STAGE A' COMPLETE                                    ║");
        info!("╚═══════════════════════════════════════════════════════╝");

        ReductionOutput {
            projected_data,
            n_items,
            reduced_dim: target_dim,
            original_dim: n_features,
            projection,
            compression_ratio: compression,
        }
    }

    /// CPU-based projection using Rayon parallelism.
    fn project_cpu(
        &self,
        data_flat: &[f32],
        f: usize,
        projection: &ImplicitProjection,
    ) -> Vec<f32> {
        let rows: Vec<Vec<f32>> = data_flat
            .chunks_exact(f)
            .map(|chunk| chunk.to_vec())
            .collect();

        let projected_rows = projection.project_batch_cpu(&rows);

        projected_rows.into_iter().flatten().collect()
    }

    /// Identity projection (no-op when F is small).
    fn identity_projection(&self, data_flat: &[f32], n: usize, f: usize) -> ReductionOutput {
        let projection = ImplicitProjection::new(f, f, self.config.seed);

        ReductionOutput {
            projected_data: data_flat.to_vec(),
            n_items: n,
            reduced_dim: f,
            original_dim: f,
            projection,
            compression_ratio: 1.0,
        }
    }
}
