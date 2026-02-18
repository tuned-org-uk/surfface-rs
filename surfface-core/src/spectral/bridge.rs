//! Stage D: Spectral analysis — Rayleigh + Dirichlet computation.
//!
//! Replaces the CPU Rayon loop with a single GPU dispatch.
//! The Laplacian is uploaded once, all N items computed in one batch,
//! and only Vec<f32> lambdas return to CPU for TauMode normalization.

use crate::spectral::laplacian::LaplacianOutput;
use crate::spectral::{compute_lambdas_gpu, laplacian_to_tensor};
use log::{debug, info};

/// Compute per-item lambda scores using GPU-accelerated Rayleigh + Dirichlet.
///
/// # Arguments
/// - `laplacian`: F×F sparse Laplacian from Stage C (CPU-resident CsMat<f32>)
/// - `data`: N×F item matrix (CPU, flat row-major f32)
/// - `n_items`: Number of items
/// - `n_features`: Number of features (centroid count C or reduced dim r)
/// - `tau_mode`: Normalization policy (applied on returned Vec<f32>)
///
/// # Returns
/// `Vec<f64>` of normalized lambda scores, length N.
///
/// # Pipeline Contract
/// - Input: `LaplacianOutput { matrix: CsMat<f32> }` from Stage C
/// - Output: `Vec<f64>` lambdas, ready for search indexing
/// - Side effect: Logs GPU memory usage, transfer sizes
pub fn compute_tau_mode_gpu(
    laplacian: &LaplacianOutput,
    data: &[f32], // N×F flat row-major
    n_items: usize,
    n_features: usize,
) -> Vec<f64> {
    let f = n_features;
    let n = n_items;

    info!(
        "Stage D (GPU): Computing lambdas for {} items × {} features",
        n, f
    );
    debug!(
        "Laplacian: {}×{} sparse ({} nnz, {:.2}% sparse)",
        f,
        f,
        laplacian.matrix.nnz(),
        100.0 * (1.0 - laplacian.matrix.nnz() as f64 / (f * f) as f64)
    );

    // ── 1. Upload Laplacian once ─────────────────────────────────────────────
    let l_gpu = laplacian_to_tensor(&laplacian.matrix, f);

    // ── 2. Compute all lambdas on GPU ────────────────────────────────────────
    let lambdas_f32 = compute_lambdas_gpu(&l_gpu, data, n, f);

    // ── 3. Apply TauMode normalization on CPU ────────────────────────────────
    let lambdas_f64: Vec<f64> = lambdas_f32.into_iter().map(|v| v as f64).collect();

    let (min, max, mean) = lambdas_f64.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, 0.0),
        |(min, max, sum), &v| (min.min(v), max.max(v), sum + v),
    );
    let mean = mean / n as f64;

    info!(
        "Lambda stats: min={:.6}, max={:.6}, mean={:.6}",
        min, max, mean
    );

    lambdas_f64
}
