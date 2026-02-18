//! GPU-accelerated Rayleigh quotient and Dirichlet dispersion computation.
//!
//! Uses the project's AutoBackend to keep all F×F Laplacian operations on-device.
//! Only the final N lambda scalars are downloaded to CPU.
//!
//! Design invariants:
//! - Laplacian [F,F] uploaded once per build, stays on GPU
//! - All item data [N,F] uploaded once, computation fully batched
//! - Output Vec<f32> [N] is the only CPU←GPU transfer
//!
//! Burn 0.20 squeeze API:
//!   .squeeze()             → removes ALL size-1 dims, panics if result is rank 0
//!   .squeeze_dim::<D2>(d)  → removes exactly one known dim d, const D2 = D-1
//!   .squeeze_dims::<D2>(&[isize]) → removes multiple named dims
//!
//! Rule used here: sum_dim(1) on [N, F] → [N, 1], then squeeze_dim::<1>(1) → [N].
pub mod bridge;
pub mod laplacian;

use burn::prelude::*;
use log::{debug, info, trace};
use sprs::CsMat;

use crate::backend::{AutoBackend, get_device};

/// Upload a sparse CsMat<f32> [F,F] to a dense GPU tensor.
/// Called once per pipeline — converts sparse → dense on CPU, then uploads.
///
/// Memory cost: F² × 4 bytes (400 MB at F=10K).
pub fn laplacian_to_tensor(matrix: &CsMat<f32>, f: usize) -> Tensor<AutoBackend, 2> {
    let device = get_device();

    trace!(
        "Densifying {}×{} sparse Laplacian ({} non-zeros)",
        f,
        f,
        matrix.nnz()
    );

    let mut flat = vec![0.0f32; f * f];
    for (&val, (row, col)) in matrix.iter() {
        flat[row * f + col] = val;
    }

    debug!(
        "Uploading {}×{} Laplacian to GPU ({:.1} MB)",
        f,
        f,
        (f * f * 4) as f64 / 1e6
    );

    Tensor::<AutoBackend, 2>::from_data(TensorData::new(flat, Shape::new([f, f])), &device)
}

/// Compute per-item Rayleigh quotient on GPU: e_i = (x_i^T L x_i) / (x_i^T x_i).
///
/// # Shapes
/// - `l_gpu`:  [F, F]  — already on device
/// - `x_gpu`:  [N, F]  — already on device
/// - returns:  [N]     — one scalar per item
///
/// Negative values are valid for non-PSD Laplacians (feature-space case).
///
/// # Burn 0.20 shape flow
///   x_gpu.clone().transpose()            [F, N]
///   l_gpu.matmul(...)                    [F, N]
///   .transpose()                         [N, F]   = lx
///   (x_gpu * lx).sum_dim(1)              [N, 1]
///   .squeeze_dim::<1>(1)                 [N]      ← key fix
pub fn rayleigh_quotient_gpu(
    l_gpu: &Tensor<AutoBackend, 2>,
    x_gpu: &Tensor<AutoBackend, 2>,
) -> Tensor<AutoBackend, 1> {
    trace!("Computing Rayleigh quotient: LX^T matmul");

    // LX^T = [F,F] × [F,N] → [F,N], then transpose → [N,F]
    let lx: Tensor<AutoBackend, 2> = l_gpu.clone().matmul(x_gpu.clone().transpose()).transpose();

    // Numerator:   Σ_f  x_if · (Lx)_if  →  [N, 1]  →  [N]
    let numerator: Tensor<AutoBackend, 1> = (x_gpu.clone() * lx).sum_dim(1).squeeze_dim::<1>(1);

    // Denominator: Σ_f  x²_if  →  [N, 1]  →  [N]
    let denominator: Tensor<AutoBackend, 1> = x_gpu
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .squeeze_dim::<1>(1);

    // Safe Rayleigh quotient, clamped to prevent overflow
    (numerator / (denominator + 1e-9)).clamp(-1e6_f32, 1e6_f32)
}

/// Compute per-item Dirichlet dispersion on GPU.
///
/// G_i = (Σ_j w_ij (x_i - x_j)²) / total
/// where W = max(0, -L) extracts positive off-diagonal weights.
///
/// # Efficient expansion (avoids explicit O(N²F) pairwise tensor)
///   Σ_j w_ij(x_i−x_j)² = diag(W·1)·x²  −  2x·(W·x)  +  W·(x²)
///
/// # Burn 0.20 shape flow
///   w.sum_dim(1).squeeze_dim::<1>(1)      [F]        degree vector
///   w.matmul(x^T).transpose()             [N, F]     WX
///   w.matmul(x²^T).transpose()            [N, F]     WX²
///   degree.unsqueeze_dim(0).expand([N,F]) [N, F]     broadcast
///   row result.sum_dim(1).squeeze_dim(1)  [N]        ← key fix
pub fn dirichlet_dispersion_gpu(
    l_gpu: &Tensor<AutoBackend, 2>,
    x_gpu: &Tensor<AutoBackend, 2>,
    n: usize,
    f: usize,
) -> Tensor<AutoBackend, 1> {
    trace!("Computing Dirichlet dispersion: extracting weight matrix");

    // W = clamp(-L, 0, ∞)  — positive off-diagonal weights [F, F]
    let w: Tensor<AutoBackend, 2> = l_gpu.clone().neg().clamp_min(0.0_f32);

    // Degree vector: d_f = Σ_f' w_ff'  →  [F, 1]  →  [F]
    let degree: Tensor<AutoBackend, 1> = w.clone().sum_dim(1).squeeze_dim::<1>(1);

    trace!("Computing Dirichlet: W·X and W·X² matmuls");

    // WX:  [F,F] × [F,N] → [F,N], transpose → [N,F]
    let wx: Tensor<AutoBackend, 2> = w.clone().matmul(x_gpu.clone().transpose()).transpose();

    // W(X²): same pattern
    let x_sq: Tensor<AutoBackend, 2> = x_gpu.clone().powf_scalar(2.0); // [N,F]
    let wx_sq: Tensor<AutoBackend, 2> = w.matmul(x_sq.clone().transpose()).transpose(); // [N,F]

    // Broadcast degree [F] → [1, F] → [N, F]
    let deg_broadcast: Tensor<AutoBackend, 2> = degree.unsqueeze_dim::<2>(0).expand([n, f]); // [N,F]

    // Per-feature edge energy: deg·x² − 2x·(Wx) + W(x²)  →  [N, F]
    let edge_energy: Tensor<AutoBackend, 2> =
        deg_broadcast * x_sq - x_gpu.clone() * wx * 2.0 + wx_sq;

    // Row-sum → [N, 1] → [N], then normalize by global total
    let row_sums: Tensor<AutoBackend, 1> = edge_energy
        .clamp_min(0.0_f32)
        .sum_dim(1)
        .squeeze_dim::<1>(1); // [N]

    let total = row_sums.clone().sum(); // scalar

    trace!("Normalizing Dirichlet by total energy");
    (row_sums / (total + 1e-12_f32)).clamp(0.0_f32, 1.0_f32)
}

/// Compute all N lambda scores (Rayleigh + Dirichlet) fully on GPU.
///
/// # Arguments
/// - `l_gpu`: [F,F] Laplacian tensor (already on device from `laplacian_to_tensor`)
/// - `data`:  Flat f32 slice [N×F] row-major item data (CPU)
/// - `n`:     Number of items
/// - `f`:     Number of features
///
/// # Returns
/// `Vec<f32>` of length N — the only data transferred from GPU to CPU.
pub fn compute_lambdas_gpu(
    l_gpu: &Tensor<AutoBackend, 2>,
    data: &[f32],
    n: usize,
    f: usize,
) -> Vec<f32> {
    let device = get_device();

    info!("Computing lambdas for {} items × {} features on GPU", n, f);
    debug!("Uploading item matrix ({:.1} MB)", (n * f * 4) as f64 / 1e6);

    // Upload item data once: [N, F]
    let x_gpu: Tensor<AutoBackend, 2> =
        Tensor::from_data(TensorData::new(data.to_vec(), Shape::new([n, f])), &device);

    let rayleigh = rayleigh_quotient_gpu(l_gpu, &x_gpu);
    let dirichlet = dirichlet_dispersion_gpu(l_gpu, &x_gpu, n, f);

    // Combine: λ = Rayleigh + Dirichlet (TauMode normalization downstream on CPU)
    let lambda_gpu: Tensor<AutoBackend, 1> = rayleigh + dirichlet;

    debug!("Downloading {} lambda scores to CPU ({} bytes)", n, n * 4);
    lambda_gpu.to_data().to_vec().unwrap()
}
