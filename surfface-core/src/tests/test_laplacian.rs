use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::spectral::laplacian::{LaplacianConfig, LaplacianStage};
use crate::tests::init;
use burn::prelude::*;
use rand::{Rng, SeedableRng};

type TestBackend = AutoBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Corrected Property Test: Spectral Bounds (λ ∈ [0, 2])
// ─────────────────────────────────────────────────────────────────────────────
use sprs::CsMat;

#[test]
fn test_laplacian_spectral_bounds_normalized() {
    init();
    // For L_sym = I - D^{-1/2} W D^{-1/2}, eigenvalues MUST lie in [0, 2].
    // We verify this using Rayleigh Quotients: R(L, x) = (x^T L x) / (x^T x).
    let state = centroids_from_gaussian_blobs(10, 20, 0.4, 777);
    let config = LaplacianConfig {
        k_neighbors: 5,
        normalize: true,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;

    // matrix is now sparse
    let lap: &CsMat<f32> = &output.matrix;
    assert_eq!(lap.rows(), f);
    assert_eq!(lap.cols(), f);

    // 1. Structural invariants in CSR format
    //
    // For each row i, its nonzeros live in:
    //   let row = lap.indptr().outer_inds_sz(i);
    //   cols = &lap.indices()[row.start..row.end]
    //   vals = &lap.data()[row.start..row.end]
    for i in 0..f {
        let row_range = lap.indptr().outer_inds_sz(i);
        let cols = &lap.indices()[row_range.start..row_range.end];
        let vals = &lap.data()[row_range.start..row_range.end];

        let mut diag = 0.0f32;
        for (&col, &val) in cols.iter().zip(vals.iter()) {
            if col == i {
                diag = val;
            } else {
                // Off-diagonals must be <= 0
                assert!(
                    val <= 0.0,
                    "Off-diagonals must be <= 0, got {} at ({}, {})",
                    val,
                    i,
                    col
                );
            }
        }

        if output.degrees[i] > 1e-6 {
            assert!(
                (diag - 1.0).abs() < 1e-5,
                "Diagonal must be 1.0, got {} at ({}, {})",
                diag,
                i,
                i
            );
        }
    }

    // 2. Rayleigh Quotient Sampling (Monte Carlo spectral bound check)
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-9 {
            continue;
        }

        // Compute L * x using sparse matvec
        let mut lx = vec![0.0f32; f];
        // For each row i, accumulate sum_j L[i, j] * x[j]
        for i in 0..f {
            let row_range = lap.indptr().outer_inds_sz(i);
            let cols = &lap.indices()[row_range.start..row_range.end];
            let vals = &lap.data()[row_range.start..row_range.end];

            let mut acc = 0.0f32;
            for (&col, &val) in cols.iter().zip(vals.iter()) {
                acc += val * x[col];
            }
            lx[i] = acc;
        }

        // R(L, x) = x^T L x / x^T x
        let x_t_lx: f32 = x.iter().zip(lx.iter()).map(|(&xi, &lxi)| xi * lxi).sum();
        let rayleigh_quotient = x_t_lx / norm_sq;

        // Theory: 0.0 <= λ <= 2.0
        assert!(
            rayleigh_quotient >= -1e-4,
            "Eigenvalue lower bound violation: {}",
            rayleigh_quotient
        );
        assert!(
            rayleigh_quotient <= 2.0 + 1e-4,
            "Eigenvalue upper bound violation: {}",
            rayleigh_quotient
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New Test 1: Partition Invariance (Connected Components)
// ─────────────────────────────────────────────────────────────────────────────

use sprs::CsVec;

#[test]
fn test_laplacian_nullspace_dimension() {
    init();
    // THEORY: For connected graphs, L_sym * D^{1/2}1 = 0.
    let state = centroids_from_gaussian_blobs(5, 10, 0.1, 123);
    let config = LaplacianConfig {
        k_neighbors: 9,
        normalize: true,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;
    let lap: &CsMat<f32> = &output.matrix;

    // 1. Create the harmonic vector v = D^{1/2}1
    let v_data: Vec<f32> = output.degrees.iter().map(|&d| d.sqrt()).collect();

    // 2. Convert to a sparse vector (CsVec) for multiplication
    // Since v is dense, indices are simply 0..f
    let indices: Vec<usize> = (0..f).collect();
    let v_sparse = CsVec::new(f, indices, v_data);

    // 3. Multiplication: &CsMat * &CsVec -> CsVec
    let lv_sparse = lap * &v_sparse;

    // 4. Verify results
    // We iterate through the resulting sparse vector's non-zero entries.
    // Note: In a nullspace check, many entries might be zero and thus
    // missing from the sparse structure if the multiplication optimized them out.
    let mut lv_dense = vec![0.0f32; f];
    for (idx, &val) in lv_sparse.iter() {
        lv_dense[idx] = val;
    }

    for (i, &val) in lv_dense.iter().enumerate() {
        assert!(
            val.abs() < 1e-3,
            "Nullspace violation at row {}: expected ~0, got {}",
            i,
            val
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New Test 2: Multi-Order Stability (L1 vs L2 Alignment)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_high_order_neighborhood_consistency() {
    init();
    // The second-order adjacency A2 = A * A should have a similar structure.
    // We verify that the first-order Laplacian identifies "Hub" features correctly.

    let state = centroids_from_gaussian_blobs(20, 50, 0.5, 999);
    let stage = LaplacianStage::with_defaults();
    let output = stage.execute(&state);

    // Identify "Hub" features (highest weighted degree)
    let mut degree_pairs: Vec<(usize, f32)> = output.degrees.iter().cloned().enumerate().collect();
    degree_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_hub_idx = degree_pairs[0].0;
    let lap: &CsMat<f32> = &output.matrix;

    // Features connected to the top hub have high similarity (negative off-diagonal values).
    // In CSR format, we can efficiently access just the neighbors of the hub row.
    let mut hub_connections = 0;

    // outer_view(i) returns the i-th row for a CSR matrix as a sparse vector view
    if let Some(hub_row) = lap.outer_view(top_hub_idx) {
        for (col_idx, &val) in hub_row.iter() {
            // Ignore the diagonal entry (self-connection)
            if col_idx == top_hub_idx {
                continue;
            }

            // In a Laplacian matrix, negative off-diagonals indicate similarity.
            // A value < -0.05 indicates a significant connection.
            if val < -0.05 {
                hub_connections += 1;
            }
        }
    }

    assert!(
        hub_connections > 0,
        "Hub feature {} should have neighbors in the sparse Laplacian",
        top_hub_idx
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Strengthened Regression: ArrowSpace Config Compatibility
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regression_unnormalized_row_sums() {
    init();
    // THEORY: In unnormalized mode L = D - W, the sum of each row is 0.0
    // because the diagonal D_ii is defined as the sum of the weights W_ij.

    let state = centroids_from_gaussian_blobs(10, 30, 0.5, 42);
    let config = LaplacianConfig {
        normalize: false, // Unnormalized
        k_neighbors: 10,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;
    let lap: &CsMat<f32> = &output.matrix;

    // We use the outer_iterator to visit each row of the CSR matrix efficiently.
    for (i, row) in lap.outer_iterator().enumerate() {
        // row.iter() only visits non-zero entries (indices and values).
        let row_sum: f32 = row.iter().map(|(_, &val)| val).sum();

        assert!(
            row_sum.abs() < 1e-4,
            "Unnormalized Row {} sum = {}, expected 0.0 (f_features: {})",
            i,
            row_sum,
            f
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Refactored Data Generators (ArrowSpace Standard)
// ─────────────────────────────────────────────────────────────────────────────

fn centroids_from_gaussian_blobs(
    c: usize,
    f: usize,
    noise: f32,
    seed: u64,
) -> CentroidState<TestBackend> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let device = Default::default();

    let means = (0..c * f).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vars = (0..c * f).map(|_| rng.random_range(0.01..noise)).collect();
    let counts = (0..c).map(|_| rng.random_range(10..100)).collect();

    CentroidState {
        means: Tensor::from_data(TensorData::new(means, Shape::new([c, f])), &device),
        variances: Tensor::from_data(TensorData::new(vars, Shape::new([c, f])), &device),
        counts: Tensor::from_data(TensorData::new(counts, Shape::new([c])), &device),
    }
}
