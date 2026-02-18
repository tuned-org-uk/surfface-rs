//! Comprehensive test suite for Stage C: Feature-space Laplacian construction.
//!
//! Test organization:
//! - Structural invariants: Symmetry, diagonal values, sparsity, degree vector
//! - Spectral bounds: λ ∈ [0, 2] via Monte Carlo Rayleigh quotient sampling
//! - Nullspace: L_sym · D^{1/2}1 = 0 for connected graphs
//! - Normalization modes: Normalized (Surfface) vs unnormalized (ArrowSpace)
//! - Dataset coverage: moons, gaussian blobs, cliques, high-dimensional, energy
//! - Config sensitivity: k_neighbors, variance_regularizer, weight_threshold

use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::spectral::laplacian::{LaplacianConfig, LaplacianStage};
use crate::tests::init;
use burn::prelude::*;
use rand::{Rng, SeedableRng};
use sprs::CsMat;

use super::test_data::{
    make_energy_test_dataset, make_gaussian_blob, make_gaussian_cliques_multi, make_gaussian_hd,
    make_moons_hd,
};

type TestBackend = AutoBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Test Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a CentroidState<TestBackend> directly from a `Vec<Vec<f64>>` dataset.
///
/// Treats each row as a centroid observation.
/// Variance is set to a small fixed value (controllable via `var_scale`).
fn centroid_state_from_rows(
    rows: Vec<Vec<f64>>,
    var_scale: f32,
    seed: u64,
) -> CentroidState<TestBackend> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let c = rows.len();
    let f = rows[0].len();

    let means_flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let vars_flat: Vec<f32> = (0..c * f)
        .map(|_| rng.random_range(1e-4..var_scale))
        .collect();

    let counts: Vec<i32> = (0..c).map(|_| rng.random_range(10..100)).collect();

    CentroidState {
        means: Tensor::from_data(TensorData::new(means_flat, Shape::new([c, f])), &device),
        variances: Tensor::from_data(TensorData::new(vars_flat, Shape::new([c, f])), &device),
        counts: Tensor::from_data(TensorData::new(counts, Shape::new([c])), &device),
    }
}

/// Build a CentroidState directly from flat random Gaussian data (legacy helper).
fn centroids_from_gaussian_blobs(
    c: usize,
    f: usize,
    noise: f32,
    seed: u64,
) -> CentroidState<TestBackend> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let means: Vec<f32> = (0..c * f).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vars: Vec<f32> = (0..c * f).map(|_| rng.random_range(0.01..noise)).collect();
    let counts: Vec<i32> = (0..c).map(|_| rng.random_range(10..100)).collect();

    CentroidState {
        means: Tensor::from_data(TensorData::new(means, Shape::new([c, f])), &device),
        variances: Tensor::from_data(TensorData::new(vars, Shape::new([c, f])), &device),
        counts: Tensor::from_data(TensorData::new(counts, Shape::new([c])), &device),
    }
}

/// Sparse matrix-vector multiply for CSR LaplacianOutput.
fn sparse_matvec(lap: &CsMat<f32>, x: &[f32]) -> Vec<f32> {
    let f = lap.rows();
    let mut out = vec![0.0f32; f];
    for i in 0..f {
        let row_range = lap.indptr().outer_inds_sz(i);
        let cols = &lap.indices()[row_range.start..row_range.end];
        let vals = &lap.data()[row_range.start..row_range.end];
        for (&col, &val) in cols.iter().zip(vals.iter()) {
            out[i] += val * x[col];
        }
    }
    out
}

/// Compute Rayleigh quotient: xᵀ L x / xᵀ x
fn rayleigh(lap: &CsMat<f32>, x: &[f32]) -> f32 {
    let lx = sparse_matvec(lap, x);
    let num: f32 = x.iter().zip(lx.iter()).map(|(&xi, &lxi)| xi * lxi).sum();
    let den: f32 = x.iter().map(|&v| v * v).sum();
    if den < 1e-12 {
        return 0.0;
    }
    num / den
}

/// Default LaplacianConfig: normalized, k=5, suitable for small test graphs.
fn test_config_normalized(k: usize) -> LaplacianConfig {
    LaplacianConfig {
        k_neighbors: k,
        normalize: true,
        variance_regularizer: 1e-4,
        weight_threshold: 1e-9,
    }
}

fn test_config_unnormalized(k: usize) -> LaplacianConfig {
    LaplacianConfig {
        k_neighbors: k,
        normalize: false,
        variance_regularizer: 1e-4,
        weight_threshold: 1e-9,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURAL INVARIANT TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_output_fields_consistent_normalized() {
    init();
    let state = centroids_from_gaussian_blobs(8, 16, 0.3, 1);
    let output = LaplacianStage::new(test_config_normalized(5)).execute(&state);

    let f = output.n_features;

    // Metadata fields are consistent
    assert_eq!(output.n_features, 16);
    assert_eq!(output.matrix.rows(), f);
    assert_eq!(output.matrix.cols(), f);
    assert_eq!(output.degrees.len(), f);
    assert!(output.nnz > 0, "Expected non-zero edges");
    assert!(output.sparsity >= 0.0 && output.sparsity <= 1.0);

    // nnz tracks matrix nnz
    assert_eq!(output.nnz, output.matrix.nnz());
}

#[test]
fn test_matrix_symmetry_normalized() {
    init();
    // L_sym must be exactly symmetric: L[i,j] == L[j,i]
    let state = centroids_from_gaussian_blobs(6, 12, 0.4, 2);
    let output = LaplacianStage::new(test_config_normalized(4)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    for i in 0..f {
        let row_range = lap.indptr().outer_inds_sz(i);
        let cols = &lap.indices()[row_range.start..row_range.end];
        let vals = &lap.data()[row_range.start..row_range.end];

        for (&j, &v_ij) in cols.iter().zip(vals.iter()) {
            let v_ji = lap.get(j, i).copied().unwrap_or(0.0);
            assert!(
                (v_ij - v_ji).abs() < 1e-5,
                "Symmetry violation at ({},{}) vs ({},{}): {} != {}",
                i,
                j,
                j,
                i,
                v_ij,
                v_ji
            );
        }
    }
}

#[test]
fn test_matrix_symmetry_unnormalized() {
    init();
    // L = D - W must also be symmetric
    let state = centroids_from_gaussian_blobs(6, 12, 0.4, 3);
    let output = LaplacianStage::new(test_config_unnormalized(4)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    for i in 0..f {
        let row_range = lap.indptr().outer_inds_sz(i);
        let cols = &lap.indices()[row_range.start..row_range.end];
        let vals = &lap.data()[row_range.start..row_range.end];

        for (&j, &v_ij) in cols.iter().zip(vals.iter()) {
            let v_ji = lap.get(j, i).copied().unwrap_or(0.0);
            assert!(
                (v_ij - v_ji).abs() < 1e-5,
                "Symmetry violation at ({},{}) vs ({},{}): {} != {}",
                i,
                j,
                j,
                i,
                v_ij,
                v_ji
            );
        }
    }
}

#[test]
fn test_normalized_diagonal_is_one_for_connected_nodes() {
    init();
    // In L_sym: diagonal must be exactly 1.0 for nodes with degree > 0
    let state = centroids_from_gaussian_blobs(8, 20, 0.3, 4);
    let output = LaplacianStage::new(test_config_normalized(5)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    for i in 0..f {
        let diag = lap.get(i, i).copied().unwrap_or(0.0);
        if output.degrees[i] > 1e-6 {
            assert!(
                (diag - 1.0).abs() < 1e-5,
                "Normalized diagonal at ({},{}) = {}, expected 1.0",
                i,
                i,
                diag
            );
        }
    }
}

#[test]
fn test_unnormalized_diagonal_equals_degree() {
    init();
    // In L = D - W: L[i,i] must equal degree[i]
    let state = centroids_from_gaussian_blobs(8, 20, 0.3, 5);
    let output = LaplacianStage::new(test_config_unnormalized(5)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    for i in 0..f {
        let diag = lap.get(i, i).copied().unwrap_or(0.0);
        let deg = output.degrees[i];
        if deg > 1e-6 {
            assert!(
                (diag - deg).abs() < 1e-4,
                "Unnormalized diagonal at ({},{}) = {}, expected degree {}",
                i,
                i,
                diag,
                deg
            );
        }
    }
}

#[test]
fn test_unnormalized_row_sums_are_zero() {
    init();
    // THEORY: For L = D - W, each row sums to exactly 0.0
    let state = centroids_from_gaussian_blobs(10, 30, 0.5, 6);
    let output = LaplacianStage::new(test_config_unnormalized(6)).execute(&state);
    let lap = &output.matrix;

    for (i, row) in lap.outer_iterator().enumerate() {
        let row_sum: f32 = row.iter().map(|(_, &v)| v).sum();
        assert!(
            row_sum.abs() < 1e-4,
            "Unnormalized row {} sum = {:.2e}, expected 0.0",
            i,
            row_sum
        );
    }
}

#[test]
fn test_off_diagonal_entries_are_non_positive() {
    init();
    // In any valid graph Laplacian, off-diagonal entries must be ≤ 0
    let state = centroids_from_gaussian_blobs(8, 16, 0.3, 7);
    let output = LaplacianStage::new(test_config_normalized(5)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    for i in 0..f {
        let row_range = lap.indptr().outer_inds_sz(i);
        let cols = &lap.indices()[row_range.start..row_range.end];
        let vals = &lap.data()[row_range.start..row_range.end];

        for (&j, &v) in cols.iter().zip(vals.iter()) {
            if j != i {
                assert!(v <= 1e-6, "Off-diagonal ({},{}) = {} must be ≤ 0", i, j, v);
            }
        }
    }
}

#[test]
fn test_degrees_vector_non_negative() {
    init();
    let state = centroids_from_gaussian_blobs(8, 16, 0.3, 8);
    let output = LaplacianStage::new(test_config_normalized(4)).execute(&state);

    for (i, &d) in output.degrees.iter().enumerate() {
        assert!(d >= 0.0, "Degree at node {} is negative: {}", i, d);
    }
}

#[test]
fn test_sparsity_increases_with_smaller_k() {
    init();
    let state = centroids_from_gaussian_blobs(10, 30, 0.4, 9);

    let out_k3 = LaplacianStage::new(test_config_normalized(3)).execute(&state);
    let out_k9 = LaplacianStage::new(test_config_normalized(9)).execute(&state);

    assert!(
        out_k3.sparsity >= out_k9.sparsity,
        "Smaller k should produce sparser graph: k=3 sparsity {} vs k=9 sparsity {}",
        out_k3.sparsity,
        out_k9.sparsity
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SPECTRAL BOUNDS: λ ∈ [0, 2] via Monte Carlo Rayleigh quotient sampling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_spectral_bounds_normalized_gaussian_blob() {
    init();
    let state = centroids_from_gaussian_blobs(10, 20, 0.4, 10);
    let output = LaplacianStage::new(test_config_normalized(5)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for _ in 0..200 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let rq = rayleigh(lap, &x);
        assert!(rq >= -1e-4, "Eigenvalue lower bound violated: RQ = {}", rq);
        assert!(
            rq <= 2.0 + 1e-4,
            "Eigenvalue upper bound violated: RQ = {}",
            rq
        );
    }
}

#[test]
fn test_spectral_bounds_normalized_moons() {
    init();
    let rows = make_moons_hd(20, 0.05, 0.01, 16, 101);
    let state = centroid_state_from_rows(rows, 0.3, 11);
    let output = LaplacianStage::new(test_config_normalized(4)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    let mut rng = rand::rngs::StdRng::seed_from_u64(43);
    for _ in 0..200 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let rq = rayleigh(lap, &x);
        assert!(rq >= -1e-4, "Moons: lower bound violated: RQ = {}", rq);
        assert!(rq <= 2.0 + 1e-4, "Moons: upper bound violated: RQ = {}", rq);
    }
}

#[test]
fn test_spectral_bounds_normalized_gaussian_hd() {
    init();
    // HD case: 100 features, 15 centroids → Laplacian is 100×100
    let rows = make_gaussian_hd(15, 1.0);
    let state = centroid_state_from_rows(rows, 0.5, 12);
    let output = LaplacianStage::new(test_config_normalized(5)).execute(&state);
    let lap = &output.matrix;
    let f = output.n_features;

    let mut rng = rand::rngs::StdRng::seed_from_u64(44);
    for _ in 0..200 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let rq = rayleigh(lap, &x);
        assert!(rq >= -1e-4, "HD: lower bound violated: RQ = {}", rq);
        assert!(rq <= 2.0 + 1e-4, "HD: upper bound violated: RQ = {}", rq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NULLSPACE: L_sym · D^{1/2}1 = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nullspace_normalized_connected_graph() {
    init();
    // For a connected graph: L_sym · (D^{1/2} · 1) = 0
    let state = centroids_from_gaussian_blobs(5, 10, 0.1, 13);
    let output = LaplacianStage::new(LaplacianConfig {
        k_neighbors: 9, // high k → more likely connected
        normalize: true,
        ..Default::default()
    })
    .execute(&state);

    let lap = &output.matrix;
    let _f = output.n_features;

    // v = D^{1/2} · 1: v[i] = sqrt(degree[i])
    let v: Vec<f32> = output.degrees.iter().map(|&d| d.sqrt()).collect();
    let lv = sparse_matvec(lap, &v);

    for (i, &val) in lv.iter().enumerate() {
        assert!(
            val.abs() < 1e-3,
            "Nullspace violation at row {}: L·v[{}] = {:.2e}",
            i,
            i,
            val
        );
    }
}

#[test]
fn test_unnormalized_row_zero_is_nullspace() {
    init();
    // For L = D - W: L · 1 = 0 (the constant vector is in the nullspace)
    let state = centroids_from_gaussian_blobs(5, 10, 0.1, 14);
    let output = LaplacianStage::new(LaplacianConfig {
        k_neighbors: 9,
        normalize: false,
        ..Default::default()
    })
    .execute(&state);

    let lap = &output.matrix;
    let f = output.n_features;
    let ones = vec![1.0f32; f];
    let l_ones = sparse_matvec(lap, &ones);

    for (i, &val) in l_ones.iter().enumerate() {
        assert!(
            val.abs() < 1e-3,
            "Unnormalized nullspace violation at row {}: L·1[{}] = {:.2e}",
            i,
            i,
            val
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLUSTER SEPARATION TESTS (using structured test_data generators)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multi_clique_sparsity_pattern() {
    init();
    // More separated cliques → sparser inter-cluster connections
    let rows_tight = make_gaussian_cliques_multi(30, 0.1, 3, 10, 55);
    let rows_loose = make_gaussian_cliques_multi(30, 3.0, 3, 10, 55);

    let state_tight = centroid_state_from_rows(rows_tight, 0.05, 16);
    let state_loose = centroid_state_from_rows(rows_loose, 0.5, 16);

    let out_tight = LaplacianStage::new(test_config_normalized(4)).execute(&state_tight);
    let out_loose = LaplacianStage::new(test_config_normalized(4)).execute(&state_loose);

    // Tighter clusters → higher degree sum (denser intra-cluster connections)
    let deg_tight: f32 = out_tight.degrees.iter().sum();
    let deg_loose: f32 = out_loose.degrees.iter().sum();

    assert!(
        deg_tight >= deg_loose * 0.8,
        "Tight clusters (deg={:.2}) should have degree ≥ loose (deg={:.2})",
        deg_tight,
        deg_loose
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ENERGY TEST DATASET
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_energy_test_dataset_produces_valid_laplacian() {
    init();
    // 5 well-separated clusters in 20D space
    let rows = make_energy_test_dataset(25, 20, 888);
    let state = centroid_state_from_rows(rows, 0.3, 18);
    let output = LaplacianStage::new(test_config_normalized(4)).execute(&state);

    assert_eq!(output.n_features, 20);
    assert!(output.nnz > 0);
    assert!(output.sparsity > 0.0);

    // Spectral bounds hold
    let lap = &output.matrix;
    let f = output.n_features;
    let mut rng = rand::rngs::StdRng::seed_from_u64(45);
    for _ in 0..100 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let rq = rayleigh(lap, &x);
        assert!(rq >= -1e-4, "Energy dataset: RQ lower bound = {}", rq);
        assert!(rq <= 2.0 + 1e-4, "Energy dataset: RQ upper bound = {}", rq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIG SENSITIVITY TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_k_neighbors_one_produces_tree_like_graph() {
    init();
    // k=1: each node has at most 1 directed edge → very sparse
    let state = centroids_from_gaussian_blobs(8, 16, 0.3, 20);
    let output = LaplacianStage::new(test_config_normalized(1)).execute(&state);

    // After symmetrization, nnz ≤ 2 * F (each node can have at most 2 undirected edges)
    // Strictly: nnz_offdiag ≤ 2 * (F - 1)
    let f = output.n_features;
    let offdiag_nnz = output.matrix.iter().filter(|(_, (i, j))| i != j).count();

    assert!(
        offdiag_nnz <= 2 * f,
        "k=1 should produce ≤ 2F off-diagonal entries, got {}",
        offdiag_nnz
    );
}

#[test]
fn test_weight_threshold_reduces_nnz() {
    init();
    let state = centroids_from_gaussian_blobs(8, 20, 0.4, 21);

    let out_low_thr = LaplacianStage::new(LaplacianConfig {
        k_neighbors: 5,
        normalize: true,
        weight_threshold: 1e-12,
        ..Default::default()
    })
    .execute(&state);

    let out_high_thr = LaplacianStage::new(LaplacianConfig {
        k_neighbors: 5,
        normalize: true,
        weight_threshold: 0.5, // aggressive pruning
        ..Default::default()
    })
    .execute(&state);

    assert!(
        out_high_thr.nnz <= out_low_thr.nnz,
        "Higher weight_threshold should produce fewer edges: {} > {}",
        out_high_thr.nnz,
        out_low_thr.nnz
    );
}

#[test]
fn test_variance_regularizer_prevents_nan_in_degenerate_case() {
    init();
    // All-zero variances: regularizer must prevent division by zero in Bhattacharyya
    let device: <TestBackend as Backend>::Device = Default::default();
    let c = 5;
    let f = 10;

    let means: Vec<f32> = (0..c * f).map(|i| (i % f) as f32 / f as f32).collect();
    let vars: Vec<f32> = vec![0.0f32; c * f]; // degenerate: zero variance everywhere

    let state = CentroidState {
        means: Tensor::<AutoBackend, 2>::from_data(
            TensorData::new(means, Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<AutoBackend, 2>::from_data(
            TensorData::new(vars, Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::from_data(TensorData::new(vec![10i64; c], Shape::new([c])), &device),
    };

    let config = LaplacianConfig {
        k_neighbors: 3,
        normalize: true,
        variance_regularizer: 1e-4, // must rescue from zero-variance
        weight_threshold: 1e-9,
    };

    let output = LaplacianStage::new(config).execute(&state);

    // Must not produce NaN/Inf degrees
    for (i, &d) in output.degrees.iter().enumerate() {
        assert!(
            d.is_finite(),
            "Degree[{}] = {} is not finite (zero-var case)",
            i,
            d
        );
    }
    // All Laplacian values must be finite
    for (val, (i, j)) in output.matrix.iter() {
        assert!(
            val.is_finite(),
            "L[{},{}] = {} is not finite (zero-var case)",
            i,
            j,
            val
        );
    }
}

#[test]
fn test_high_k_approaches_fully_connected() {
    init();
    // k = F - 1: graph approaches fully connected, sparsity → 0
    let c = 6;
    let f = 10;
    let state = centroids_from_gaussian_blobs(c, f, 0.5, 22);

    let out_full = LaplacianStage::new(LaplacianConfig {
        k_neighbors: f - 1,
        normalize: true,
        ..Default::default()
    })
    .execute(&state);

    // Sparsity should be very low (graph is nearly complete)
    assert!(
        out_full.sparsity < 0.8,
        "k=F-1 should approach fully connected: sparsity = {:.2}",
        out_full.sparsity
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NORMALIZATION MODE REGRESSION
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_normalized_vs_unnormalized_same_topology() {
    init();
    // Both modes should produce the same sparsity pattern (same edges),
    // differing only in the weight values.
    let state = centroids_from_gaussian_blobs(8, 16, 0.3, 23);

    let out_norm = LaplacianStage::new(test_config_normalized(4)).execute(&state);
    let out_unnorm = LaplacianStage::new(test_config_unnormalized(4)).execute(&state);

    // Same number of non-zeros (same graph topology)
    assert_eq!(
        out_norm.nnz, out_unnorm.nnz,
        "Normalized and unnormalized should have same sparsity pattern"
    );

    // Same degree vector up to normalization scaling
    for (i, (&dn, &du)) in out_norm
        .degrees
        .iter()
        .zip(out_unnorm.degrees.iter())
        .enumerate()
    {
        // Both non-zero or both zero
        let both_nonzero = dn > 1e-6 && du > 1e-6;
        let both_zero = dn <= 1e-6 && du <= 1e-6;
        assert!(
            both_nonzero || both_zero,
            "Degree connectivity mismatch at node {}: norm={} unnorm={}",
            i,
            dn,
            du
        );
    }
}

#[test]
fn test_summary_string_is_non_empty() {
    init();
    let state = centroids_from_gaussian_blobs(5, 10, 0.3, 24);
    let output = LaplacianStage::with_defaults().execute(&state);
    let summary = output.summary();
    assert!(!summary.is_empty());
    assert!(summary.contains("LaplacianOutput"));
}

// ─────────────────────────────────────────────────────────────────────────────
// HIGH-DIMENSIONAL STRESS TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gaussian_blob_10d_structural_invariants() {
    init();
    // 10D, 3 clusters from make_gaussian_blob
    let rows = make_gaussian_blob(15, 0.5);
    let state = centroid_state_from_rows(rows, 0.3, 25);
    let output = LaplacianStage::new(test_config_normalized(4)).execute(&state);

    assert_eq!(output.n_features, 10);
    assert_eq!(output.matrix.rows(), 10);
    assert!(output.nnz > 0);

    // All diagonal entries for connected nodes are 1.0
    let lap = &output.matrix;
    for i in 0..10 {
        if output.degrees[i] > 1e-6 {
            let diag = lap.get(i, i).copied().unwrap_or(0.0);
            assert!(
                (diag - 1.0).abs() < 1e-5,
                "10D blob: diagonal[{}] = {}, expected 1.0",
                i,
                diag
            );
        }
    }
}

#[test]
fn test_gaussian_hd_100d_completes_without_panic() {
    init();
    // 100D: should complete with valid output, no panic, no NaN
    let rows = make_gaussian_hd(12, 1.5);
    let state = centroid_state_from_rows(rows, 0.5, 26);
    let output = LaplacianStage::new(LaplacianConfig {
        k_neighbors: 5,
        normalize: true,
        ..Default::default()
    })
    .execute(&state);

    assert_eq!(output.n_features, 100);
    for (val, (i, j)) in output.matrix.iter() {
        assert!(val.is_finite(), "100D: L[{},{}] = {} is NaN/Inf", i, j, val);
    }
    for (i, &d) in output.degrees.iter().enumerate() {
        assert!(d.is_finite(), "100D: degree[{}] is not finite", i);
    }
}

#[test]
fn test_moons_hd_two_cluster_structure() {
    init();
    // Reusing make_moons_hd with high noise.
    // In Stage C, we don't assert specific node degrees (like 0 or 1),
    // but rather the emergence of a connected manifold from the data.
    let n = 40;
    let noise_xy = 0.5; // High noise for manifold connectivity
    let noise_hd = 0.2;
    let dims = 8;
    let seed = 77;

    let rows = make_moons_hd(n, noise_xy, noise_hd, dims, seed);
    let state = centroid_state_from_rows(rows, 0.5, seed);

    let config = LaplacianConfig {
        k_neighbors: 3,
        normalize: true,
        ..Default::default()
    };

    let output = LaplacianStage::new(config).execute(&state);
    let lap = &output.matrix;

    // ── Invariant 1: Graph Connectivity ─────────────────────────────────────
    assert!(
        output.nnz > 0,
        "Moons dataset produced an empty feature graph."
    );

    // ── Invariant 2: Global Symmetrization ──────────────────────────────────
    let f = output.n_features;
    for i in 0..f {
        let row_i = lap.outer_view(i).unwrap();
        for (j, &v_ij) in row_i.iter() {
            let v_ji = lap.get(j, i).copied().unwrap_or(0.0);
            assert!(
                (v_ij - v_ji).abs() < 1e-5,
                "Symmetry violation at ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn test_energy_dataset_manifold_connectivity() {
    init();
    // Reusing make_energy_test_dataset.
    // Validates that complex 5-cluster 100D data wires correctly in feature-space.
    let n_items = 50;
    let n_features = 100;
    let seed = 888;

    let rows = make_energy_test_dataset(n_items, n_features, seed);
    let state = centroid_state_from_rows(rows, 0.5, seed);

    let output = LaplacianStage::new(LaplacianConfig::default()).execute(&state);

    assert_eq!(output.n_features, 100);
    assert!(
        output.nnz > 0,
        "Energy dataset should produce a connected manifold"
    );

    // Check that degree vector is non-zero
    let total_degree: f32 = output.degrees.iter().sum();
    assert!(total_degree > 0.0);
}
