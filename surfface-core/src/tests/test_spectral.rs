//! Comprehensive test suite for surfface spectral GPU implementation.

use crate::backend::{AutoBackend, get_device};
use crate::centroid::CentroidState;
use crate::spectral::laplacian::{LaplacianConfig, LaplacianOutput, LaplacianStage};
use crate::spectral::{
    bridge::compute_tau_mode_gpu, compute_lambdas_gpu, dirichlet_dispersion_gpu,
    laplacian_to_tensor, rayleigh_quotient_gpu,
};
use approx::relative_eq;
use burn::prelude::*;
use sprs::TriMat;

type TestBackend = AutoBackend;

// ═══════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════

fn centroids_from_gaussian_blobs(
    c: usize,
    f: usize,
    noise: f32,
    seed: u64,
) -> CentroidState<TestBackend> {
    use rand::Rng;
    use rand::SeedableRng;
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

// ═══════════════════════════════════════════════════════════════════════
// STAGE C & D INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_stage_c_to_d_pipeline() {
    let state = centroids_from_gaussian_blobs(2, 3, 0.2, 42);

    let config = LaplacianConfig {
        k_neighbors: 2,
        normalize: true, // Use Surfface symmetric normalization
        ..Default::default()
    };
    let stage_c = LaplacianStage::new(config);
    let lap_output = stage_c.execute(&state);

    // Assert Stage C invariants
    assert_eq!(lap_output.n_features, 3);
    assert!(lap_output.nnz > 0);
    // In symmetric normalized L, diagonal should be 1.0 for connected nodes
    let diag_0 = *lap_output.matrix.get(0, 0).unwrap_or(&0.0);
    assert!(relative_eq!(diag_0, 1.0, epsilon = 1e-5));

    // 2. Setup Stage D: GPU Lambda computation
    // Input data for 2 items: item 0 matches feature 0, item 1 is random
    let data = vec![
        1.0, 0.0, 0.0, // Item 0
        0.5, 0.5, 0.5, // Item 1
    ];

    let lambdas = compute_tau_mode_gpu(
        &lap_output,
        &data,
        2, // n_items
        3, // n_features
    );

    assert_eq!(lambdas.len(), 2);
    assert!(lambdas.iter().all(|&l| l.is_finite()));
}

// ═══════════════════════════════════════════════════════════════════════
// UNIT TESTS — Kernel Correctness
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_laplacian_upload_dense_roundtrip() {
    let mut tri = TriMat::new((3, 3));
    tri.add_triplet(0, 0, 2.0);
    tri.add_triplet(0, 1, -1.0);
    tri.add_triplet(1, 1, 1.0);
    let sparse = tri.to_csr();

    let tensor = laplacian_to_tensor(&sparse, 3);
    let back: Vec<f32> = tensor.to_data().to_vec().unwrap();

    assert!((back[0] - 2.0).abs() < 1e-6); // [0,0]
    assert!((back[1] + 1.0).abs() < 1e-6); // [0,1]
}

#[test]
fn test_rayleigh_quotient_eigenvector() {
    let device = get_device();
    // L = [[1, -1], [-1, 1]]
    let l = Tensor::<AutoBackend, 2>::from_data(
        TensorData::new(vec![1.0, -1.0, -1.0, 1.0], Shape::new([2, 2])),
        &device,
    );

    // x = [1, 1] -> eigenvector with eigenvalue 0
    let x = Tensor::<AutoBackend, 2>::from_data(
        TensorData::new(vec![1.0, 1.0], Shape::new([1, 2])),
        &device,
    );

    let e = rayleigh_quotient_gpu(&l, &x);
    let result: Vec<f32> = e.to_data().to_vec().unwrap();

    // R(L, x) = 0
    assert!(result[0].abs() < 1e-5);
}

#[test]
fn test_dirichlet_dispersion_uniform() {
    let device = get_device();
    let l = Tensor::<AutoBackend, 2>::from_data(
        TensorData::new(vec![1.0, -0.5, -0.5, 1.0], Shape::new([2, 2])),
        &device,
    );

    // Items with identical features: zero dispersion
    let x = Tensor::<AutoBackend, 2>::from_data(
        TensorData::new(vec![1.0, 1.0, 1.0, 1.0], Shape::new([2, 2])),
        &device,
    );

    let g = dirichlet_dispersion_gpu(&l, &x, 2, 2);
    let result: Vec<f32> = g.to_data().to_vec().unwrap();

    for &val in &result {
        assert!(val.abs() < 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EDGE CASES
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_unnormalized_mode_compatibility() {
    // Test that setting normalize: false produces L = D - W
    let state = centroids_from_gaussian_blobs(2, 2, 0.3, 42);

    let config = LaplacianConfig {
        normalize: false, // Unnormalized mode
        ..Default::default()
    };
    let stage_c = LaplacianStage::new(config);
    let lap_output = stage_c.execute(&state);

    // In unnormalized mode, diag = degree.
    let diag_0 = *lap_output.matrix.get(0, 0).unwrap();
    let degree_0 = lap_output.degrees[0];
    assert!(relative_eq!(diag_0, degree_0, epsilon = 1e-5));
}

#[test]
fn test_zero_vector_safety() {
    let mut tri = TriMat::new((2, 2));
    tri.add_triplet(0, 0, 1.0);
    tri.add_triplet(1, 1, 1.0);
    let lap = LaplacianOutput {
        matrix: tri.to_csr(),
        n_features: 2,
        nnz: 2,
        degrees: vec![1.0, 1.0],
        sparsity: 0.5,
    };

    let data = vec![0.0, 0.0, 1.0, 1.0];
    let lambdas = compute_tau_mode_gpu(&lap, &data, 2, 2);

    assert!(lambdas.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_compute_lambdas_gpu_integration() {
    crate::tests::init(); // Initialize logging and backend

    // 1. Setup a 3x3 Laplacian: A simple chain graph 0-1-2
    // L = [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
    let mut tri = sprs::TriMat::new((3, 3));
    tri.add_triplet(0, 0, 1.0);
    tri.add_triplet(0, 1, -1.0);
    tri.add_triplet(1, 0, -1.0);
    tri.add_triplet(1, 1, 2.0);
    tri.add_triplet(1, 2, -1.0);
    tri.add_triplet(2, 1, -1.0);
    tri.add_triplet(2, 2, 1.0);
    let sparse_l = tri.to_csr();

    // Upload to GPU
    let l_gpu = laplacian_to_tensor(&sparse_l, 3);

    // 2. Define 2 items with 3 features each
    // Item 0: [1, 1, 1] -> Constant vector (Rayleigh should be 0)
    // Item 1: [1, 0, -1] -> High gradient (Rayleigh should be high)
    let n_items = 2;
    let n_features = 3;
    let item_data = vec![
        1.0, 1.0, 1.0, // Item 0
        1.0, 0.0, -1.0, // Item 1
    ];

    // 3. Compute λ = Rayleigh + Dirichlet on GPU
    let lambdas = compute_lambdas_gpu(&l_gpu, &item_data, n_items, n_features);

    // 4. Assertions
    assert_eq!(lambdas.len(), n_items);

    // Item 0 is constant: L * [1,1,1] = [0,0,0]. Rayleigh = 0.
    // Dirichlet for constant vector is also 0.
    assert!(
        lambdas[0].abs() < 1e-5,
        "Constant vector should have λ ≈ 0, got {}",
        lambdas[0]
    );

    // Item 1 has high variance relative to the graph:
    // Rayleigh(L, [1,0,-1]) = [1,0,-1] * [1,-2,1] / 2 = (1+0-1)/2 = 0?
    // Wait, L * [1,0,-1] = [1, -1-1, 1] = [1, -2, 1].
    // x^T L x = 1(1) + 0(-2) + (-1)(1) = 0.
    // However, Dirichlet dispersion extracts W = max(0, -L).
    // W = [[0, 1, 0], [1, 0, 1], [0, 1, 0]].
    // Dirichlet will be non-zero for [1, 0, -1].
    assert!(
        lambdas[1] > lambdas[0],
        "High gradient vector should have higher λ than constant vector"
    );

    // Ensure all values are finite and not NaN
    for (i, val) in lambdas.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Lambda at index {} is not finite: {}",
            i,
            val
        );
    }
}
