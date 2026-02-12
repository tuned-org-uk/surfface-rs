// surfface-core/src/tests/test_distance.rs

use crate::backend::AutoBackend;
use crate::distance::*;
use burn::prelude::*;

type TestBackend = AutoBackend;

#[test]
fn test_bhattacharyya_identical_distributions() {
    crate::init();
    let device = Default::default();

    let mean = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let var = Tensor::<TestBackend, 1>::from_floats([0.5, 0.5, 0.5], &device);

    let distance = bhattacharyya_diagonal(mean.clone(), var.clone(), mean, var);

    let dist_val: f32 = distance.into_scalar();
    assert!(
        dist_val < 1e-5,
        "Distance between identical distributions should be ~0, got {}",
        dist_val
    );
}

#[test]
fn test_bhattacharyya_different_means() {
    crate::init();
    let device = Default::default();

    let mean_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
    let mean_j = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
    let var = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);

    let distance = bhattacharyya_diagonal(mean_i, var.clone(), mean_j, var);

    let dist_val: f32 = distance.into_scalar();
    assert!(dist_val > 0.0, "Distance should be positive");
}

#[test]
fn test_bhattacharyya_different_variances() {
    crate::init();
    let device = Default::default();

    let mean = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
    let var_i = Tensor::<TestBackend, 1>::from_floats([0.5, 0.5], &device);
    let var_j = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0], &device);

    let distance = bhattacharyya_diagonal(mean.clone(), var_i, mean, var_j);

    let dist_val: f32 = distance.into_scalar();
    assert!(
        dist_val > 0.0,
        "Distance should be positive for different variances"
    );
}

#[test]
fn test_bhattacharyya_slice_vs_tensor() {
    crate::init();
    let device = Default::default();

    let mean_i_vec = vec![1.0, 2.0, 3.0];
    let mean_j_vec = vec![1.5, 2.5, 3.5];
    let var_i_vec = vec![0.5, 0.5, 0.5];
    let var_j_vec = vec![0.6, 0.6, 0.6];

    // Slice version
    let dist_slice =
        bhattacharyya_distance_diagonal(&mean_i_vec, &var_i_vec, &mean_j_vec, &var_j_vec);

    // Tensor version
    let mean_i = Tensor::<TestBackend, 1>::from_floats(mean_i_vec.as_slice(), &device);
    let mean_j = Tensor::<TestBackend, 1>::from_floats(mean_j_vec.as_slice(), &device);
    let var_i = Tensor::<TestBackend, 1>::from_floats(var_i_vec.as_slice(), &device);
    let var_j = Tensor::<TestBackend, 1>::from_floats(var_j_vec.as_slice(), &device);

    let dist_tensor: f32 = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j).into_scalar();

    assert!(
        (dist_slice - dist_tensor).abs() < 1e-4,
        "Slice and tensor versions should match: {} vs {}",
        dist_slice,
        dist_tensor
    );
}

#[test]
fn test_bhattacharyya_affinity() {
    crate::init();
    let device = Default::default();

    // Identical distributions should have affinity ~1
    let mean_i = Tensor::<TestBackend, 1>::zeros([3], &device);
    let mean_j = Tensor::<TestBackend, 1>::zeros([3], &device);
    let var = Tensor::<TestBackend, 1>::ones([3], &device);

    let affinity = bhattacharyya_affinity(mean_i, var.clone(), mean_j, var);
    let aff_val: f32 = affinity.into_scalar();

    assert!(
        aff_val > 0.99,
        "Affinity between identical distributions should be ~1, got {}",
        aff_val
    );
}

#[test]
fn test_bhattacharyya_affinity_decay() {
    crate::init();
    let device = Default::default();

    let var = Tensor::<TestBackend, 1>::ones([2], &device);

    // Test affinity decreases with distance
    let mean_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);

    let mean_close = Tensor::<TestBackend, 1>::from_floats([0.5, 0.5], &device);
    let mean_far = Tensor::<TestBackend, 1>::from_floats([5.0, 5.0], &device);

    let aff_close: f32 =
        bhattacharyya_affinity(mean_i.clone(), var.clone(), mean_close, var.clone()).into_scalar();
    let aff_far: f32 = bhattacharyya_affinity(mean_i, var.clone(), mean_far, var).into_scalar();

    assert!(
        aff_close > aff_far,
        "Affinity should decay with distance: close={}, far={}",
        aff_close,
        aff_far
    );
}

#[test]
fn test_bhattacharyya_pairwise_shape() {
    crate::init();
    let device = Default::default();

    // 4 features, 3 centroids
    let features = Tensor::<TestBackend, 2>::random(
        [4, 3],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let variances = Tensor::<TestBackend, 2>::ones([4, 3], &device).mul_scalar(0.5);

    let distances = bhattacharyya_pairwise(features, variances);

    assert_eq!(distances.dims(), [4, 4], "Output should be [F, F]");
}

#[test]
fn test_bhattacharyya_pairwise_diagonal() {
    crate::init();
    let device = Default::default();

    let features = Tensor::<TestBackend, 2>::random(
        [5, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let variances = Tensor::<TestBackend, 2>::ones([5, 4], &device).mul_scalar(0.5);

    let distances = bhattacharyya_pairwise(features, variances);

    // Diagonal should be ~0 (self-distance)
    let diag_data = distances.to_data();
    let diag_vec: Vec<f32> = diag_data.to_vec().unwrap();

    for i in 0..5 {
        let self_dist = diag_vec[i * 5 + i];
        assert!(
            self_dist < 1e-4,
            "Self-distance at {} should be ~0, got {}",
            i,
            self_dist
        );
    }
}

#[test]
fn test_bhattacharyya_pairwise_symmetry() {
    crate::init();
    let device = Default::default();

    let features = Tensor::<TestBackend, 2>::random(
        [3, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let variances = Tensor::<TestBackend, 2>::ones([3, 4], &device).mul_scalar(0.5);

    let distances = bhattacharyya_pairwise(features, variances);

    let data = distances.to_data();
    let vec: Vec<f32> = data.to_vec().unwrap();

    // Check symmetry: D[i,j] ≈ D[j,i]
    for i in 0..3 {
        for j in 0..3 {
            let d_ij = vec[i * 3 + j];
            let d_ji = vec[j * 3 + i];
            assert!(
                (d_ij - d_ji).abs() < 1e-4,
                "Distance matrix should be symmetric: D[{},{}]={}, D[{},{}]={}",
                i,
                j,
                d_ij,
                j,
                i,
                d_ji
            );
        }
    }
}

#[test]
fn test_euclidean_distance_3_4_5_triangle() {
    crate::init();
    let device = Default::default();

    let vec_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
    let vec_j = Tensor::<TestBackend, 1>::from_floats([3.0, 4.0], &device);

    let dist: f32 = euclidean_distance(vec_i, vec_j).into_scalar();

    assert!(
        (dist - 5.0).abs() < 1e-5,
        "3-4-5 triangle: expected 5.0, got {}",
        dist
    );
}

#[test]
fn test_squared_euclidean_distance() {
    crate::init();
    let device = Default::default();

    let vec_i = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let vec_j = Tensor::<TestBackend, 1>::from_floats([4.0, 5.0, 6.0], &device);

    let sq_dist: f32 = squared_euclidean_distance(vec_i, vec_j).into_scalar();

    // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
    assert!(
        (sq_dist - 27.0).abs() < 1e-5,
        "Expected 27.0, got {}",
        sq_dist
    );
}

#[test]
fn test_euclidean_slice() {
    crate::init();

    let vec_i = vec![0.0, 0.0];
    let vec_j = vec![3.0, 4.0];

    let dist = euclidean_distance_slice(&vec_i, &vec_j);

    assert!((dist - 5.0).abs() < 1e-5, "Expected 5.0, got {}", dist);
}

#[test]
fn test_cosine_similarity_parallel_vectors() {
    crate::init();
    let device = Default::default();

    // Parallel vectors
    let vec_i = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
    let vec_j = Tensor::<TestBackend, 1>::from_floats([2.0, 2.0], &device);

    let sim: f32 = cosine_similarity(vec_i, vec_j).into_scalar();

    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Parallel vectors should have cos=1, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    crate::init();
    let device = Default::default();

    // Orthogonal vectors
    let vec_a = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
    let vec_b = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0], &device);

    let sim: f32 = cosine_similarity(vec_a, vec_b).into_scalar();

    assert!(
        sim.abs() < 1e-5,
        "Orthogonal vectors should have cos=0, got {}",
        sim
    );
}

#[test]
fn test_cosine_distance() {
    crate::init();
    let device = Default::default();

    let vec_i = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
    let vec_j = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0], &device);

    let dist: f32 = cosine_distance(vec_i, vec_j).into_scalar();

    // Orthogonal: cos=0, distance=1
    assert!(
        (dist - 1.0).abs() < 1e-5,
        "Expected distance=1, got {}",
        dist
    );
}

#[test]
fn test_numerical_stability_tiny_variances() {
    crate::init();
    let device = Default::default();

    // Very small variances (near-zero)
    let mean_i = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0], &device);
    let mean_j = Tensor::<TestBackend, 1>::from_floats([1.1, 2.1], &device);
    let var_i = Tensor::<TestBackend, 1>::from_floats([1e-12, 1e-12], &device);
    let var_j = Tensor::<TestBackend, 1>::from_floats([1e-12, 1e-12], &device);

    let distance = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j);
    let dist_val: f32 = distance.into_scalar();

    assert!(
        dist_val.is_finite(),
        "Distance should be finite even with tiny variances, got {}",
        dist_val
    );
}

#[test]
fn test_numerical_stability_large_variances() {
    crate::init();
    let device = Default::default();

    // Large variances
    let mean_i = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
    let mean_j = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);
    let var_i = Tensor::<TestBackend, 1>::from_floats([100.0, 100.0], &device);
    let var_j = Tensor::<TestBackend, 1>::from_floats([100.0, 100.0], &device);

    let distance = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j);
    let dist_val: f32 = distance.into_scalar();

    assert!(
        dist_val.is_finite() && dist_val >= 0.0,
        "Distance should be finite and non-negative, got {}",
        dist_val
    );
}

// Replace the failing test with these three tests:

#[test]
fn test_bhattacharyya_not_a_metric() {
    crate::init();
    let device = Default::default();

    // Bhattacharyya distance does NOT satisfy triangle inequality
    // This is a known property - it's a divergence, not a metric

    let mean_a = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
    let mean_b = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
    let mean_c = Tensor::<TestBackend, 1>::from_floats([2.0, 0.0], &device);
    let var = Tensor::<TestBackend, 1>::ones([2], &device);

    let d_ab: f32 =
        bhattacharyya_diagonal(mean_a.clone(), var.clone(), mean_b.clone(), var.clone())
            .into_scalar();
    let d_bc: f32 =
        bhattacharyya_diagonal(mean_b.clone(), var.clone(), mean_c.clone(), var.clone())
            .into_scalar();
    let d_ac: f32 = bhattacharyya_diagonal(mean_a, var.clone(), mean_c, var).into_scalar();

    println!(
        "Bhattacharyya distances: d_ab={:.4}, d_bc={:.4}, d_ac={:.4}, sum={:.4}",
        d_ab,
        d_bc,
        d_ac,
        d_ab + d_bc
    );

    // Verify distances are positive and finite
    assert!(
        d_ab > 0.0 && d_ab.is_finite(),
        "d_ab should be positive and finite"
    );
    assert!(
        d_bc > 0.0 && d_bc.is_finite(),
        "d_bc should be positive and finite"
    );
    assert!(
        d_ac > 0.0 && d_ac.is_finite(),
        "d_ac should be positive and finite"
    );

    // Note: Bhattacharyya is a divergence, not a metric
    // It does NOT necessarily satisfy triangle inequality
    // This is expected and acceptable for our use case
}

#[test]
fn test_bhattacharyya_symmetry() {
    crate::init();
    let device = Default::default();

    // Bhattacharyya distance IS symmetric: D(P||Q) = D(Q||P)
    let mean_i = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0], &device);
    let mean_j = Tensor::<TestBackend, 1>::from_floats([1.5, 2.5, 3.5], &device);
    let var_i = Tensor::<TestBackend, 1>::from_floats([0.5, 0.6, 0.7], &device);
    let var_j = Tensor::<TestBackend, 1>::from_floats([0.8, 0.9, 1.0], &device);

    let d_ij: f32 =
        bhattacharyya_diagonal(mean_i.clone(), var_i.clone(), mean_j.clone(), var_j.clone())
            .into_scalar();

    let d_ji: f32 = bhattacharyya_diagonal(mean_j, var_j, mean_i, var_i).into_scalar();

    assert!(
        (d_ij - d_ji).abs() < 1e-5,
        "Bhattacharyya should be symmetric: D(i,j)={:.6} vs D(j,i)={:.6}",
        d_ij,
        d_ji
    );
}

#[test]
fn test_bhattacharyya_increases_with_separation() {
    crate::init();
    let device = Default::default();

    // Distance should increase as distributions become more separated
    let var = Tensor::<TestBackend, 1>::ones([2], &device);
    let mean_ref = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);

    let separations = vec![0.5, 1.0, 2.0, 5.0];
    let mut distances = Vec::new();

    for sep in &separations {
        let mean_test = Tensor::<TestBackend, 1>::from_floats([*sep, 0.0], &device);
        let dist: f32 =
            bhattacharyya_diagonal(mean_ref.clone(), var.clone(), mean_test, var.clone())
                .into_scalar();
        distances.push(dist);
    }

    // Verify monotonic increase
    for i in 1..distances.len() {
        assert!(
            distances[i] > distances[i - 1],
            "Distance should increase with separation: d[{}]={:.4} <= d[{}]={:.4}",
            i - 1,
            distances[i - 1],
            i,
            distances[i]
        );
    }
}
