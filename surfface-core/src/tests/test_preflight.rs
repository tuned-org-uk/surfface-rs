use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::preflight::*;

use burn::tensor::{Int, Tensor};

type TestBackend = AutoBackend;

#[test]
fn test_preflight_no_normalization_needed() {
    crate::init();
    let device = Default::default();

    // Create centroids with similar magnitudes
    let centroids = Tensor::<TestBackend, 2>::ones([5, 10], &device);
    let counts = Tensor::<TestBackend, 1, Int>::ones([5], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let stage = PreflightStage::with_defaults();
    let output = stage.execute(state);

    assert!(
        !output.was_normalized,
        "Should not normalize uniform magnitudes"
    );
    assert!(output.magnitude_stats.variance_ratio < 10.0);
}

#[test]
fn test_preflight_normalization_triggered() {
    crate::init();
    let device = Default::default();

    // Create centroids with 20× magnitude variance
    let centroids_data = vec![
        1.0, 1.0, 1.0, // norm ≈ 1.73
        10.0, 10.0, 10.0, // norm ≈ 17.3 (10× larger)
        20.0, 20.0, 20.0, // norm ≈ 34.6 (20× larger)
    ];

    let centroids = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(centroids_data, burn::tensor::Shape::new([3, 3])),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([3], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let stage = PreflightStage::with_defaults();
    let output = stage.execute(state);

    assert!(output.was_normalized, "Should normalize 20× variance");
    assert!(output.magnitude_stats.variance_ratio > 10.0);

    // Verify normalized centroids have similar norms
    let norms = output
        .normalized_centroids
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt();

    let norms_data = norms.to_data();
    let norms_vec: Vec<f32> = norms_data.to_vec().unwrap();

    for &norm in &norms_vec {
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Normalized centroid should have unit norm, got {}",
            norm
        );
    }
}

#[test]
fn test_zero_variance_detection() {
    crate::init();
    let device = Default::default();

    // Create variances with some near-zero features
    let var_data = vec![
        0.1, 1e-8, 0.2, // Feature 1 is near-zero
        0.15, 1e-7, 0.25, 0.12, 1e-9, 0.22,
    ];

    let variances = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(var_data, burn::tensor::Shape::new([3, 3])),
        &device,
    );

    let centroids = Tensor::<TestBackend, 2>::ones([3, 3], &device);
    let counts = Tensor::<TestBackend, 1, Int>::ones([3], &device);

    let mut state = CentroidState::from_clustering(centroids, counts, 0.1);
    state.variances = variances;

    let stage = PreflightStage::with_defaults();
    let output = stage.execute(state);

    assert!(
        !output.zero_variance_features.is_empty(),
        "Should detect zero-variance features"
    );
    assert!(
        output.zero_variance_features.contains(&1),
        "Feature 1 should be flagged"
    );
}

#[test]
fn test_variance_regularization() {
    crate::init();
    let device = Default::default();

    // Create variances with extreme values
    let var_data = vec![
        1e-10, 0.5, 200.0, // Too small, normal, too large
        1e-9, 0.3, 150.0,
    ];

    let variances = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(var_data, burn::tensor::Shape::new([2, 3])),
        &device,
    );

    let centroids = Tensor::<TestBackend, 2>::ones([2, 3], &device);
    let counts = Tensor::<TestBackend, 1, Int>::ones([2], &device);

    let mut state = CentroidState::from_clustering(centroids, counts, 0.1);
    state.variances = variances;

    let stage = PreflightStage::with_defaults();
    let output = stage.execute(state);

    // Check variances are clamped
    let reg_var_data = output.variances.to_data();
    let reg_var_vec: Vec<f32> = reg_var_data.to_vec().unwrap();

    for &var in &reg_var_vec {
        assert!(var >= 1e-4, "Variance {} should be >= min", var);
        assert!(var <= 100.0, "Variance {} should be <= max", var);
    }
}

#[test]
fn test_conservative_config() {
    let config = PreflightConfig::conservative();
    assert_eq!(config.magnitude_threshold, 5.0);
    assert!(config.variance_epsilon > 1e-4);
}

#[test]
fn test_magnitude_stats_computation() {
    crate::init();
    let device = Default::default();

    let centroids_data = vec![
        3.0, 4.0, // norm = 5.0
        6.0, 8.0, // norm = 10.0
        9.0, 12.0, // norm = 15.0
    ];

    let centroids = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(centroids_data, burn::tensor::Shape::new([3, 2])),
        &device,
    );

    let stage = PreflightStage::with_defaults();
    let stats = stage.compute_magnitude_stats(&centroids);

    assert!((stats.min_norm - 5.0).abs() < 0.01);
    assert!((stats.max_norm - 15.0).abs() < 0.01);
    assert!((stats.mean_norm - 10.0).abs() < 0.01);
    assert!((stats.variance_ratio - 3.0).abs() < 0.01);
}

#[test]
fn test_normalization_preserves_direction() {
    crate::init();
    let device = Default::default();

    let centroids_data = vec![
        3.0, 4.0, // Direction: [0.6, 0.8]
        6.0, 8.0, // Direction: [0.6, 0.8] (same direction, 2× magnitude)
    ];

    let centroids = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(centroids_data, burn::tensor::Shape::new([2, 2])),
        &device,
    );

    let stage = PreflightStage::with_defaults();
    let normalized = stage.normalize_l2(&centroids);

    let norm_data = normalized.to_data();
    let norm_vec: Vec<f32> = norm_data.to_vec().unwrap();

    // Both should normalize to [0.6, 0.8]
    assert!((norm_vec[0] - 0.6).abs() < 0.01);
    assert!((norm_vec[1] - 0.8).abs() < 0.01);
    assert!((norm_vec[2] - 0.6).abs() < 0.01);
    assert!((norm_vec[3] - 0.8).abs() < 0.01);
}
