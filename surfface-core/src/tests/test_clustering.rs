use crate::backend::AutoBackend;
use crate::clustering::*;
use crate::reduction::{ImplicitProjection, compute_jl_dimension};

type TestBackend = AutoBackend;

#[test]
fn test_jl_projection() {
    crate::tests::init();
    let proj = ImplicitProjection::new(100, 20, Some(42));
    let row = vec![1.0; 100];
    let projected = proj.project(&row);

    assert_eq!(projected.len(), 20);
}

#[test]
fn test_compute_jl_dimension() {
    crate::tests::init();
    let dim = compute_jl_dimension(10000, 50000, 0.3);
    assert!(dim > 32);
    assert!(dim < 50000);
}

#[test]
fn test_clustering_without_projection() {
    crate::tests::init();
    let device = Default::default();

    // Small dimension - no projection
    let data: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..10).map(|_| rand::random()).collect())
        .collect();

    let config = ClusteringConfig {
        max_clusters: 20,
        radius_threshold: 1.0,
        seed: Some(42),
        use_projection: true,
        projection_threshold: 1000,
        jl_epsilon: 0.3,
        min_projected_dim: 64,
    };

    let stage = ClusteringStage::new(config);
    let output = stage.execute_from_vec::<TestBackend>(data, &device);

    assert!(output.state.num_centroids() > 0);
    assert!(output.state.num_centroids() <= 20);
    assert_eq!(output.working_dim, 10); // No projection
    assert!(output.projection.is_none());
}

#[test]
fn test_clustering_with_projection() {
    crate::tests::init();
    let device = Default::default();

    // High dimension - trigger projection
    let data: Vec<Vec<f32>> = (0..500)
        .map(|_| (0..2000).map(|_| rand::random()).collect())
        .collect();

    let config = ClusteringConfig::high_dimensional();
    let stage = ClusteringStage::new(config);
    let output = stage.execute_from_vec::<TestBackend>(data, &device);

    assert!(output.projection.is_some());
    assert!(output.working_dim < output.original_dim);
    assert!(output.working_dim >= 64);

    log::info!(
        "Projected {} → {} dimensions",
        output.original_dim,
        output.working_dim
    );
}

#[test]
fn test_incremental_clustering_deterministic() {
    crate::tests::init();
    let device = Default::default();

    let data: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..10).map(|_| rand::random()).collect())
        .collect();

    let config = ClusteringConfig {
        max_clusters: 15,
        radius_threshold: 1.5,
        seed: Some(42),
        use_projection: false,
        projection_threshold: 1000,
        jl_epsilon: 0.3,
        min_projected_dim: 64,
    };

    let stage1 = ClusteringStage::new(config.clone());
    let output1 = stage1.execute_from_vec::<TestBackend>(data.clone(), &device);

    let stage2 = ClusteringStage::new(config);
    let output2 = stage2.execute_from_vec::<TestBackend>(data, &device);

    assert_eq!(
        output1.state.num_centroids(),
        output2.state.num_centroids(),
        "Should produce same number of centroids"
    );
}
