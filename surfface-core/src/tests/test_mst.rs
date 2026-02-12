use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::mst::{DistanceMetric, MSTConfig, MSTStage, ThicknessWeight};
use burn::prelude::*;

type TestBackend = AutoBackend;

#[test]
fn test_mst_basic_linear_chain() {
    crate::init();
    let device = Default::default();

    // Create 5 centroids in a line
    let centroids_data = vec![
        0.0, 0.0, // Node 0
        1.0, 0.0, // Node 1
        2.0, 0.0, // Node 2
        3.0, 0.0, // Node 3
        4.0, 0.0, // Node 4
    ];

    let centroids = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(centroids_data, burn::tensor::Shape::new([5, 2])),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([5], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    // Should have C-1 = 4 MST edges
    assert_eq!(output.mst_edges.len(), 4, "MST should have C-1 edges");
    assert_eq!(
        output.centroid_order.len(),
        5,
        "Order should cover all centroids"
    );
    assert_eq!(
        output.thickness.len(),
        5,
        "Thickness should be computed for all centroids"
    );
}

#[test]
fn test_mst_tree_property() {
    crate::init();
    let device = Default::default();

    // Create 10 random centroids
    let centroids = Tensor::<TestBackend, 2>::random(
        [10, 5],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([10], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    // Tree property: |E| = |V| - 1
    assert_eq!(
        output.mst_edges.len(),
        9,
        "MST should have exactly |V|-1 edges"
    );

    // No self-loops
    for edge in &output.mst_edges {
        assert_ne!(edge.u, edge.v, "MST should not contain self-loops");
    }
}

#[test]
fn test_mst_thickness_weighting() {
    crate::init();
    let device = Default::default();

    // Create centroids with varying variances
    let centroids = Tensor::<TestBackend, 2>::ones([3, 4], &device);
    let variances_data = vec![
        0.1, 0.1, 0.1, 0.1, // Low variance (thin)
        1.0, 1.0, 1.0, 1.0, // High variance (thick)
        0.5, 0.5, 0.5, 0.5, // Medium variance
    ];

    let variances = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(variances_data, burn::tensor::Shape::new([3, 4])),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([3], &device);
    let mut state = CentroidState::from_clustering(centroids, counts, 0.1);
    state.variances = variances;

    let config = MSTConfig {
        k_neighbors: 2,
        distance_metric: DistanceMetric::Euclidean,
        thickness_weight: ThicknessWeight::Mean,
        compute_trunk: true,
    };

    let mst_stage = MSTStage::new(config);
    let output = mst_stage.execute(&state);

    // Thickest centroid (index 1, variance=1.0) should be root or in trunk
    assert!(
        output.trunk_nodes.contains(&1) || output.centroid_order[0] == 1,
        "Thickest centroid should be root or in trunk"
    );

    // Verify thickness values
    assert!(output.thickness[1] > output.thickness[0]);
    assert!(output.thickness[1] > output.thickness[2]);
}

#[test]
fn test_mst_different_distance_metrics() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [8, 5],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([8], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    // Test Euclidean
    let config_l2 = MSTConfig {
        distance_metric: DistanceMetric::Euclidean,
        ..Default::default()
    };
    let output_l2 = MSTStage::new(config_l2).execute(&state);
    assert_eq!(
        output_l2.mst_edges.len(),
        7,
        "Euclidean MST should have 7 edges"
    );

    // Test Squared Euclidean
    let config_sq = MSTConfig {
        distance_metric: DistanceMetric::SquaredEuclidean,
        ..Default::default()
    };
    let output_sq = MSTStage::new(config_sq).execute(&state);
    assert_eq!(
        output_sq.mst_edges.len(),
        7,
        "Squared Euclidean MST should have 7 edges"
    );

    // Test Bhattacharyya
    let config_bhatt = MSTConfig {
        distance_metric: DistanceMetric::Bhattacharyya,
        ..Default::default()
    };
    let output_bhatt = MSTStage::new(config_bhatt).execute(&state);
    assert_eq!(
        output_bhatt.mst_edges.len(),
        7,
        "Bhattacharyya MST should have 7 edges"
    );
}

#[test]
fn test_trunk_identification() {
    crate::init();
    let device = Default::default();

    // Create a "star" topology: one central thick node, thin nodes around it
    let centroids = Tensor::<TestBackend, 2>::random(
        [6, 3],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let variances_data = vec![
        5.0, 5.0, 5.0, // Center (very thick)
        0.1, 0.1, 0.1, // Peripheral nodes (thin)
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    ];

    let variances = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(variances_data, burn::tensor::Shape::new([6, 3])),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([6], &device);
    let mut state = CentroidState::from_clustering(centroids, counts, 0.1);
    state.variances = variances;

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    // Thickest node (0) should be in trunk
    assert!(
        output.trunk_nodes.contains(&0),
        "Thickest node should be in trunk"
    );
    assert!(!output.trunk_nodes.is_empty(), "Trunk should not be empty");
}

#[test]
fn test_dfs_ordering_completeness() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [12, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([12], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    // All centroids should appear in ordering exactly once
    assert_eq!(
        output.centroid_order.len(),
        12,
        "Ordering should cover all nodes"
    );

    let mut sorted_order = output.centroid_order.clone();
    sorted_order.sort();
    let expected: Vec<usize> = (0..12).collect();
    assert_eq!(
        sorted_order, expected,
        "Ordering should be a permutation of 0..C"
    );
}

#[test]
fn test_dfs_ordering_determinism() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [8, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([8], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let config = MSTConfig {
        k_neighbors: 4,
        ..Default::default()
    };

    // Run twice
    let output1 = MSTStage::new(config.clone()).execute(&state);
    let output2 = MSTStage::new(config).execute(&state);

    // Should produce same ordering (deterministic)
    assert_eq!(
        output1.centroid_order, output2.centroid_order,
        "MST ordering should be deterministic"
    );
}

#[test]
fn test_thickness_weight_functions() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::ones([4, 3], &device);
    let variances = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(
            vec![
                0.5, 0.5, 0.5, // t=0.5
                1.0, 1.0, 1.0, // t=1.0
                0.2, 0.2, 0.2, // t=0.2
                0.8, 0.8, 0.8, // t=0.8
            ],
            burn::tensor::Shape::new([4, 3]),
        ),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([4], &device);
    let mut state = CentroidState::from_clustering(centroids, counts, 0.1);
    state.variances = variances;

    // Test different weighting functions
    let weights = vec![
        ThicknessWeight::Mean,
        ThicknessWeight::Min,
        ThicknessWeight::Max,
        ThicknessWeight::GeometricMean,
        ThicknessWeight::None,
    ];

    for weight in weights {
        let config = MSTConfig {
            thickness_weight: weight,
            k_neighbors: 3,
            ..Default::default()
        };

        let output = MSTStage::new(config).execute(&state);

        assert_eq!(
            output.mst_edges.len(),
            3,
            "MST should have 3 edges for {:?}",
            weight
        );
        assert!(
            output.total_weight > 0.0,
            "Total weight should be positive for {:?}",
            weight
        );
    }
}

#[test]
fn test_mst_k_neighbors_parameter() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [10, 5],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([10], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    // Test with different k values
    for k in [2, 4, 8] {
        let config = MSTConfig {
            k_neighbors: k,
            ..Default::default()
        };

        let output = MSTStage::new(config).execute(&state);

        // Candidate graph should have ~k edges per node (directed)
        let avg_edges = output.candidate_edges.len() as f32 / 10.0;
        assert!(
            avg_edges >= k as f32 * 0.8 && avg_edges <= k as f32 * 1.2,
            "Average edges per node should be ~{}, got {:.1}",
            k,
            avg_edges
        );

        // MST should still have C-1 edges
        assert_eq!(output.mst_edges.len(), 9, "MST should have 9 edges");
    }
}

#[test]
fn test_mst_config_presets() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [6, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([6], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    // Test default config
    let default_output = MSTStage::with_defaults().execute(&state);
    assert_eq!(default_output.mst_edges.len(), 5);

    // Test high-dimensional config
    let hd_config = MSTConfig::high_dimensional();
    let hd_output = MSTStage::new(hd_config).execute(&state);
    assert_eq!(hd_output.mst_edges.len(), 5);

    // Test prototype config
    let proto_config = MSTConfig::prototype();
    let proto_output = MSTStage::new(proto_config).execute(&state);
    assert_eq!(proto_output.mst_edges.len(), 5);
    assert!(
        proto_output.trunk_nodes.is_empty(),
        "Prototype config should skip trunk computation"
    );
}

#[test]
fn test_mst_edge_properties() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [5, 3],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([5], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    for edge in &output.mst_edges {
        // Edge costs should be positive
        assert!(
            edge.cost > 0.0,
            "Edge cost should be positive, got {}",
            edge.cost
        );

        // Distance should be non-negative
        assert!(
            edge.distance >= 0.0,
            "Distance should be non-negative, got {}",
            edge.distance
        );

        // Thickness should be positive
        assert!(edge.thickness_u > 0.0 && edge.thickness_v > 0.0);

        // Test edge helper methods
        assert!(edge.contains(edge.u));
        assert!(edge.contains(edge.v));
        assert_eq!(edge.other(edge.u), Some(edge.v));
        assert_eq!(edge.other(edge.v), Some(edge.u));
    }
}

#[test]
fn test_mst_total_weight_consistency() {
    crate::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::random(
        [7, 4],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );

    let counts = Tensor::<TestBackend, 1, Int>::ones([7], &device);
    let state = CentroidState::from_clustering(centroids, counts, 0.1);

    let mst_stage = MSTStage::with_defaults();
    let output = mst_stage.execute(&state);

    // Manual sum of edge costs
    let manual_sum: f32 = output.mst_edges.iter().map(|e| e.cost).sum();

    assert!(
        (output.total_weight - manual_sum).abs() < 1e-4,
        "Total weight should match sum of edge costs: {} vs {}",
        output.total_weight,
        manual_sum
    );
}
