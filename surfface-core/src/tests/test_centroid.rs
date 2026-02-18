use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use kalman_clustering::KalmanClusterer;

use burn::prelude::*;
use burn::tensor::{Int, Shape, Tensor, TensorData};

type TestBackend = AutoBackend;

#[test]
fn test_from_kalman_clusterer() {
    crate::tests::init();
    let device: <AutoBackend as Backend>::Device = Default::default();

    // Create synthetic data as Vec<Vec<f32>>
    let n = 100;
    let f = 10;
    let data_vec: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..f).map(|_| rand::random::<f32>()).collect())
        .collect();

    // Run kalman_clustering
    let mut clusterer = KalmanClusterer::<TestBackend>::new(
        10, // max_k
        n,  // nrows
        device.clone(),
    );

    clusterer.fit(&data_vec);

    // Use convenience method
    let state = CentroidState::from_kalman_clusterer(&clusterer, &device);

    assert!(state.num_centroids() <= 10);
    assert_eq!(state.feature_dim(), 10);

    println!(
        "✓ Created {} centroids from {} items",
        state.num_centroids(),
        n
    );
}

#[test]
fn test_manual_conversion() {
    crate::tests::init();
    let device: <AutoBackend as Backend>::Device = Default::default();

    let data_vec: Vec<Vec<f32>> = (0..50)
        .map(|_| (0..5).map(|_| rand::random::<f32>()).collect())
        .collect();

    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 50, device.clone());
    clusterer.fit(&data_vec);

    // Manual extraction
    let centroids_vec = clusterer.export_centroids();
    let c = centroids_vec.len();
    let f = centroids_vec[0].len();

    println!("Clustering created {} centroids with {} features", c, f);

    // Flatten and create tensor with explicit shape
    let centroids_flat: Vec<f32> = centroids_vec.into_iter().flatten().collect();

    assert_eq!(
        centroids_flat.len(),
        c * f,
        "Flattened data length should match c * f"
    );

    let centroids_tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(centroids_flat, Shape::new([c, f])),
        &device,
    );

    // Get variances
    let variances_data: Vec<f32> = clusterer
        .centroids
        .iter()
        .flat_map(|cent| cent.variance_to_vec())
        .collect();

    assert_eq!(
        variances_data.len(),
        c * f,
        "Variances data length should match c * f"
    );

    let variances_tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(variances_data, Shape::new([c, f])),
        &device,
    );

    // Get assignments
    let assignments_vec = clusterer.assignments.clone();
    let assignments_tensor = Tensor::<TestBackend, 1, Int>::from_ints(
        assignments_vec
            .iter()
            .map(|&x| x.unwrap() as i64)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );

    // Create state
    let state =
        CentroidState::from_clusterer(centroids_tensor, variances_tensor, &assignments_tensor);

    assert_eq!(state.num_centroids(), c);
    assert_eq!(state.feature_dim(), f);
}

#[test]
fn test_thickness() {
    crate::tests::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::ones([5, 10], &device);
    let _variances = Tensor::<TestBackend, 2>::ones([5, 10], &device).mul_scalar(0.2);
    let counts = Tensor::<TestBackend, 1, Int>::ones([5], &device);

    let state = CentroidState::from_clustering(centroids, counts, 0.2);

    let thickness = state.get_thickness();
    assert_eq!(thickness.dims()[0], 5);

    let thickness_data = thickness.to_data();
    let values: Vec<f32> = thickness_data.to_vec().unwrap();
    for val in values {
        assert!((val - 0.2).abs() < 0.01, "Expected ~0.2, got {}", val);
    }
}

#[test]
fn test_feature_space_transpose() {
    crate::tests::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::ones([10, 100], &device);
    let counts = Tensor::<TestBackend, 1, Int>::ones([10], &device);

    let state = CentroidState::from_clustering(centroids, counts, 0.1);
    let features = state.to_feature_nodes();

    // Should transpose to [F=100, C=10]
    assert_eq!(features.dims(), [100, 10]);
}

#[test]
fn test_regularize_variances() {
    crate::tests::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::ones([3, 5], &device);
    let variances = Tensor::<TestBackend, 2>::ones([3, 5], &device).mul_scalar(0.01);
    let assignments = Tensor::<TestBackend, 1, Int>::zeros([10], &device);

    let mut state = CentroidState::from_clusterer(centroids, variances, &assignments);

    state.regularize_variances(1e-4, 0.05, 10.0);

    let var_data = state.variances.to_data();
    let values: Vec<f32> = var_data.to_vec().unwrap();
    for val in values {
        assert!(val >= 0.05, "Variance should be >= 0.05, got {}", val);
    }
}

#[test]
fn test_get_feature_variances() {
    crate::tests::init();
    let device = Default::default();

    let centroids = Tensor::<TestBackend, 2>::ones([5, 10], &device);
    let _variances = Tensor::<TestBackend, 2>::ones([5, 10], &device).mul_scalar(0.3);
    let counts = Tensor::<TestBackend, 1, Int>::ones([5], &device);

    let state = CentroidState::from_clustering(centroids, counts, 0.3);
    let feature_vars = state.get_feature_variances();

    assert_eq!(feature_vars.dims(), [10, 5]);
}
