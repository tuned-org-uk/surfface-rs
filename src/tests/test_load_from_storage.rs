use crate::{
    core::ArrowSpace,
    graph::{GraphLaplacian, GraphParams},
    storage::parquet::{save_dense_matrix, save_lambda, save_sparse_matrix},
    // taumode::TauMode,
};
use approx::assert_relative_eq;
use approx::relative_eq;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::TriMat;
use std::path::Path;
use tempfile::TempDir;

/// Helper: Create test data and save to storage
fn setup_test_storage(storage_dir: &Path, dataset_name: &str) -> (usize, usize, Vec<f64>) {
    // Create test data: 50 items × 10 features
    let nitems = 50;
    let nfeatures = 10;

    let data: Vec<Vec<f64>> = (0..nitems)
        .map(|i| {
            (0..nfeatures)
                .map(|j| ((i * nfeatures + j) as f64) * 0.1)
                .collect()
        })
        .collect();

    // Create matrices
    let data_matrix = DenseMatrix::from_2d_vec(&data).unwrap();

    // Create lambdas (1-row matrix)
    let lambdas: Vec<f64> = (0..nitems).map(|i| (i as f64) * 0.05 + 0.1).collect();

    // Create sparse GL matrix (simple diagonal + off-diagonal structure)
    let mut trimat = TriMat::new((nitems, nitems));
    for i in 0..nitems {
        trimat.add_triplet(i, i, 2.0); // Diagonal
        if i > 0 {
            trimat.add_triplet(i, i - 1, -0.5); // Off-diagonal
        }
        if i < nitems - 1 {
            trimat.add_triplet(i, i + 1, -0.5); // Off-diagonal
        }
    }
    let gl_matrix = trimat.to_csr();

    // Create clustered-dm (same as data_matrix for simplicity)
    let clustered_matrix = data_matrix.clone();

    // Save all files
    save_dense_matrix(
        &data_matrix,
        storage_dir,
        &format!("{}-raw_input", dataset_name),
        None,
    )
    .unwrap();

    save_lambda(
        &lambdas,
        storage_dir,
        &format!("{}-lambdas", dataset_name),
        None,
    )
    .unwrap();

    save_sparse_matrix(
        &gl_matrix,
        storage_dir,
        &format!("{}-gl-matrix", dataset_name),
        None,
    )
    .unwrap();

    save_dense_matrix(
        &clustered_matrix,
        storage_dir,
        &format!("{}-clustered-dm", dataset_name),
        None,
    )
    .unwrap();

    (nitems, nfeatures, lambdas)
}

#[test]
fn test_arrowspace_new_from_storage_basic() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "test_dataset";

    let (expected_nitems, expected_nfeatures, expected_lambdas) =
        setup_test_storage(tempdir.path(), dataset_name);

    // Load ArrowSpace
    let aspace = ArrowSpace::new_from_storage(tempdir.path(), dataset_name)
        .expect("Failed to load ArrowSpace");

    // Verify dimensions
    assert_eq!(aspace.nitems, expected_nitems);
    assert_eq!(aspace.nfeatures, expected_nfeatures);

    // Verify lambdas
    assert_eq!(aspace.lambdas.len(), expected_lambdas.len());
    for (i, (&loaded, &expected)) in aspace
        .lambdas
        .iter()
        .zip(expected_lambdas.iter())
        .enumerate()
    {
        assert!(
            relative_eq!(loaded, expected, epsilon = 1e-10),
            "Lambda mismatch at index {}",
            i
        );
    }

    // Verify lambda statistics are computed
    assert!(aspace.min_lambdas > 0.0);
    assert!(aspace.max_lambdas > aspace.min_lambdas);
    assert_relative_eq!(
        aspace.range_lambdas,
        aspace.max_lambdas - aspace.min_lambdas,
        epsilon = 1e-10
    );

    // Verify sorted index is built
    assert_eq!(aspace.lambdas_sorted.to_vec().len(), expected_nitems);
}

#[test]
fn test_graphlaplacian_new_from_storage_basic() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "test_gl";

    let (expected_nitems, _, _) = setup_test_storage(tempdir.path(), dataset_name);

    let params = GraphParams {
        eps: 0.5,
        k: 10,
        topk: 3,
        p: 2.0,
        sigma: None,
        sparsity_check: false,
        normalise: false,
    };

    // Load GraphLaplacian
    let gl = GraphLaplacian::new_from_storage(tempdir.path(), dataset_name, params, false)
        .expect("Failed to load GraphLaplacian");

    // Verify dimensions
    assert_eq!(gl.nnodes, expected_nitems);
    assert_eq!(gl.matrix.rows(), expected_nitems);
    assert_eq!(gl.matrix.cols(), expected_nitems);

    // Verify init_data is loaded
    assert_eq!(gl.init_data.shape(), (expected_nitems, 10));

    // Verify graph params
    assert_relative_eq!(gl.graph_params.eps, 0.5, epsilon = 1e-10);
    assert_eq!(gl.graph_params.k, 10);
}

#[test]
fn test_roundtrip_arrowspace_and_gl() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "roundtrip_test";

    setup_test_storage(tempdir.path(), dataset_name);

    let params = GraphParams {
        eps: 0.3,
        k: 8,
        topk: 2,
        p: 2.0,
        sigma: Some(0.5),
        sparsity_check: false,
        normalise: false,
    };

    // Load both
    let aspace = ArrowSpace::new_from_storage(tempdir.path(), dataset_name)
        .expect("Failed to load ArrowSpace");

    let gl = GraphLaplacian::new_from_storage(tempdir.path(), dataset_name, params.clone(), false)
        .expect("Failed to load GraphLaplacian");

    // Verify consistency
    assert_eq!(
        aspace.nitems, gl.nnodes,
        "ArrowSpace items should match GL nodes"
    );

    // Verify data matrix matches init_data
    assert_eq!(aspace.data.shape(), gl.init_data.shape());

    // Verify we can access items
    for i in 0..aspace.nitems {
        let item = aspace.get_item(i);
        assert_eq!(item.item.len(), aspace.nfeatures);
        assert!(item.lambda.is_finite());
    }
}

#[test]
fn test_arrowspace_sorted_index_correctness() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "sorted_index_test";

    let (nitems, _, lambdas) = setup_test_storage(tempdir.path(), dataset_name);

    let aspace = ArrowSpace::new_from_storage(tempdir.path(), dataset_name)
        .expect("Failed to load ArrowSpace");

    // Verify sorted index is correct
    let mut sorted_lambdas: Vec<f64> = lambdas.clone();
    sorted_lambdas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Check that sorted index returns lambdas in ascending order
    let mut prev_lambda = f64::NEG_INFINITY;
    let lambdas_vec = aspace.lambdas_sorted.to_vec();
    for i in 0..nitems {
        let (lambda, _idx) = lambdas_vec[i];
        assert!(
            lambda >= prev_lambda,
            "Sorted index not in ascending order at {}",
            i
        );
        prev_lambda = lambda;
    }
}

#[test]
fn test_lambda_count_mismatch() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "mismatch_test";

    let nitems = 50;
    let nfeatures = 10;

    // Create data
    let data: Vec<Vec<f64>> = (0..nitems)
        .map(|i| (0..nfeatures).map(|j| (i * j) as f64).collect())
        .collect();
    let data_matrix = DenseMatrix::from_2d_vec(&data).unwrap();

    // Create lambdas with WRONG count
    let wrong_lambdas: Vec<f64> = (0..nitems - 5).map(|i| i as f64).collect();

    // Save files
    save_dense_matrix(
        &data_matrix,
        tempdir.path(),
        &format!("{}-raw_input", dataset_name),
        None,
    )
    .unwrap();
    save_lambda(
        &wrong_lambdas,
        tempdir.path(),
        &format!("{}-lambdas", dataset_name),
        None,
    )
    .unwrap();

    // Try to load - should fail
    let result = ArrowSpace::new_from_storage(tempdir.path(), dataset_name);
    assert!(
        result.is_err(),
        "Should fail when lambda count doesn't match items"
    );

    if let Err(e) = result {
        assert!(
            e.to_string().contains("doesn't match"),
            "Error should mention mismatch"
        );
    }
}

#[test]
fn test_graph_laplacian_matrix_structure() {
    let tempdir = TempDir::new().unwrap();
    let dataset_name = "gl_structure";

    setup_test_storage(tempdir.path(), dataset_name);

    let params = GraphParams {
        eps: 0.5,
        k: 10,
        topk: 3,
        p: 2.0,
        sigma: None,
        sparsity_check: false,
        normalise: false,
    };

    let gl = GraphLaplacian::new_from_storage(tempdir.path(), dataset_name, params, false)
        .expect("Failed to load GraphLaplacian");

    // Verify matrix is square
    let (rows, cols) = gl.matrix.shape();
    assert_eq!(rows, cols, "GL matrix should be square");
    assert_eq!(rows, gl.nnodes);

    // Verify diagonal elements exist and are positive (Laplacian property)
    for i in 0..gl.nnodes {
        let diag = gl.matrix.get(i, i).copied().unwrap_or(0.0);
        assert!(diag > 0.0, "Diagonal element {} should be positive", i);
    }

    // Verify matrix is sparse
    let nnz = gl.matrix.nnz();
    let total = gl.nnodes * gl.nnodes;
    let sparsity = 1.0 - (nnz as f64 / total as f64);
    assert!(sparsity > 0.5, "Matrix should be reasonably sparse");
}

#[test]
fn test_multiple_datasets_same_directory() {
    let tempdir = TempDir::new().unwrap();

    let dataset1 = "dataset_a";
    let dataset2 = "dataset_b";

    setup_test_storage(tempdir.path(), dataset1);
    setup_test_storage(tempdir.path(), dataset2);

    // Load both independently
    let aspace1 =
        ArrowSpace::new_from_storage(tempdir.path(), dataset1).expect("Failed to load dataset1");

    let aspace2 =
        ArrowSpace::new_from_storage(tempdir.path(), dataset2).expect("Failed to load dataset2");

    // Verify both are valid and independent
    assert_eq!(aspace1.nitems, 50);
    assert_eq!(aspace2.nitems, 50);
    assert_eq!(aspace1.nfeatures, 10);
    assert_eq!(aspace2.nfeatures, 10);
}
