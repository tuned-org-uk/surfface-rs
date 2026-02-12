// surfface-core/src/matrix.rs
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

pub struct RowMatrix<B: Backend> {
    pub tensor: Tensor<B, 2>,
    pub nrows: usize,
    pub ncols: usize,
}

impl<B: Backend> RowMatrix<B> {
    pub fn new(tensor: Tensor<B, 2>) -> Self {
        let [nrows, ncols] = tensor.dims();
        Self {
            tensor,
            nrows,
            ncols,
        }
    }

    /// Factory to create from raw data (e.g., from Python/Ffi)
    pub fn from_vec(data: Vec<f32>, nrows: usize, ncols: usize, device: &B::Device) -> Self {
        // 1. Create the TensorData directly from your vector and shape
        let data_container = TensorData::new(data, [nrows, ncols]);

        // 2. Initialize the Tensor
        let tensor = Tensor::<B, 2>::from_data(data_container, &device);

        Self {
            tensor,
            nrows,
            ncols,
        }
    }

    /// Physics Invariant: Transpose centroids to feature-nodes [file:2]
    pub fn transpose(&self) -> Self {
        Self::new(self.tensor.clone().transpose())
    }

    /// Parallel row norm (GPU accelerated) [file:2]
    pub fn row_norms(&self) -> Tensor<B, 1> {
        self.tensor
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .flatten(0, 1)
    }
}
