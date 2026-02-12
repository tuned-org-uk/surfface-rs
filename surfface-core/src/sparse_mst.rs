// surfface-core/src/sparse.rs (NEW FILE)
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseGraph {
    pub indptr: Vec<usize>,  // [n_nodes + 1]
    pub indices: Vec<usize>, // [nnz]
    pub data: Vec<f32>,      // [nnz]
    pub n_nodes: usize,
    pub nnz: usize,
}

impl SparseGraph {
    /// Build from COO edge list
    pub fn from_edges(mut edges: Vec<(usize, usize, f32)>, n_nodes: usize) -> Self {
        // Sort by (row, col) for CSR
        edges.sort_by_key(|(r, c, _)| (*r, *c));

        let nnz = edges.len();
        let mut indptr = vec![0; n_nodes + 1];
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);

        for (row, col, weight) in edges {
            indices.push(col);
            data.push(weight);
            indptr[row + 1] += 1;
        }

        // Cumulative sum for indptr
        for i in 0..n_nodes {
            indptr[i + 1] += indptr[i];
        }

        Self {
            indptr,
            indices,
            data,
            n_nodes,
            nnz,
        }
    }

    /// Laplacian: L = D - W
    pub fn to_laplacian(&self) -> Self {
        let mut lap_edges = Vec::new();

        // Compute degrees
        let mut degrees = vec![0.0f32; self.n_nodes];
        for i in 0..self.n_nodes {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            degrees[i] = self.data[start..end].iter().sum();
        }

        // Build Laplacian edges
        for i in 0..self.n_nodes {
            // Diagonal: L_ii = degree_i
            lap_edges.push((i, i, degrees[i]));

            // Off-diagonal: L_ij = -W_ij
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            for k in start..end {
                let j = self.indices[k];
                let w = self.data[k];
                lap_edges.push((i, j, -w));
            }
        }

        Self::from_edges(lap_edges, self.n_nodes)
    }

    pub fn sparsity(&self) -> f64 {
        1.0 - (self.nnz as f64) / (self.n_nodes * self.n_nodes) as f64
    }
}
