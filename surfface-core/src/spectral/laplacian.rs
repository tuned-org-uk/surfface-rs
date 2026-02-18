//! Stage C: Feature-space Laplacian construction.
//!
//! Pipeline position: [C, F] smoothed centroids → L[F, F] symmetric
//! normalised Laplacian.
//!
//! Key difference from ArrowSpace:
//!   ArrowSpace builds the *unnormalized* Laplacian  L = D – W.
//!   Surfface builds the *symmetric normalized* Laplacian
//!       L_sym = I – D^{-½} W D^{-½}
//!   which has eigenvalues in [0, 2] regardless of degree, avoids bias
//!   toward high-cardinality centroids, and is required by Normalized Cut.
//!   Reference: IEEELaplacian2 (Multi-View Spectral Clustering).
//!
//! Wiring kernel: Diagonal Gaussian Bhattacharyya Coefficient (BC).
//!   This replaces the cosine distance used in ArrowSpace and integrates
//!   Kalman-smoothed variances directly into edge weights.
//!   Reference: file:4 (Diagonal Gaussian Bhattacharyya Pipeline).
//!
//! Sparsity: Only the top-k neighbours per feature node are retained
//!   (O(F·k·C) total work), keeping the graph sparse and avoiding the
//!   O(F²·C) blowup for large feature dimensions.
//!
//! ### Key Design Decisions vs ArrowSpace
//! | Concern | ArrowSpace | Surfface Stage C |
//! | :-- | :-- | :-- |
//! | **Wiring kernel** | Cosine distance + Gaussian kernel | Diagonal Gaussian Bhattacharyya BC |
//! | **Variance used?** | ❌ Ignored | ✅ Kalman σ² enters every edge weight |
//! | **Laplacian form** | Unnormalized `L = D − W` | Symmetric normalized `L_sym = I − D^{-½}WD^{-½}` |
//! | **Symmetrization** | `symmetrise_adjancency` via DashMap | Same DashMap pattern, max-symmetrization |
//! | **Sparsity** | `topk` truncation in `build_adjacency` | Top-k BC truncation per feature node |
//! | **Degree bias** | Present (unnormalized λ biased toward high-degree nodes) | Removed by `D^{-½}` scaling |
//!
//! The `normalize: bool` flag on `LaplacianConfig` preserves full backward compatibility — setting it to `false`
//!  produces the exact unnormalized convention ArrowSpace uses, which is useful for debugging and regression tests
//!  against the reference codebase.

use crate::centroid::CentroidState;
use crate::distance::bhattacharyya_coefficient;
use burn::prelude::*;
use dashmap::DashMap;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the feature-space Laplacian stage.
#[derive(Debug, Clone)]
pub struct LaplacianConfig {
    /// Number of nearest neighbours per feature node (sparsity knob).
    /// Higher k → denser graph, more connected manifold.
    pub k_neighbors: usize,

    /// Variance regularisation floor applied inside the Bhattacharyya kernel.
    /// Prevents log(0) and infinite-confidence features from dominating.
    pub variance_regularizer: f32,

    /// If true, build the symmetric normalised Laplacian L_sym = I − D^{-½}WD^{-½}.
    /// If false, build the unnormalised L = D − W (ArrowSpace-compatible mode).
    /// Surfface default is true (normalized).
    pub normalize: bool,

    /// Minimum edge weight to store.  Edges below this threshold are dropped,
    /// preventing near-zero weights from polluting the sparsity structure.
    pub weight_threshold: f32,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 15,
            variance_regularizer: 1e-6,
            normalize: true,
            weight_threshold: 1e-9,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output type
// ─────────────────────────────────────────────────────────────────────────────

/// Output of Stage C.
pub struct LaplacianOutput {
    /// Symmetric (normalized or unnormalized) Laplacian matrix [F, F].
    pub matrix: sprs::CsMat<f32>,

    /// Number of feature nodes F.
    pub n_features: usize,

    /// Total non-zero entries (for sparsity diagnostics).
    pub nnz: usize,

    /// Per-node degree vector D_ii [F] — useful for Stage D Rayleigh quotients.
    pub degrees: Vec<f32>,

    /// Sparsity ratio: fraction of [F×F] that is zero.
    pub sparsity: f32,
}

impl LaplacianOutput {
    pub fn summary(&self) -> String {
        format!(
            "LaplacianOutput: F={}, nnz={}, sparsity={:.2}%, normalized={}",
            self.n_features,
            self.nnz,
            self.sparsity * 100.0,
            true,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage executor
// ─────────────────────────────────────────────────────────────────────────────

/// Stage C executor: builds the feature-space Laplacian from smoothed centroids.
pub struct LaplacianStage {
    pub config: LaplacianConfig,
}

impl LaplacianStage {
    pub fn new(config: LaplacianConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(LaplacianConfig::default())
    }

    /// Execute Stage C.
    ///
    /// Input:  smoothed CentroidState with means [C, F] and variances [C, F].
    /// Output: LaplacianOutput containing L[F, F].
    pub fn execute<B: Backend>(&self, state: &CentroidState<B>) -> LaplacianOutput {
        let device = state.means.device();
        let shape = state.means.dims(); // [C, F]
        let c = shape[0];
        let f = shape[1];

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE C: FEATURE-SPACE LAPLACIAN                     ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!(
            "📐 Transposing [{C}×{F}] centroids → [{F}×{C}] feature profiles",
            C = c,
            F = f
        );
        log::info!(
            "  • k={}, variance_reg={:.2e}, normalize={}",
            self.config.k_neighbors,
            self.config.variance_regularizer,
            self.config.normalize,
        );

        // Pull to CPU for sequential/parallel processing.
        let means_flat: Vec<f32> = state.means.to_data().to_vec().unwrap(); // [C*F]
        let vars_flat: Vec<f32> = state.variances.to_data().to_vec().unwrap(); // [C*F]

        // ── Step 1: Transpose [C, F] → feature profiles ──────────────────────
        // feat_means[i][c] = mean of feature i at centroid c.
        // This is the core "TransposeCentroids" operation from ArrowSpace
        // (GraphFactory::build_laplacian_matrix_from_k_cluster calls
        // clustered.transpose() before passing to build_laplacian_matrix).
        let feat_means: Vec<Vec<f32>> = Self::transpose_to_feature_profiles(&means_flat, c, f);
        let feat_vars: Vec<Vec<f32>> = Self::transpose_to_feature_profiles(&vars_flat, c, f);

        log::debug!(
            "Step 1/3: Transpose complete ({} feature profiles of dim {})",
            f,
            c
        );

        // ── Step 2: Sparse k-NN Bhattacharyya weights ─────────────────────────
        log::debug!(
            "Step 2/3: Computing sparse BC affinities (k={})",
            self.config.k_neighbors
        );
        let edges = self.compute_bhattacharyya_weights(&feat_means, &feat_vars, f);
        log::info!("  • {} directed edges before symmetrization", edges.len());

        // ── Step 3: Symmetrize → degree → Laplacian ───────────────────────────
        log::debug!("Step 3/3: Symmetrizing adjacency and building Laplacian");
        let (matrix_flat, degrees, nnz) =
            self.build_laplacian_flat(&edges, f, self.config.normalize);

        let total = f * f;
        let sparsity = 1.0 - (nnz as f32 / total as f32);

        log::info!(
            "  ✓ Laplacian complete: F={}, nnz={}, sparsity={:.2}%",
            f,
            nnz,
            sparsity * 100.0
        );
        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE C COMPLETE                                     ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");

        // Reconstruct Burn tensor [F, F].
        let matrix = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(matrix_flat, burn::tensor::Shape::new([f, f])),
            &device,
        );

        // One-liner: Tensor<AutoBackend, 2> [F, F]  →  CsMat<f32>
        // Place this immediately after your existing matrix tensor construction.

        let matrix: sprs::CsMat<f32> = {
            let flat: Vec<f32> = matrix.to_data().to_vec().unwrap();
            let mut tri = sprs::TriMat::new((f, f));
            flat.chunks(f).enumerate().for_each(|(r, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, v)| v.abs() > 1e-9)
                    .for_each(|(c, &v)| tri.add_triplet(r, c, v));
            });
            tri.to_csr()
        };

        LaplacianOutput {
            matrix,
            n_features: f,
            nnz,
            degrees,
            sparsity,
        }
    }

    /// ─────────────────────────────────────────────────────────────────────────
    /// Step 1 helper: transpose flat [C*F] row-major → Vec<Vec<f32>> [F][C]
    /// ─────────────────────────────────────────────────────────────────────────
    fn transpose_to_feature_profiles(flat: &[f32], c: usize, f: usize) -> Vec<Vec<f32>> {
        // flat is row-major [C, F]: index = centroid * F + feat
        (0..f)
            .map(|feat| (0..c).map(|cent| flat[cent * f + feat]).collect())
            .collect()
    }

    /// ─────────────────────────────────────────────────────────────────────────
    /// Step 2: Sparse k-NN Bhattacharyya weights
    ///
    /// For each feature i, find its k nearest neighbours by Bhattacharyya
    /// Coefficient (BC).  BC = exp(-DB) ∈ (0, 1]; higher = more similar.
    ///
    /// This mirrors ArrowSpace's build_adjacency (src/laplacian.rs), which
    /// uses CosinePair for approximate k-NN.  Here we use a brute-force O(F²)
    /// scan for correctness; the k-NN candidate pruning keeps the *stored*
    /// edges O(F·k).  When F is large, replace the inner scan with a
    /// FastPair/LSH structure as the next iteration (see file:3).
    ///
    /// Returns: Vec of directed (i, j, weight) triples, unsymmetrized.
    /// ─────────────────────────────────────────────────────────────────────────
    fn compute_bhattacharyya_weights(
        &self,
        feat_means: &[Vec<f32>],
        feat_vars: &[Vec<f32>],
        f: usize,
    ) -> Vec<(usize, usize, f32)> {
        let k = self.config.k_neighbors.min(f.saturating_sub(1));
        let reg = self.config.variance_regularizer;
        let thr = self.config.weight_threshold;

        // Parallel: each feature node i collects its top-k neighbours.
        // DashMap lets us accumulate edges lock-free across threads.
        let all_edges: Vec<(usize, usize, f32)> = (0..f)
            .into_par_iter()
            .flat_map(|i| {
                // Compute BC(i, j) for all j ≠ i.
                let mut scored: Vec<(usize, f32)> = (0..f)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let bc = bhattacharyya_coefficient(
                            &feat_means[i],
                            &feat_vars[i],
                            &feat_means[j],
                            &feat_vars[j],
                            reg,
                        );
                        (j, bc)
                    })
                    .filter(|(_, w)| *w > thr)
                    .collect();

                // Keep only the top-k by descending BC weight.
                // Mirrors ArrowSpace's top-k truncation in build_adjacency.
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);

                scored
                    .into_iter()
                    .map(move |(j, w)| (i, j, w))
                    .collect::<Vec<_>>()
            })
            .collect();

        all_edges
    }

    /// Step 3: Symmetrize → Degree → Laplacian (flat [F*F])
    ///
    /// ArrowSpace pattern (src/laplacian.rs):
    ///   symmetrise_adjancency  →  build_sparse_laplacian  (unnormalized)
    ///
    /// Surfface extension:
    ///   Same symmetrization via DashMap, then either:
    ///     • Unnormalized:  L[i,i] = d_i,  L[i,j] = -w_ij   (ArrowSpace default)
    ///     • Normalized:    L_sym  = I − D^{-½} W D^{-½}
    ///       where L_sym[i,i] = 1,  L_sym[i,j] = -w_ij / sqrt(d_i * d_j)
    ///
    /// Returns (flat [F*F] f32 matrix, degrees[F], nnz_count).
    fn build_laplacian_flat(
        &self,
        edges: &[(usize, usize, f32)],
        f: usize,
        normalize: bool,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let thr = self.config.weight_threshold;

        // 1) Canonical undirected edges: key=(min(i,j), max(i,j)), weight=max
        let undirected: DashMap<(usize, usize), f32> = DashMap::new();
        edges.par_iter().for_each(|&(i, j, w)| {
            if i == j || w <= thr {
                return;
            }
            let key = if i < j { (i, j) } else { (j, i) };
            undirected
                .entry(key)
                .and_modify(|v| *v = v.max(w))
                .or_insert(w);
        });

        // 2) Compute degrees from the final symmetric W
        let mut degrees = vec![0.0f32; f];
        for entry in undirected.iter() {
            let (i, j) = *entry.key();
            let w = *entry.value();
            degrees[i] += w;
            degrees[j] += w;
        }

        // 3) Build Laplacian
        let mut l = vec![0.0f32; f * f];
        let mut nnz = 0usize;

        if normalize {
            // Diagonal: 1 for connected nodes
            for i in 0..f {
                if degrees[i] > thr {
                    l[i * f + i] = 1.0;
                    nnz += 1;
                }
            }

            // Off-diagonal: -w / sqrt(d_i d_j)
            for entry in undirected.iter() {
                let (i, j) = *entry.key();
                let w = *entry.value();

                let di = degrees[i];
                let dj = degrees[j];
                if di <= thr || dj <= thr {
                    continue;
                }

                let v = -w / (di * dj).sqrt();

                // Store symmetric
                l[i * f + j] = v;
                l[j * f + i] = v;
                nnz += 2;
            }
        } else {
            // Unnormalized: diagonal = degree, off-diagonal = -w
            for i in 0..f {
                if degrees[i] > thr {
                    l[i * f + i] = degrees[i];
                    nnz += 1;
                }
            }

            for entry in undirected.iter() {
                let (i, j) = *entry.key();
                let w = *entry.value();

                let v = -w;
                l[i * f + j] = v;
                l[j * f + i] = v;
                nnz += 2;
            }
        }

        (l, degrees, nnz)
    }
}
