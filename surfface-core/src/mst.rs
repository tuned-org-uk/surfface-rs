//! MST Skeleton stage: Build thickness-weighted transport network
//!
//! This is Stage B1 of the Surfface pipeline [file:4]:
//! - Build sparse k-NN candidate graph between centroids
//! - Compute thickness-weighted edge costs c_ij = d_ij * φ(t_i, t_j)
//! - Extract MST using Prim's algorithm
//! - Identify trunk (tree diameter) and produce 1D ordering
//!
//! The MST skeleton influences centroid state regularization (Stage B2)
//! but does NOT merge into the statistical Laplacian (Stage C) [file:2]

use crate::centroid::CentroidState;
use crate::distance::bhattacharyya_distance_diagonal;
/// Input: CentroidState [C, F] with variances
///
/// Step 1: Compute thickness proxy
///   t_i = mean(variance[i, :])  # Average variance per centroid
///
/// Step 2: Build sparse candidate graph
///   For each centroid i:
///     Find k nearest neighbors using Bhattacharyya distance
///     Add edge (i, j, d_ij, t_i, t_j)
///
/// Step 3: Compute edge costs
///   c_ij = d_ij * (t_i + t_j) / 2
///
/// Step 4: Run MST (Prim or Kruskal)
///   MST = minimum_spanning_tree(candidate_graph, costs=c)
///
/// Step 5: Trunk-sprouts traversal
///   root = argmax(t_i)                    # Thickest centroid
///   trunk = tree_diameter(MST, costs=c)   # Longest path
///   order = dfs(MST, root, sort_children_by=thickness, descending=True)
///
/// Output:
///   - centroid_order: Vec<usize>  [C]
///   - mst_edges: Vec<(usize, usize, f32)>
///   - trunk_nodes: Vec<usize>
use burn::prelude::*;
use std::collections::{BinaryHeap, VecDeque};

/// Configuration for MST skeleton construction
#[derive(Debug, Clone)]
pub struct MSTConfig {
    /// Number of nearest neighbors for candidate graph (default: 8)
    pub k_neighbors: usize,

    /// Distance metric for edges
    pub distance_metric: DistanceMetric,

    /// Thickness weighting function
    pub thickness_weight: ThicknessWeight,

    /// Enable trunk identification via tree diameter
    pub compute_trunk: bool,
}

/// Distance metric options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Bhattacharyya distance (diagonal Gaussian) [file:3]
    Bhattacharyya,

    /// Euclidean L2 distance
    Euclidean,

    /// Squared Euclidean (no sqrt)
    SquaredEuclidean,
}

/// Thickness weighting function φ(t_i, t_j)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThicknessWeight {
    /// Mean: (t_i + t_j) / 2
    Mean,

    /// Minimum: min(t_i, t_j)
    Min,

    /// Maximum: max(t_i, t_j)
    Max,

    /// Geometric mean: sqrt(t_i * t_j)
    GeometricMean,

    /// No weighting: 1.0 (pure distance)
    None,
}

impl Default for MSTConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            distance_metric: DistanceMetric::Bhattacharyya,
            thickness_weight: ThicknessWeight::Mean,
            compute_trunk: true,
        }
    }
}

impl MSTConfig {
    /// Config for high-dimensional data (more neighbors)
    pub fn high_dimensional() -> Self {
        Self {
            k_neighbors: 16,
            distance_metric: DistanceMetric::Bhattacharyya,
            thickness_weight: ThicknessWeight::Mean,
            compute_trunk: true,
        }
    }

    /// Config for quick prototyping (fewer neighbors)
    pub fn prototype() -> Self {
        Self {
            k_neighbors: 4,
            distance_metric: DistanceMetric::SquaredEuclidean,
            thickness_weight: ThicknessWeight::None,
            compute_trunk: false,
        }
    }
}

/// Edge in the candidate or MST graph
#[derive(Debug, Clone)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub distance: f32,
    pub thickness_u: f32,
    pub thickness_v: f32,
    pub cost: f32,
}

impl Edge {
    /// Check if edge connects node i
    pub fn contains(&self, i: usize) -> bool {
        self.u == i || self.v == i
    }

    /// Get the other endpoint
    pub fn other(&self, i: usize) -> Option<usize> {
        if self.u == i {
            Some(self.v)
        } else if self.v == i {
            Some(self.u)
        } else {
            None
        }
    }
}

/// Output of the MST skeleton stage
pub struct MSTOutput {
    /// Candidate graph edges (k-NN)
    pub candidate_edges: Vec<Edge>,

    /// MST edges (C-1 edges forming tree)
    pub mst_edges: Vec<Edge>,

    /// 1D centroid ordering (trunk-sprouts DFS)
    pub centroid_order: Vec<usize>,

    /// Trunk node indices (longest path in tree)
    pub trunk_nodes: Vec<usize>,

    /// Thickness per centroid
    pub thickness: Vec<f32>,

    /// Total MST weight (sum of edge costs)
    pub total_weight: f32,
}

impl MSTOutput {
    pub fn summary(&self) -> String {
        format!(
            "MST: {} edges, total_weight={:.2}, trunk_len={}, order_len={}",
            self.mst_edges.len(),
            self.total_weight,
            self.trunk_nodes.len(),
            self.centroid_order.len()
        )
    }
}

/// MST skeleton stage executor
pub struct MSTStage {
    config: MSTConfig,
}

impl MSTStage {
    pub fn new(config: MSTConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(MSTConfig::default())
    }

    /// Execute MST skeleton construction
    pub fn execute<B: Backend>(&self, state: &CentroidState<B>) -> MSTOutput {
        let c = state.num_centroids();

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE B1: MST SKELETON                               ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!(
            "🌲 Building MST for {} centroids (k={})",
            c,
            self.config.k_neighbors
        );

        // STEP 1: Compute thickness proxy
        log::debug!("Step 1/5: Computing thickness...");
        let thickness = self.compute_thickness(state);
        log::info!(
            "  ✓ Thickness: min={:.4}, max={:.4}, mean={:.4}",
            thickness.iter().cloned().fold(f32::INFINITY, f32::min),
            thickness.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            thickness.iter().sum::<f32>() / thickness.len() as f32
        );

        // STEP 2: Build sparse candidate graph (k-NN)
        log::debug!("Step 2/5: Building k-NN candidate graph...");
        let candidate_edges = self.build_candidate_graph(state, &thickness);
        log::info!(
            "  ✓ Candidate graph: {} edges ({:.2} edges/centroid)",
            candidate_edges.len(),
            candidate_edges.len() as f32 / c as f32
        );

        // STEP 3: Extract MST using Prim's algorithm
        log::debug!("Step 3/5: Running Prim's MST...");
        let (mst_edges, total_weight) = self.prim_mst(&candidate_edges, c);
        log::info!(
            "  ✓ MST: {} edges, total_weight={:.2}",
            mst_edges.len(),
            total_weight
        );

        // STEP 4: Identify trunk (tree diameter)
        let trunk_nodes = if self.config.compute_trunk {
            log::debug!("Step 4/5: Computing trunk (tree diameter)...");
            let trunk = self.compute_trunk(&mst_edges, &thickness, c);
            log::info!("  ✓ Trunk: {} nodes", trunk.len());
            trunk
        } else {
            log::debug!("Step 4/5: Skipping trunk computation");
            Vec::new()
        };

        // STEP 5: DFS traversal for 1D ordering
        log::debug!("Step 5/5: DFS traversal for centroid ordering...");
        let centroid_order = self.dfs_ordering(&mst_edges, &thickness, c);
        log::info!("  ✓ Ordering: {} centroids", centroid_order.len());

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  MST SKELETON COMPLETE                                ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");

        MSTOutput {
            candidate_edges,
            mst_edges,
            centroid_order,
            trunk_nodes,
            thickness,
            total_weight,
        }
    }

    /// Compute thickness proxy: mean variance per centroid
    fn compute_thickness<B: Backend>(&self, state: &CentroidState<B>) -> Vec<f32> {
        let thickness_tensor = state.get_thickness();
        let data = thickness_tensor.to_data();
        data.to_vec().unwrap()
    }

    /// Build k-NN candidate graph
    fn build_candidate_graph<B: Backend>(
        &self,
        state: &CentroidState<B>,
        thickness: &[f32],
    ) -> Vec<Edge> {
        let c = state.num_centroids();
        let k = self.config.k_neighbors.min(c - 1);

        let means_data = state.means.to_data();
        let means_vec: Vec<f32> = means_data.to_vec().unwrap();
        let f = state.feature_dim();

        let variances_data = state.variances.to_data();
        let variances_vec: Vec<f32> = variances_data.to_vec().unwrap();

        let mut all_edges = Vec::new();

        // For each centroid, find k nearest neighbors
        for i in 0..c {
            let mut neighbors: Vec<(usize, f32)> = Vec::new();

            for j in 0..c {
                if i == j {
                    continue;
                }

                let distance = self.compute_distance(i, j, &means_vec, &variances_vec, f);

                neighbors.push((j, distance));
            }

            // Sort by distance and take top k
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            neighbors.truncate(k);

            // Create edges with thickness weighting
            for (j, distance) in neighbors {
                let cost = self.compute_edge_cost(distance, thickness[i], thickness[j]);

                all_edges.push(Edge {
                    u: i,
                    v: j,
                    distance,
                    thickness_u: thickness[i],
                    thickness_v: thickness[j],
                    cost,
                });
            }
        }

        all_edges
    }

    /// Compute distance between two centroids
    fn compute_distance(
        &self,
        i: usize,
        j: usize,
        means: &[f32],
        variances: &[f32],
        f: usize,
    ) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Bhattacharyya => {
                let mean_i = &means[(i * f)..((i + 1) * f)];
                let mean_j = &means[(j * f)..((j + 1) * f)];
                let var_i = &variances[(i * f)..((i + 1) * f)];
                let var_j = &variances[(j * f)..((j + 1) * f)];

                bhattacharyya_distance_diagonal(mean_i, var_i, mean_j, var_j)
            }
            DistanceMetric::Euclidean => {
                let sum: f32 = (0..f)
                    .map(|d| {
                        let diff = means[i * f + d] - means[j * f + d];
                        diff * diff
                    })
                    .sum();
                sum.sqrt()
            }
            DistanceMetric::SquaredEuclidean => (0..f)
                .map(|d| {
                    let diff = means[i * f + d] - means[j * f + d];
                    diff * diff
                })
                .sum(),
        }
    }

    /// Compute edge cost with thickness weighting: c_ij = d_ij * φ(t_i, t_j)
    fn compute_edge_cost(&self, distance: f32, t_i: f32, t_j: f32) -> f32 {
        let phi = match self.config.thickness_weight {
            ThicknessWeight::Mean => (t_i + t_j) / 2.0,
            ThicknessWeight::Min => t_i.min(t_j),
            ThicknessWeight::Max => t_i.max(t_j),
            ThicknessWeight::GeometricMean => (t_i * t_j).sqrt(),
            ThicknessWeight::None => 1.0,
        };

        distance * phi
    }

    /// Prim's MST algorithm
    fn prim_mst(&self, edges: &[Edge], n_nodes: usize) -> (Vec<Edge>, f32) {
        use std::cmp::Ordering;

        // Build adjacency list
        let mut adj: Vec<Vec<(usize, f32, usize)>> = vec![Vec::new(); n_nodes];
        for (edge_idx, edge) in edges.iter().enumerate() {
            adj[edge.u].push((edge.v, edge.cost, edge_idx));
            adj[edge.v].push((edge.u, edge.cost, edge_idx));
        }

        // Priority queue: (cost, node, edge_idx)
        #[derive(Copy, Clone)]
        struct State {
            cost: f32,
            node: usize,
            edge_idx: usize,
        }

        impl Eq for State {}
        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.cost == other.cost && self.node == other.node
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap: reverse comparison
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut in_mst = vec![false; n_nodes];
        let mut mst_edge_indices = Vec::new();
        let mut total_weight = 0.0;

        // Start from node 0
        in_mst[0] = true;
        for &(neighbor, cost, edge_idx) in &adj[0] {
            heap.push(State {
                cost,
                node: neighbor,
                edge_idx,
            });
        }

        while let Some(State {
            cost,
            node,
            edge_idx,
        }) = heap.pop()
        {
            if in_mst[node] {
                continue;
            }

            in_mst[node] = true;
            mst_edge_indices.push(edge_idx);
            total_weight += cost;

            for &(neighbor, neighbor_cost, neighbor_edge_idx) in &adj[node] {
                if !in_mst[neighbor] {
                    heap.push(State {
                        cost: neighbor_cost,
                        node: neighbor,
                        edge_idx: neighbor_edge_idx,
                    });
                }
            }
        }

        let mst_edges: Vec<Edge> = mst_edge_indices
            .into_iter()
            .map(|idx| edges[idx].clone())
            .collect();

        (mst_edges, total_weight)
    }

    /// Compute trunk via tree diameter (longest path)
    fn compute_trunk(&self, mst_edges: &[Edge], thickness: &[f32], n_nodes: usize) -> Vec<usize> {
        if mst_edges.is_empty() {
            return Vec::new();
        }

        // Build adjacency list from MST
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_nodes];
        for edge in mst_edges {
            adj[edge.u].push((edge.v, edge.cost));
            adj[edge.v].push((edge.u, edge.cost));
        }

        // Find thickest node as starting point
        let root = thickness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // BFS to find farthest node from root
        let (farthest1, _) = self.bfs_farthest(&adj, root, n_nodes);

        // BFS from farthest1 to find diameter endpoint
        let (farthest2, distances) = self.bfs_farthest(&adj, farthest1, n_nodes);

        // Reconstruct path from farthest1 to farthest2
        let trunk = self.reconstruct_path(&adj, farthest1, farthest2, &distances);

        trunk
    }

    /// BFS to find farthest node and distances
    fn bfs_farthest(
        &self,
        adj: &[Vec<(usize, f32)>],
        start: usize,
        n_nodes: usize,
    ) -> (usize, Vec<f32>) {
        let mut distances = vec![f32::INFINITY; n_nodes];
        let mut queue = VecDeque::new();

        distances[start] = 0.0;
        queue.push_back(start);

        while let Some(u) = queue.pop_front() {
            for &(v, cost) in &adj[u] {
                let new_dist = distances[u] + cost;
                if new_dist < distances[v] {
                    distances[v] = new_dist;
                    queue.push_back(v);
                }
            }
        }

        let farthest = distances
            .iter()
            .enumerate()
            .filter(|(_, d)| **d < f32::INFINITY)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(start);

        (farthest, distances)
    }

    /// Reconstruct path between two nodes
    fn reconstruct_path(
        &self,
        adj: &[Vec<(usize, f32)>],
        start: usize,
        end: usize,
        distances: &[f32],
    ) -> Vec<usize> {
        let mut path = vec![end];
        let mut current = end;

        while current != start {
            let mut next = current;
            for &(neighbor, cost) in &adj[current] {
                if (distances[current] - distances[neighbor] - cost).abs() < 1e-6 {
                    next = neighbor;
                    break;
                }
            }

            if next == current {
                break; // Couldn't find path
            }

            path.push(next);
            current = next;
        }

        path.reverse();
        path
    }

    /// DFS traversal for 1D ordering (trunk-sprouts)
    fn dfs_ordering(&self, mst_edges: &[Edge], thickness: &[f32], n_nodes: usize) -> Vec<usize> {
        if mst_edges.is_empty() {
            return (0..n_nodes).collect();
        }

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for edge in mst_edges {
            adj[edge.u].push(edge.v);
            adj[edge.v].push(edge.u);
        }

        // Sort children by thickness (descending)
        for neighbors in &mut adj {
            neighbors.sort_by(|&a, &b| thickness[b].partial_cmp(&thickness[a]).unwrap());
        }

        // Find root (thickest node)
        let root = thickness
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // DFS
        let mut order = Vec::new();
        let mut visited = vec![false; n_nodes];
        self.dfs_visit(root, &adj, &mut visited, &mut order);

        order
    }

    /// DFS helper
    fn dfs_visit(
        &self,
        node: usize,
        adj: &[Vec<usize>],
        visited: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        visited[node] = true;
        order.push(node);

        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                self.dfs_visit(neighbor, adj, visited, order);
            }
        }
    }
}
