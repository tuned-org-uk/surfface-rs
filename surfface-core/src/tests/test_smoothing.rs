/// The only theoretical "trap" in this implementation is the assumption that mst_output.centroid_order
///  is a single contiguous chain. If the MST construction in B1 produces a forest (multiple disconnected components),
///  the smoother will treat the transition between the end of one tree and the start of another as a "step" in time.
/// Recommendation: In a future iteration, you might check if order[t] and order[prev_t] are actually adjacent
///  in the MST. If they aren't, you could reset the Kalman filter at that step (treating it as a new $t=0$).
///  However, for the current "Stage B2" scope, the linear traversal of the centroid_order is the standard
///  and expected behavior.
use std::collections::HashSet;

use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::mst::{Edge, MSTOutput};
use crate::smoothing_chain::{SmoothingConfig, SmoothingStage, TransitionModel};
use burn::tensor::{Int, Tensor, backend::Backend};

type TestBackend = AutoBackend;

/// Create synthetic noisy centroid state with a deterministic sine + noise signal
fn create_noisy_centroids(c: usize, f: usize, noise_level: f32) -> CentroidState<TestBackend> {
    crate::tests::init();
    let device = Default::default();
    let mut means = Vec::with_capacity(c * f);
    for i in 0..c {
        for j in 0..f {
            let base = (i as f32 * 0.5 + j as f32 * 0.1).sin();
            let noise = (i * 7 + j * 13) as f32 % 100.0 / 100.0 - 0.5;
            means.push(base + noise * noise_level);
        }
    }
    let variances = vec![noise_level * noise_level; c * f];
    let counts = vec![10i64; c];

    CentroidState {
        means: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(means, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(variances, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::<TestBackend, 1, Int>::from_data(
            burn::tensor::TensorData::new(counts, burn::tensor::Shape::new([c])),
            &device,
        ),
    }
}

/// Create a simple linear chain MSTOutput (0 → 1 → 2 → … → c-1).
/// trunk_edges is empty; all edges form the single path so there is no
/// meaningful trunk/branch distinction in a pure chain.
fn create_linear_mst(c: usize) -> MSTOutput {
    crate::tests::init();
    let order: Vec<usize> = (0..c).collect();
    let mut edges = Vec::new();
    for i in 0..c.saturating_sub(1) {
        edges.push(Edge {
            u: i,
            v: i + 1,
            distance: 1.0,
            thickness_u: 1.0,
            thickness_v: 1.0,
            cost: 1.0,
        });
    }
    MSTOutput {
        candidate_edges: edges.clone(),
        mst_edges: edges,
        centroid_order: order,
        trunk_nodes: vec![],
        trunk_edges: HashSet::new(),
        thickness: vec![1.0; c],
        total_weight: c.saturating_sub(1) as f32,
        nodes_in_mst: c,
    }
}

/// Create an MSTOutput where the trunk runs through every node of the chain.
/// Useful for testing TrunkAware: all transitions are trunk transitions.
fn create_full_trunk_mst(c: usize) -> MSTOutput {
    crate::tests::init();
    let order: Vec<usize> = (0..c).collect();
    let mut edges = Vec::new();
    for i in 0..c.saturating_sub(1) {
        edges.push(Edge {
            u: i,
            v: i + 1,
            distance: 1.0,
            thickness_u: 1.0,
            thickness_v: 1.0,
            cost: 1.0,
        });
    }
    // Every consecutive pair is a trunk edge (bidirectional)
    let trunk_nodes: Vec<usize> = (0..c).collect();
    let trunk_edges: HashSet<(usize, usize)> = trunk_nodes
        .windows(2)
        .flat_map(|w| [(w[0], w[1]), (w[1], w[0])])
        .collect();

    MSTOutput {
        candidate_edges: edges.clone(),
        mst_edges: edges,
        centroid_order: order,
        trunk_nodes,
        trunk_edges,
        thickness: vec![1.0; c],
        total_weight: c.saturating_sub(1) as f32,
        nodes_in_mst: c,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_kalman_reduces_variance() {
    let c = 10;
    let f = 3;
    let noise_level = 0.5;

    let noisy_state = create_noisy_centroids(c, f, noise_level);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    println!(
        "Variance reduction: {:.2}%",
        output.variance_reduction * 100.0
    );

    assert!(
        output.variance_reduction > 0.0,
        "Smoothing should reduce variance"
    );
    assert!(
        output.variance_reduction < 1.0,
        "Variance reduction should be < 100%"
    );

    let raw_vars: Vec<f32> = noisy_state.variances.to_data().to_vec().unwrap();
    let smooth_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();
    let raw_mean = raw_vars.iter().sum::<f32>() / raw_vars.len() as f32;
    let smooth_mean = smooth_vars.iter().sum::<f32>() / smooth_vars.len() as f32;

    assert!(
        smooth_mean < raw_mean,
        "Smoothed variance {:.4} should be < raw {:.4}",
        smooth_mean,
        raw_mean
    );
}

#[test]
fn test_kalman_preserves_counts() {
    let noisy_state = create_noisy_centroids(5, 2, 0.3);
    let mst_output = create_linear_mst(5);

    let output = SmoothingStage::new(SmoothingConfig::default()).execute(&noisy_state, &mst_output);

    let input_counts: Vec<i32> = noisy_state
        .counts
        .to_data()
        .convert::<i32>()
        .to_vec()
        .unwrap();
    let output_counts: Vec<i32> = output.counts.to_data().convert::<i32>().to_vec().unwrap();

    assert_eq!(
        input_counts, output_counts,
        "Counts must be preserved during smoothing"
    );
}

#[test]
fn test_kalman_smoothness_property() {
    let c = 20;
    let f = 1;

    let noisy_state = create_noisy_centroids(c, f, 1.0);
    let mst_output = create_linear_mst(c);

    let output =
        SmoothingStage::new(SmoothingConfig::aggressive()).execute(&noisy_state, &mst_output);

    let raw_means: Vec<f32> = noisy_state.means.to_data().to_vec().unwrap();
    let smooth_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    let total_variation = |v: &[f32]| -> f32 { v.windows(2).map(|w| (w[1] - w[0]).abs()).sum() };

    let raw_tv = total_variation(&raw_means);
    let smooth_tv = total_variation(&smooth_means);

    println!("Raw TV: {:.4}  Smoothed TV: {:.4}", raw_tv, smooth_tv);

    assert!(
        smooth_tv < raw_tv,
        "Smoothed trajectory should have lower total variation"
    );
}

#[test]
fn test_kalman_conservative_vs_aggressive() {
    let c = 10;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.5);
    let mst_output = create_linear_mst(c);

    let output_cons =
        SmoothingStage::new(SmoothingConfig::conservative()).execute(&noisy_state, &mst_output);
    let output_agg =
        SmoothingStage::new(SmoothingConfig::aggressive()).execute(&noisy_state, &mst_output);

    let calc_tv = |data: &[f32]| -> f32 {
        data.chunks_exact(f)
            .collect::<Vec<_>>()
            .windows(2)
            .flat_map(|w| w[0].iter().zip(w[1]).map(|(a, b)| (a - b).abs()))
            .sum()
    };

    let cons_means: Vec<f32> = output_cons.smoothed_means.to_data().to_vec().unwrap();
    let agg_means: Vec<f32> = output_agg.smoothed_means.to_data().to_vec().unwrap();
    let cons_tv = calc_tv(&cons_means);
    let agg_tv = calc_tv(&agg_means);

    println!(
        "Conservative TV: {:.4}  Aggressive TV: {:.4}",
        cons_tv, agg_tv
    );

    assert!(
        agg_tv < cons_tv,
        "Aggressive (TV={:.4}) should be smoother than conservative (TV={:.4})",
        agg_tv,
        cons_tv
    );

    // Aggressive should also have higher smoothing gains
    let mean_gain = |gains: &[f32]| gains.iter().sum::<f32>() / gains.len() as f32;
    let cons_gain = mean_gain(&output_cons.smoothing_gains);
    let agg_gain = mean_gain(&output_agg.smoothing_gains);

    println!(
        "Conservative gain: {:.4}  Aggressive gain: {:.4}",
        cons_gain, agg_gain
    );

    assert!(
        agg_gain > cons_gain,
        "Aggressive should have higher mean smoothing gain"
    );
}

#[test]
fn test_kalman_single_centroid() {
    let c = 1;
    let f = 3;

    let state = create_noisy_centroids(c, f, 0.2);
    let mst_output = MSTOutput {
        candidate_edges: vec![],
        mst_edges: vec![],
        centroid_order: vec![0],
        trunk_nodes: vec![],
        trunk_edges: HashSet::new(),
        thickness: vec![1.0],
        total_weight: 0.0,
        nodes_in_mst: 1,
    };

    let output = SmoothingStage::new(SmoothingConfig::default()).execute(&state, &mst_output);

    // Single centroid: no neighbours → means must be unchanged
    let input_means: Vec<f32> = state.means.to_data().to_vec().unwrap();
    let output_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    for (i, (inp, out)) in input_means.iter().zip(&output_means).enumerate() {
        assert!(
            (inp - out).abs() < 1e-5,
            "Single-centroid mean should be unchanged at index {}",
            i
        );
    }

    // Single centroid: gains vec must be empty (no transitions)
    assert!(
        output.smoothing_gains.is_empty(),
        "Single centroid should produce no smoothing gains"
    );
}

#[test]
fn test_kalman_numerical_stability() {
    let c = 5;
    let f = 2;
    let device: <TestBackend as Backend>::Device = Default::default();

    let means = vec![1.0f32; c * f];
    // Extreme variances: zero, huge, tiny, normal, large
    let variances: Vec<f32> = [0.0f32, 1e10, 1e-10, 1.0, 1e5]
        .iter()
        .flat_map(|&v| std::iter::repeat(v).take(f))
        .collect();
    let counts = vec![5i32; c];

    let state = CentroidState {
        means: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(means, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(variances, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::<TestBackend, 1, Int>::from_data(
            burn::tensor::TensorData::new(counts, burn::tensor::Shape::new([c])),
            &device,
        ),
    };

    let output =
        SmoothingStage::new(SmoothingConfig::default()).execute(&state, &create_linear_mst(c));

    let config = SmoothingConfig::default();

    let smooth_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();
    let smooth_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    for (i, val) in smooth_means.iter().enumerate() {
        assert!(val.is_finite(), "Smoothed mean at {} is not finite", i);
    }
    for (i, val) in smooth_vars.iter().enumerate() {
        assert!(val.is_finite(), "Smoothed variance at {} is not finite", i);
        assert!(*val >= 0.0, "Smoothed variance at {} is negative", i);
        assert!(
            *val >= config.variance_floor,
            "Variance {} below floor: {}",
            i,
            val
        );
        assert!(
            *val <= config.variance_ceiling,
            "Variance {} above ceiling: {}",
            i,
            val
        );
    }
}

#[test]
fn test_kalman_to_centroid_state() {
    let c = 5;
    let f = 3;

    let noisy_state = create_noisy_centroids(c, f, 0.3);
    let output = SmoothingStage::new(SmoothingConfig::default())
        .execute(&noisy_state, &create_linear_mst(c));

    let smoothed_state = output.to_centroid_state();

    assert_eq!(smoothed_state.num_centroids(), c);
    assert_eq!(smoothed_state.feature_dim(), f);

    let original_counts: Vec<i32> = noisy_state
        .counts
        .to_data()
        .convert::<i32>()
        .to_vec()
        .unwrap();
    let smoothed_counts: Vec<i32> = smoothed_state
        .counts
        .to_data()
        .convert::<i32>()
        .to_vec()
        .unwrap();
    assert_eq!(original_counts, smoothed_counts, "Counts must be preserved");
}

#[test]
fn test_kalman_forward_backward_consistency() {
    let c = 8;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.4);
    let output = SmoothingStage::new(SmoothingConfig::default())
        .execute(&noisy_state, &create_linear_mst(c));

    let filtered_vars: Vec<f32> = output.filtered_variances.to_data().to_vec().unwrap();
    let smoothed_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    // RTS optimality: P_smooth[t] ≤ P_filt[t] for all t, f
    for (i, (filt, smooth)) in filtered_vars.iter().zip(&smoothed_vars).enumerate() {
        assert!(
            *smooth <= *filt + 1e-5, // small epsilon for float rounding
            "Smoothed variance {:.6} should be ≤ filtered variance {:.6} at index {}",
            smooth,
            filt,
            i
        );
    }

    // Smoothed means should stay within a 1-sigma neighbourhood of filtered means
    let raw_means: Vec<f32> = noisy_state.means.to_data().to_vec().unwrap();
    let filtered_means: Vec<f32> = output.filtered_means.to_data().to_vec().unwrap();
    let smoothed_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    for i in 0..c * f {
        let lo = raw_means[i].min(filtered_means[i]) - 1.0;
        let hi = raw_means[i].max(filtered_means[i]) + 1.0;
        assert!(
            smoothed_means[i] >= lo && smoothed_means[i] <= hi,
            "Smoothed mean {:.4} out of bounds [{:.4}, {:.4}] at index {}",
            smoothed_means[i],
            lo,
            hi,
            i
        );
    }

    // Smoothed trajectory should be more consistent (lower std of step differences)
    let step_std = |means: &[f32]| -> f32 {
        let diffs: Vec<f32> = means
            .chunks_exact(f)
            .collect::<Vec<_>>()
            .windows(2)
            .flat_map(|w| w[0].iter().zip(w[1]).map(|(a, b)| b - a))
            .collect();
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let var = diffs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / diffs.len() as f32;
        var.sqrt()
    };

    let filt_std = step_std(&filtered_means);
    let smooth_std = step_std(&smoothed_means);

    println!(
        "Filtered step std: {:.4}  Smoothed step std: {:.4}",
        filt_std, smooth_std
    );

    assert!(
        smooth_std <= filt_std + 1e-5,
        "Smoothed trajectory should be more consistent (lower step std)"
    );
}

#[test]
fn test_kalman_smoothing_gains() {
    let c = 10;
    let f = 2;

    let output = SmoothingStage::new(SmoothingConfig::default())
        .execute(&create_noisy_centroids(c, f, 0.5), &create_linear_mst(c));

    // Gains vector must have exactly C-1 entries (one per transition)
    assert_eq!(
        output.smoothing_gains.len(),
        c - 1,
        "Should have exactly C-1 smoothing gains"
    );

    for (i, gain) in output.smoothing_gains.iter().enumerate() {
        assert!(gain.is_finite(), "Gain at transition {} is not finite", i);
        assert!(*gain >= 0.0, "Gain at transition {} is negative", i);
        assert!(
            *gain <= 1.0 + 1e-6,
            "Gain at transition {} exceeds 1.0: {}",
            i,
            gain
        );
    }

    println!("Smoothing gains: {:?}", output.smoothing_gains);
}

#[test]
fn test_kalman_disconnected_mst() {
    let c = 5;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.3);
    let mst_output = MSTOutput {
        candidate_edges: vec![],
        mst_edges: vec![Edge {
            u: 0,
            v: 1,
            distance: 1.0,
            thickness_u: 1.0,
            thickness_v: 1.0,
            cost: 1.0,
        }],
        centroid_order: vec![0, 1, 2, 3, 4],
        trunk_nodes: vec![],
        trunk_edges: HashSet::new(),
        thickness: vec![1.0; c],
        total_weight: 1.0,
        nodes_in_mst: 2,
    };

    let output = SmoothingStage::new(SmoothingConfig::default()).execute(&noisy_state, &mst_output);

    // Must not panic, must return a valid state covering all centroids
    assert!(output.variance_reduction.is_finite());
    assert_eq!(output.to_centroid_state().num_centroids(), c);
}

#[test]
fn test_kalman_config_variants() {
    let default = SmoothingConfig::default();
    let conservative = SmoothingConfig::conservative();
    let aggressive = SmoothingConfig::aggressive();
    let trunk_aware = SmoothingConfig::trunk_aware(0.5);

    assert!(conservative.observation_noise < default.observation_noise);
    assert!(aggressive.observation_noise > default.observation_noise);
    assert!(aggressive.process_noise < default.process_noise);
    assert!(matches!(
        trunk_aware.transition_model,
        TransitionModel::TrunkAware { .. }
    ));
}

#[test]
fn test_kalman_deterministic() {
    let c = 10;
    let f = 3;

    let noisy_state = create_noisy_centroids(c, f, 0.4);
    let mst_output = create_linear_mst(c);
    let kalman = SmoothingStage::new(SmoothingConfig::default());

    let output1 = kalman.execute(&noisy_state, &mst_output);
    let output2 = kalman.execute(&noisy_state, &mst_output);

    let means1: Vec<f32> = output1.smoothed_means.to_data().to_vec().unwrap();
    let means2: Vec<f32> = output2.smoothed_means.to_data().to_vec().unwrap();
    let vars1: Vec<f32> = output1.smoothed_variances.to_data().to_vec().unwrap();
    let vars2: Vec<f32> = output2.smoothed_variances.to_data().to_vec().unwrap();

    for (i, (m1, m2)) in means1.iter().zip(&means2).enumerate() {
        assert!((m1 - m2).abs() < 1e-6, "Mean not deterministic at {}", i);
    }
    for (i, (v1, v2)) in vars1.iter().zip(&vars2).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Variance not deterministic at {}",
            i
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New tests covering fixed bugs
// ─────────────────────────────────────────────────────────────────────────────

/// RTS optimality invariant: P_smooth ≤ P_filt for every (t, f) entry.
/// This validates that the backward pass only reduces (never inflates) variance.
#[test]
fn test_kalman_smoothed_variance_leq_filtered() {
    let c = 15;
    let f = 4;

    let output = SmoothingStage::new(SmoothingConfig::default())
        .execute(&create_noisy_centroids(c, f, 0.6), &create_linear_mst(c));

    let filt: Vec<f32> = output.filtered_variances.to_data().to_vec().unwrap();
    let smooth: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    for i in 0..c * f {
        assert!(
            smooth[i] <= filt[i] + 1e-5,
            "RTS invariant violated at (t={}, f={}): P_smooth={:.6} > P_filt={:.6}",
            i / f,
            i % f,
            smooth[i],
            filt[i]
        );
    }
}

/// Damped(α) must apply α² scaling to the prior variance.
/// We verify indirectly: with α=0.5 the state is pulled toward zero,
/// so the smoothed mean norm must be smaller than the raw mean norm.
#[test]
fn test_damped_covariance_scaling() {
    let c = 8;
    let f = 2;
    let device: <TestBackend as Backend>::Device = Default::default();

    // Centroids far from zero so the damping effect is visible
    let means: Vec<f32> = (0..c * f).map(|i| 5.0 + i as f32 * 0.1).collect();
    let variances = vec![0.1f32; c * f];
    let counts = vec![10i32; c];

    let state = CentroidState {
        means: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(means.clone(), burn::tensor::Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(variances, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::<TestBackend, 1, Int>::from_data(
            burn::tensor::TensorData::new(counts, burn::tensor::Shape::new([c])),
            &device,
        ),
    };

    let config = SmoothingConfig {
        process_noise: 0.01,
        observation_noise: 0.1,
        transition_model: TransitionModel::Damped(0.5),
        variance_floor: 1e-6,
        variance_ceiling: 1e3,
    };

    let output = SmoothingStage::new(config).execute(&state, &create_linear_mst(c));
    let smooth_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    // With strong damping the filtered predictions contract toward zero,
    // so the mean absolute value of smoothed means should be < raw means.
    let raw_abs: f32 = means.iter().map(|x| x.abs()).sum::<f32>() / means.len() as f32;
    let smooth_abs: f32 =
        smooth_means.iter().map(|x| x.abs()).sum::<f32>() / smooth_means.len() as f32;

    println!(
        "Raw mean abs: {:.4}  Smoothed mean abs: {:.4}",
        raw_abs, smooth_abs
    );

    assert!(
        smooth_abs < raw_abs,
        "Damped(0.5) should pull smoothed means toward zero: raw={:.4} smooth={:.4}",
        raw_abs,
        smooth_abs
    );

    // Variances must still satisfy RTS invariant
    let filt_vars: Vec<f32> = output.filtered_variances.to_data().to_vec().unwrap();
    let smooth_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();
    for i in 0..c * f {
        assert!(
            smooth_vars[i] <= filt_vars[i] + 1e-5,
            "RTS invariant violated for Damped at index {}",
            i
        );
    }
}

/// TrunkAware with trunk_factor < 1 should produce a smoother trajectory
/// on a full-trunk MST than plain Identity with the same base process_noise,
/// because trunk edges get lower Q (tighter smoothing).
#[test]
fn test_trunk_aware_lower_smoothing() {
    let c = 10;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.5);
    // All edges are trunk edges → every transition uses trunk_factor * Q
    let full_trunk_mst = create_full_trunk_mst(c);

    let identity_output =
        SmoothingStage::new(SmoothingConfig::default()).execute(&noisy_state, &full_trunk_mst);

    let trunk_output = SmoothingStage::new(SmoothingConfig::trunk_aware(0.1))
        .execute(&noisy_state, &full_trunk_mst);

    let calc_tv = |data: &[f32]| -> f32 {
        data.chunks_exact(f)
            .collect::<Vec<_>>()
            .windows(2)
            .flat_map(|w| w[0].iter().zip(w[1]).map(|(a, b)| (a - b).abs()))
            .sum()
    };

    let identity_means: Vec<f32> = identity_output.smoothed_means.to_data().to_vec().unwrap();
    let trunk_means: Vec<f32> = trunk_output.smoothed_means.to_data().to_vec().unwrap();

    let identity_tv = calc_tv(&identity_means);
    let trunk_tv = calc_tv(&trunk_means);

    println!(
        "Identity TV: {:.4}  TrunkAware(0.1) TV: {:.4}",
        identity_tv, trunk_tv
    );

    assert!(
        trunk_tv < identity_tv,
        "TrunkAware(0.1) on full-trunk MST should be smoother than Identity: \
         trunk_tv={:.4} identity_tv={:.4}",
        trunk_tv,
        identity_tv
    );
}
