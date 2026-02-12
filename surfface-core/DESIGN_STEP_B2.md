## **Key Design Decisions**

### **1. RTS Smoother**[^3][^1][^2]
 **Rauch-Tung-Striebel (RTS) smoother**
The RTS smoother consists of two passes:[^3][^2]

1. **Forward pass (Kalman filter)**: Process observations sequentially, obtaining filtered estimates $\hat{\mu}_{t|t}, P_{t|t}$
2. **Backward pass (RTS equations)**: Refine estimates using future data, obtaining smoothed estimates $\hat{\mu}_{t|T}, P_{t|T}$

**Key advantage**: Smoothed estimates are more accurate than filtered estimates because they use **all observations** (past and future)

- **Forward pass**: Standard Kalman filter along MST order
- **Backward pass**: RTS equations refine estimates using future data
- **Optimality**: RTS is the optimal smoother for linear Gaussian systems


### **2. State-Space Model**

**Process model**:

$$
\mathbf{x}_t = F \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, Q)
$$

**Observation model**:

$$
\mathbf{y}_t = H \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(0, R)
$$

Where:

- $F = I$ (identity, random walk model)
- $H = I$ (observe state directly)
- $Q = \sigma_Q^2 I$ (process noise, tunable)
- $R = \sigma_R^2 I + \Sigma_{\text{obs}}$ (observation noise + centroid variance)


### **3. Numerical Stability**[^4]

- Variance clamping (`variance_floor`, `variance_ceiling`)
- Symmetry enforcement for covariance matrices
- Prevents numerical explosion in low-data regimes


### **4. MST Order as 1D Chain**

- Centroids processed sequentially along `mst_output.centroid_order`
- Thick → thin traversal (core → periphery)
- Future: Add trunk edges as additional constraints
