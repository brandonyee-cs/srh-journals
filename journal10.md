# Research Journal 10
*Prometheus: J1-J2 Heisenberg Model & RDM-VAE Extension*

---

## February 9, 2026 — 16 hours outside of class
**Focus:** Problem Setup, Exact Diagonalization Pipeline, and the Scaling Wall

Today was the first real day on the J1-J2 project. After the DTFIM paper we talked a lot about what the right next problem was — something that's actually open, not just a validation exercise. The J1-J2 Heisenberg model kept coming up. It's been debated for 30+ years and nobody agrees on what lives between the Néel and stripe phases. That felt like exactly the right target for Prometheus: a system with known behavior at the extremes, genuine uncertainty in the middle, and enough numerical literature that we can benchmark without having ground truth.

Spent the morning reading through the DMRG and ED literature on J1-J2. The competing proposals for the intermediate phase (J2/J1 ~ 0.4–0.6) are all over the place — plaquette valence bond solid, nematic order, quantum spin liquid, or maybe no intermediate phase at all and just a direct first-order transition. The fact that different groups using the same method get different answers depending on boundary conditions and bond dimension says something about how genuinely hard this problem is. 

### Problem Setup

The Hamiltonian is:

$$H = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j + J_2 \sum_{\langle\langle i,k \rangle\rangle} \vec{S}_i \cdot \vec{S}_k$$

We set J1 = 1 and vary the frustration ratio J2/J1. For small J2/J1, you get Néel antiferromagnetic order (checkerboard pattern). For large J2/J1, stripe (columnar) order dominates. The middle is the question.

**The scaling barrier** became obvious immediately when I tried setting up the L=6 ED. The Hilbert space dimension in the Sz_tot = 0 sector scales as C(N, N/2):

- L=4 (N=16): C(16,8) = **12,870** — tractable
- L=6 (N=36): C(36,18) ≈ **9.1 × 10^9** — impossible to store, let alone diagonalize
- L=8 (N=64): C(64,32) ≈ **1.8 × 10^18** — not even close

So L=4 gets exact diagonalization. Everything larger needs a different approach. That's going to be the whole methodological story of this paper — building a pipeline that works at L=4 and then figuring out how to scale it.

### Exact Diagonalization for L=4

Set up the Hamiltonian in QuSpin's `spin_basis_general` with Nup = N/2 to restrict to the Sz = 0 sector. The spin operators decompose as:

$$\vec{S}_i \cdot \vec{S}_j = S^z_i S^z_j + \frac{1}{2}(S^+_i S^-_j + S^-_i S^+_j)$$

Stored in CSR sparse format. Ground states via ARPACK Lanczos (`scipy.eigsh`), convergence criterion |E_n - E_{n-1}| < 10^{-10}, max 1000 iterations. Normalized explicitly and validated ||ψ||² - 1| < 10^{-8}.

Parameter sweep: J2/J1 ∈ [0.3, 0.7] with step 0.01 → 41 ground states. Focused on the intermediate regime with enough margin into the known Néel and stripe regimes for validation. Ran in parallel across 6 cores with checkpointing.

### Observable Computation

Computing the staggered magnetization was tricky — in the Sz = 0 sector, ⟨Sz_i⟩ = 0 everywhere by symmetry, so you can't compute it directly. Had to go through spin-spin correlations:

$$m_s^2 = \frac{1}{N^2} \sum_{i,j} (-1)^{i_x+i_y+j_x+j_y} \langle \vec{S}_i \cdot \vec{S}_j \rangle, \quad m_s = \sqrt{|m_s^2|}$$

Computed the full correlation matrix ⟨Si · Sj⟩ for all pairs, then used it to get staggered magnetization, structure factors at (π,π), (π,0), (0,π), plaquette order (four-spin correlations on 2×2 plaquettes), nematic order (anisotropy between x and y bonds), dimer order (alternating bond strengths), and entanglement entropy via SVD of the bipartite reshaped wavefunction.

Observable results for selected points:

| J2/J1 | e | ms | S(π,π) | S(π,0) |
|-------|---|----|--------|--------|
| 0.30 | -2.332 | 0.902 | 13.02 | -0.185 |
| 0.50 | -2.114 | 0.707 | 8.00 | +0.421 |
| 0.56 | -2.093 | 0.581 | 5.40 | +1.431 |
| 0.60 | -2.104 | 0.460 | 3.38 | +2.901 |
| 0.70 | -2.255 | 0.216 | -0.74 | +7.541 |

The energy minimum sits around J2/J1 ≈ 0.56, reflecting the maximum frustration between the two ordering tendencies. S(π,π) and S(π,0) cross around J2/J1 ≈ 0.58. One thing that immediately stood out: nematic order, dimer order, and stripe order parameter all came out numerically zero (< 10^{-11}) throughout the entire parameter range. That's almost certainly a finite-size artifact — the 4×4 lattice is just too small to support the symmetry breaking those phases require. The structure factor S(π,0) grows correctly even when mstripe vanishes, which is what you'd expect.

### The RDM Idea

Hit the scaling wall when trying to think about how to get the Q-VAE working at L=6. The Q-VAE needs full wavefunction coefficients {cσ} as input — those are the 2D-dimensional vectors (real and imaginary parts concatenated). At L=4 that's 25,740 numbers. At L=6 it would be ~18 billion. Not happening.

But here's the thing: what actually distinguishes different quantum phases? Correlation patterns. The Néel order parameter is a sum of two-point spin correlations. Structure factors are Fourier transforms of spin correlations. Plaquette order involves four-spin correlations on elementary plaquettes. All of this is encoded in **reduced density matrices** of small subsystems. DMRG can compute those efficiently even for large systems — it just can't give you the full wavefunction.

The RDM of a subsystem A is ρ_A = Tr_B |ψ⟩⟨ψ|, with dimension 2^{n_A} × 2^{n_A} for n_A sites. For small subsystems (single sites, pairs, plaquettes) this stays completely manageable regardless of total system size. The plan: extract RDMs from DMRG ground states, flatten and concatenate them into a feature vector, feed that to a VAE. If phase structure is encoded in local correlations — and it should be — then the latent space should organize the same way it did for the Q-VAE.

Tomorrow: implement the DMRG pipeline and test whether this actually works.

---

## February 10, 2026 — 18 hours outside of class
**Focus:** DMRG Ground States, RDM Extraction, and Training Both VAEs

The main event today was building the DMRG pipeline, implementing the RDM feature extraction, and training all three models (Q-VAE on L=4 full wavefunctions, RDM-VAE on L=6 and L=8). Long day but everything converged.

### DMRG Pipeline

Used TeNPy for DMRG with MPS bond dimension χ = 200–400 (increased until energy convergence). Cylindrical boundary conditions with periodic boundaries along one direction — this reduces finite-size effects while keeping things computationally tractable. Convergence criterion: ΔE < 10^{-8} between successive sweeps, 10–20 sweeps per calculation.

Parameter sweeps:
- L=6: J2/J1 ∈ [0, 1], step 0.025 → 41 ground states
- L=8: same → 41 ground states

Covered the full phase diagram for both, unlike L=4 where I focused on the intermediate regime. Total compute time across both sizes: most of the afternoon.

### RDM Feature Construction

For each DMRG ground state, extracted four types of RDMs and concatenated them into a single feature vector:

| Type | Matrix dim | Count (L=6) | Count (L=8) |
|------|-----------|-------------|-------------|
| Single-site ρ_i | 2×2 | 36 | 64 |
| Nearest-neighbor ρ_{ij} | 4×4 | 72 | 128 |
| Next-nearest-neighbor ρ_{ik} | 4×4 | 72 | 128 |
| Plaquette ρ_□ | 16×16 | 36 | 64 |
| **Total feature dim** | | **706** | **2148** |

The feature vector is x_RDM = [vec(ρ_1), vec(ρ_2), ..., vec(ρ_{ij}), vec(ρ_□,1), ...] ∈ R^{d_RDM}.

The choice of which subsystems to include covers all the correlation types that the physically relevant order parameters depend on — the full set of single-site, pairwise, and plaquette correlations. If phase information is encoded locally, this representation should capture it.

### Q-VAE Architecture (L=4)

Adapted the Q-VAE from the DTFIM paper. Input is [Re(ψ), Im(ψ)] ∈ R^{2D} where D = 12,870 for L=4, giving input dimension 25,740. The Heisenberg wavefunction is complex, unlike the DTFIM which had real coefficients, so the inner product structure in the loss function needed updating.

Encoder: Linear(25740 → 512) → LayerNorm → ReLU → Linear(512 → 256) → LayerNorm → ReLU → Linear(256 → 128) → LayerNorm → ReLU → then two parallel heads for μ and log σ² (each dim=8).

Decoder: symmetric 128 → 256 → 512 → 2D, with explicit wavefunction normalization on the output:

$$\psi_{\text{norm}} = \frac{\psi_{\text{out}}}{\sqrt{\sum_i (|\text{Re}(\psi_i)|^2 + |\text{Im}(\psi_i)|^2) + \epsilon}}$$

Loss function — quantum fidelity:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[1 - F(\psi_{\text{in}}, \psi_{\text{recon}})] + \beta D_{\text{KL}}$$

where the fidelity F = |⟨ψ_in|ψ_recon⟩|² uses the full complex inner product. β = 0.1 throughout.

Data augmentation via Sz flip symmetry: complex conjugation ψ → ψ* maps [Re(ψ), Im(ψ)] → [Re(ψ), -Im(ψ)]. Applied with 50% probability during training.

### RDM-VAE Architecture (L=6, L=8)

Simpler since RDM features are real — no need for fidelity loss or explicit normalization. Standard MSE reconstruction.

Encoder: Linear(d_RDM → 256) → LayerNorm → ReLU → (256 → 128) → LayerNorm → ReLU → (128 → 64) → LayerNorm → ReLU → μ and log σ² (dim=8). Decoder symmetric. Loss = MSE + β·KL, β = 0.1.

### Training

All models: Adam optimizer, lr = 10^{-3} with cosine annealing, batch size 32, max 1000 epochs, early stopping at 50 epochs without val improvement, gradient clipping max norm 1.0, 80/20 train/val split.

Results:

| Model | System | Epochs | Final val loss |
|-------|--------|--------|----------------|
| Q-VAE | L=4 | 1000 (full) | 0.01 (fidelity), F > 0.99 |
| RDM-VAE | L=6 | 224 | 0.0046 (MSE) |
| RDM-VAE | L=8 | 300 | 0.013 (MSE) |

Q-VAE needed all 1000 epochs — compressing a 25,740-dimensional complex wavefunction down to 8 latent dimensions is genuinely hard. L=6 converged early at 224 epochs, which was a good sign. L=8 took longer due to the larger feature dimension (2148 vs 706) but still achieved solid performance.

### Quick Cross-Validation Before Full Analysis

Before running the complete correlation analysis, I wanted a quick check that the RDM approach was actually capturing the same physics as the full wavefunction method. Extracted latent representations from both the L=4 Q-VAE and an RDM-VAE trained on L=4 RDM features (a comparison run), and checked whether they both picked up the same observables.

They did. Both leading latent dimensions correlated strongly with S(π,π) and S(π,0). The RDM feature approach isn't losing the relevant phase information. That was reassuring — the theoretical argument (phase structure is local) seems to hold numerically.

### DMRG Observable Data (L=6 sample)

| J2/J1 | e | S(π,π) | S(π,0) | S_ent |
|-------|---|--------|--------|-------|
| 0.00 | -0.670 | 2.585 | 0.182 | 2.341 |
| 0.40 | -0.522 | 1.657 | 0.242 | 2.115 |
| 0.575 | -0.489 | 0.707 | 0.679 | 1.461 |
| 0.600 | -0.491 | 0.497 | 1.095 | 1.477 |
| 0.800 | -0.581 | 0.124 | 2.566 | 2.407 |

Energy minimum at J2/J1 ≈ 0.575, consistent with L=4. S(π,π) and S(π,0) cross near J2/J1 ≈ 0.575. Entanglement entropy minimum also in that region — that's actually interesting, and something I want to think about more tomorrow. A true quantum critical point would typically show *enhanced* entanglement. A minimum is more consistent with a crossover.

Saved all ground states, RDM feature vectors, latent representations, and observable data to HDF5. Tomorrow is the full analysis.

---

## February 12, 2026 — 14 hours outside of class
**Focus:** Order Parameter Discovery, Critical Point Detection, and Paper

Analysis day. Ran the complete Prometheus discovery pipeline across all three system sizes, built out the critical point detection ensemble, validated against literature, and wrote the paper. The results were cleaner than expected.

### Latent Representation Extraction

Extracted deterministic latent representations z = μ(x) for each ground state (no sampling — use the encoder mean). This gives a compressed trajectory through an 8-dimensional latent space as J2/J1 varies.

### Correlation Analysis

Computed Pearson correlations r(z_k, O) between each of the 8 latent dimensions and all physical observables, with statistical significance from permutation tests (10,000 permutations) and bootstrap confidence intervals (1,000 resamples). Significance threshold: |r| ≥ 0.8, p < 0.01.

**L=4 Q-VAE results:**

| Observable | r(z0, ·) | r(z2, ·) | p-value |
|-----------|----------|----------|---------|
| Energy | +0.976 | +0.900 | 1.4×10⁻⁶ |
| Staggered mag. ms | -0.970 | -0.898 | 3.6×10⁻⁶ |
| S(π,π) | -0.971 | -0.899 | 3.1×10⁻⁶ |
| S(π,0) | +0.956 | +0.888 | 1.5×10⁻⁵ |
| Plaquette order P | -0.965 | -0.895 | 6.0×10⁻⁶ |

z0 autonomously discovered the Néel order parameter with r = -0.970. No labels, no prior knowledge of the relevant physics. The negative sign just means z0 decreases as Néel order increases. The full correlation matrix (all 8 latent dims × all observables) shows z0 and z2 carry nearly all the signal; the other six dimensions encode secondary structure not captured by any of the 11 observables I measured.

**L=6 RDM-VAE results:**

| Observable | Best latent | r | Second best | r |
|-----------|-------------|---|-------------|---|
| S(π,π) | z6 | -0.990 | z3 | -0.982 |
| S(π,0) | z1 | +0.973 | z4 | +0.965 |
| Energy density | z2 | +0.958 | z5 | +0.941 |
| S_ent | z0 | +0.892 | z7 | +0.876 |

r = -0.990 between z6 and S(π,π). That's remarkably close to perfect tracking of the Néel structure factor from local RDM inputs alone, without any access to the global wavefunction. The redundancy (z1 *and* z4 both correlating with S(π,0)) is expected given an 8-dimensional latent space and a relatively simple phase diagram — the VAE learns multiple representations of the same physics.

L=8 confirmed the same structure with |r| > 0.95 for structure factor correlations.

### Critical Point Detection — Three Methods

**Method 1: Latent variance.** At critical points the latent representation changes most rapidly, so variance peaks there:

$$\chi_z(J_2/J_1) = \sum_k \text{Var}[z_k(J_2/J_1)]$$

Smoothed with Savitzky-Golay filter (window 5, order 2), peaks via `scipy.signal.find_peaks` with prominence threshold 10% of max. Uncertainty from FWHM → σ conversion.

**Method 2: Reconstruction error.** Critical states have maximum complexity and are hardest to compress — peaks in 1 - F(ψ_in, ψ_recon).

**Method 3: Fidelity susceptibility.** Measures how fast the ground state changes with the parameter:

$$\chi_F(J_2/J_1) \approx -\frac{\log F_{-\delta} - 2\log F_0 + \log F_{+\delta}}{\delta^2}$$

Peaks where the ground state is most sensitive to parameter changes. For L=4 the fidelity is exact. For L=6, L=8 I used RDM feature overlap as a proxy.

**Ensemble estimate** via inverse-variance weighting across all three methods:

$$(J_2/J_1)^c_{\text{ensemble}} = \frac{\sum_m w_m (J_2/J_1)^c_m}{\sum_m w_m}, \quad w_m = 1/\sigma_m^2$$

L=4 result: **(J2/J1)^c = 0.63 ± 0.004**. This falls within the literature range of [0.55, 0.65] for the intermediate-to-stripe transition from DMRG and QMC studies.

### Validation Framework

Automated validation in known phase regimes:

- **Néel regime (J2/J1 < 0.4):** max |r(z_k, ms)| ≥ 0.7 required → **PASS** with r = 0.970
- **Stripe regime (J2/J1 > 0.6):** max |r(z_k, mstripe)| ≥ 0.7 required → **FAIL**

The stripe validation failure is expected and not a methodology problem. The 4×4 lattice is simply too small to support true stripe long-range order — mstripe is numerically zero throughout, even though S(π,0) and S(0,π) grow correctly in the stripe regime. The structure factors capture the physics that mstripe can't on a finite lattice.

k-means clustering (k=2) on the L=4 latent representations: silhouette score **0.817**, two well-separated clusters with boundary at J2/J1 ≈ 0.55.

### Crossover Consistency Across System Sizes

The big result — crossover region is consistent across all three sizes:

| Property | L=4 (Q-VAE) | L=6 (RDM-VAE) | L=8 (RDM-VAE) |
|---------|-------------|---------------|----------------|
| Hilbert space dim | 12,870 | 9.1×10⁹ | 1.8×10¹⁸ |
| Feature dim (VAE input) | 25,740 | 706 | 2,148 |
| \|r(z, S(π,π))\| | 0.971 | 0.990 | 0.962 |
| \|r(z, S(π,0))\| | 0.956 | 0.973 | 0.958 |
| Crossover region | 0.55–0.65 | 0.55–0.60 | 0.55–0.60 |

The slightly narrower crossover at larger sizes is consistent with finite-size broadening — bigger systems give sharper features. The fact that the crossover region is consistent across two completely different methodologies (full wavefunction vs RDM features) and three very different system sizes is strong evidence it's real physics and not a methodological artifact.

### What Does This Say About the J1-J2 Problem?

The smooth, monotonic evolution of the latent trajectory through the intermediate regime — no discontinuities, no sudden jumps, no evidence of the trajectory clustering into a third distinct region — is more consistent with a crossover or weakly first-order transition than with a robust intermediate phase. If there were a genuine new phase with distinct order, I'd expect the latent space to cluster it separately.

The plaquette order parameter P decreases continuously without any enhancement in the intermediate regime, arguing against a plaquette VBS ground state at these system sizes. Nematic and dimer order are zero throughout. But all of these null results are subject to the standard caveat: L ≤ 8 may just be too small to see the relevant symmetry breaking.

The entanglement entropy minimum at J2/J1 ≈ 0.575 for L=6 was interesting to think about. A sharp quantum critical point would typically show *enhanced* entanglement — the minimum is more consistent with a smooth crossover where the system is transitioning from one type of order to another, passing through a regime where neither dominates strongly.

### Paper

Wrote the full manuscript today. The story is structured as Prometheus III — after the 2D Ising paper (exact solution validation) and the 3D/DTFIM paper (no-exact-solution and exotic quantum criticality), this is the first genuine discovery application to an open problem. The three-paper sequence is what makes the discovery claims credible: we're not asking reviewers to trust a brand-new method.

The methodological contribution — RDM features enable scaling from N~20 (ED limit) to N~64 (DMRG accessible) while maintaining comparable order parameter discovery performance — is significant independent of what the physics conclusion ends up being. A 10^{14}× increase in accessible Hilbert space dimension is real.

**Paper status:** Complete draft, 22 pages + appendices. Six main figures. Targeting arXiv preprint submission this week, Physical Review B as primary venue.

Immediate next steps: Prep for CSEF.

--

## February 26, 2026

**Focus:** Worked on editing the Prometheus Youtube Video for CSEF.

## February 27, 2026

Listened to the talk and participated with the WIS teachers. 

My ideas regarding my presentation are to show the power of computation for problem-solving. Through demos (I can make full coding demos, visualizations, animations, etc) in the form of an interactive presentation/webapp.

Ideally I don't have to do a worksheet I agree with Lancaster I would like to be removed from that because I don't know how much the kids will retain. 
--
