# Research Journal 11
*Beam-Plasma Collective Oscillations: Paper Implementation & Submission*

---

## March 2, 2026 — 8 hours outside of class
**Focus:** Part I Theory Implementation — Kinetic Field Theory and Dielectric Response

In plain terms, this session was about writing down the physics of how particles in a dense beam interact with each other collectively — essentially asking whether a beam of charged particles, once it gets dense enough, starts behaving more like a plasma than a collection of independent particles. The formalism needed to answer that question rigorously is what got committed to the manuscript today.

Today was the first intensive implementation session on the CERN BL4S paper. The theoretical scaffolding had been sketched across several earlier discussions with Wilson, but today I committed the full formalism to the manuscript: the Vlasov–Poisson kinetic framework, the second-quantization setup, and the derivation of the Lindhard dielectric function for all three beam distributions. A lot of this draws on machinery developed for the Prometheus J1-J2 paper (effective action methods, RPA resummation), but the physics is genuinely different — we are now in the intermediate-energy regime ($\gamma \sim 10$–100) rather than the equilibrium condensed matter setting.

### Establishing the N-Particle Beam Hamiltonian

The starting point is the second-quantized Hamiltonian in the laboratory frame. For electrons (spin-1/2 fermions):

$$\hat{H} = \hat{H}_0 + \hat{H}_{\text{ext}} + \hat{H}_{\text{int}}$$

where the interaction term captures instantaneous Coulomb repulsion:

$$\hat{H}_{\text{int}} = \frac{1}{2} \int d^3x \int d^3x' \frac{\hat{\rho}(\mathbf{x})\hat{\rho}(\mathbf{x}')}{4\pi\epsilon_0|\mathbf{x} - \mathbf{x}'|}$$

One subtlety worked through carefully: the beam geometry breaks full Poincaré invariance. Standard finite-temperature field theory lives in equilibrium and assumes homogeneity; our system does not. The resolution is the decomposition $\hat{\psi} = \langle \hat{\psi} \rangle + \hat{\delta}\psi$ — a mean-field beam profile plus fluctuations — followed by deriving an effective action by integrating out high-frequency modes via a momentum-shell Wilsonian procedure.

**Effective Action and Renormalizability (Theorem 1):**

The one-loop trace-log expansion of $\det_>(\not{D}_A + m)$ generates several operator structures:

| Diagram | Operator | Divergence | Counterterm |
|---|---|---|---|
| Vacuum polarization | $F_{\mu\nu}F^{\mu\nu}$ | $1/\varepsilon$ | $Z_3$ |
| Fermion self-energy | $\bar{\psi}\gamma^\mu\partial_\mu\psi$, $m\bar{\psi}\psi$ | $1/\varepsilon$ | $Z_2$, $\delta m$ |
| Vertex correction | $e\bar{\psi}\gamma^\mu A_\mu\psi$ | $1/\varepsilon$ | $Z_1 = Z_2$ (Ward) |
| Tadpole | $A_\mu$ | 0 | none |
| Triangle | $A^3$ | 0 (Furry) | none |
| Light-by-light | $(F^2)^2$ | finite | none |

Furry's theorem kills the triangle diagram; the Ward identity $Z_1 = Z_2$ ensures gauge invariance is preserved at one loop. Theorem 1 — no new counterterms needed, the theory is renormalizable at one loop — follows. The physical density-dependent corrections (in-medium mass shift, Lindhard polarization, screened Coulomb interaction) are finite and are not UV counterterms. Important to state clearly since referees occasionally conflate the two.

### Lindhard Function for Three Beam Distributions

The non-interacting polarization tensor in the RPA:

$$\Pi^{00}_0(q^\mu) = -\frac{2}{(2\pi)^3} \int \frac{d^3k}{2E_k} \frac{f(E_k) - f(E_{k+q})}{\omega + i\eta - (E_{k+q} - E_k)}$$

Evaluating this for three beam momentum distributions required distinctly different analytic techniques — spent most of the afternoon working through each one.

**Degenerate Fermi gas:** In dimensionless variables $u = m\omega/(k_F q)$ and $z = q/(2k_F)$, the real part takes the standard Lindhard form:

$$\text{Re}[\chi^{\text{NR}}_0] = -\frac{mk_F}{2\pi^2}\left[\frac{1}{2} + \frac{1-(u-z)^2}{8z}\ln\left|\frac{u-z+1}{u-z-1}\right| + \frac{1-(u+z)^2}{8z}\ln\left|\frac{u+z+1}{u+z-1}\right|\right]$$

The imaginary part is piecewise linear, vanishing above $\omega_+ = v_F q + q^2/(2m)$. The log divergence at $\omega_+$ is the van Hove singularity — this gives $n_c^{\text{Fermi}} = 0$ (collective modes exist at any density).

**Gaussian beam:** The saddle-point approximation works cleanly for $\sigma_v \ll v_0$. The imaginary part is approximately Gaussian centered at the Doppler-shifted frequency $\omega = v_0 q$; the real part expresses via the Dawson function.

**Uniform beam:** The imaginary part is piecewise linear with two shell boundaries, the real part involves dilogarithms $\text{Li}_2$. The continuum has a gap $\omega_- > 0$ for $q < k^0_{\min}$, modifying the IVT argument for Theorem 2 slightly.

All three cases summarized in Table 2. The key result: the f-sum rule coefficient $c_0 = \Omega_p^2 = ne^2/(m_{\text{eff}}\epsilon_0)$ is always fixed by $n$ and $m_{\text{eff}}$ regardless of distribution shape. All higher dispersion coefficients depend on velocity moments.

### Theorem 2 — Existence Proof

Proved via the intermediate value theorem on $\epsilon(\omega, q) = 1 - V(q)\Pi^{00}_0(\omega, q, n)$:

- As $\omega \to \infty$: $\epsilon \to 1^-$ (positive).
- As $\omega \to \omega_+^+$: for the Fermi gas, $\chi_0 \to +\infty$ (van Hove), so $\epsilon \to -\infty$. For smooth distributions, $V\chi_0(\omega_+^+)$ grows with $n$ — above $n_c$, $\epsilon$ is negative there.

IVT guarantees at least one zero $\omega^\star \in (\omega_+, \infty)$. The alternative proof via the argument principle (winding number of $\epsilon$ around a contour) confirms uniqueness for small $|q|$ and topological stability. Critical densities: $n_c^{\text{Fermi}} = 0$, $n_c^{\text{Gauss}} = n_c^{\text{unif}} = m_{\text{eff}}\epsilon_0\sigma_v^2 q^2/e^2$ (Table 3).

### Nonlinear Collective Effects

The cubic vertex from the fermion triangle diagram:

$$g_{abc}(q_1, q_2, q_3) = V(q_1)V(q_2)V(q_3)\chi^{(3)}_0(q_1, q_2, q_3)$$

The three-wave cross-section scales as $\sigma_{\text{3-wave}} \propto r_s^4/k_F^2 \propto n^{-2}$: nonlinear effects weaken at higher density, validating the RPA in the collective regime. Spent the final evening session polishing all equations in LaTeX and verifying all table entries. Full theory section in good shape.

---

## March 4, 2026 — Class time + 6 hours outside of class
**Focus:** Part I Completion — Symmetry Analysis, Ward Identities, and Experimental Predictions

This session wrapped up the theoretical half of the paper by working out what the symmetry of the beam system forces to be true vs. what is left as something experiments have to measure. It also translated the theory into concrete, testable predictions for real accelerator facilities.

### In Class

Worked on the symmetry section draft during available class time.

### Outside of Class

Finished the theoretical half of the paper. The Ward identity section is the most formal part of the manuscript and also the most conceptually satisfying — deriving what symmetry constrains vs. what it leaves free tells you almost everything about what experiments can and cannot distinguish.

**Residual Symmetry Group:**

The beam geometry reduces the full Poincaré group to:

$$G = \mathbb{R}_z \times SO(2)_\phi \times \mathbb{R}_t$$

Translations along the beam axis, rotations about it, and time translations in steady state. Noether's theorem gives conservation of longitudinal momentum from $\mathbb{R}_z$.

**Theorem 4 (Dispersion Constraints from Ward Identities):**

The master Ward identity $q_\mu\Pi^{\mu\nu}(q) = 0$ implies $\omega\Pi^{00} = \mathbf{q}\cdot\boldsymbol{\Pi}^0$. In the dispersion expansion $\omega_\star^2 = \sum_{n\geq 0} c_n|\mathbf{q}|^{2n}$:

| Coefficient | Fixed by Ward? | Value | Determined by |
|---|---|---|---|
| $c_0 = \Omega_p^2$ | yes (f-sum rule) | $ne^2/(m_{\text{eff}}\epsilon_0)$ | $n$ and $m_{\text{eff}}$ only |
| $c_1$ | no | $\beta v_{\text{char}}^2$ | $\langle v^4\rangle/\langle v^2\rangle$ |
| $c_n\ (n \geq 2)$ | no | distribution-dependent | higher velocity moments |

The constructive proof: the Fermi gas gives $c_1 = 3v_F^2/5$ while a Gaussian beam with the same $n$ and $m_{\text{eff}}$ gives $c_1 = 3\sigma_v^2$. The Ward identity cannot distinguish these distributions, so $c_1$ is purely dynamical. Physical consequence: a single measurement of $\Omega_p$ cannot distinguish beam distributions, but higher-order dispersion measurements can — a concrete experimental prescription.

**Theorem 5 (Selection Rules):**

Each collective mode carries quantum numbers $(n_r, \ell, k_z)$. A probe with quantum numbers $(Q_z, L, \Omega)$ allows transition $(k_z, \ell, \omega) \to (k_z + Q_z, \ell + L, \omega + \Omega)$.

| Transition from $\ell = 0$ | Dipole ($L = \pm 1$) | Quadrupole ($L = 0, \pm 2$) |
|---|---|---|
| $\to \ell' = 0$ (breathing) | $\times$ | $\checkmark$ |
| $\to \ell' = \pm 1$ (kink) | $\checkmark$ | $\times$ |
| $\to \ell' = \pm 2$ (quadrupole) | $\times$ | $\checkmark$ |
| $\to \ell' = \pm 3$ (octupole) | $\times$ | $\times$ |

Dipole and quadrupole probes access complementary transitions — a practical guide for designing beam diagnostics at BL4S.

**Emergent Conformal Invariance at $n_c$:**

Near $n = n_c$, the effective Lagrangian reduces to $\phi^4$ theory. The two-loop beta function yields the Wilson–Fisher fixed point, placing the beam-plasma transition in the 3D Ising universality class:

| Quantity | $\varepsilon = 1$ estimate | Exact (3D Ising) |
|---|---|---|
| $\eta$ | 0.019 | 0.036 |
| $\nu$ | 0.627 | 0.630 |
| $\gamma$ | 1.235 | 1.237 |

This is the structural link back to the Prometheus series — the same universality class as the 2D Ising and J1-J2 work. The caveat about non-equilibrium driving is stated explicitly: the assignment assumes effective equilibrium near $n_c$ on the timescale of collective mode formation, standard for driven-dissipative quantum critical systems.

**Experimental Predictions (Section 6):**

Six signatures compiled for intermediate-energy facilities (10–100 MeV): scattering cross-section enhancement near the plasmon resonance, stopping power peaks at $\hbar\omega_p$, anomalous beam broadening $\langle\Delta p_\perp^2\rangle_{\text{coll}} \propto \sqrt{n - n_c}$ (kink at $n_c$), density-dependent resonances at $\omega_p \propto \sqrt{n}$, Friedel oscillations with $q = 2k_F$ wavevector, and the energy-loss sum rule. Spent the final hour going through each numerical estimate in Table 7 to verify orders of magnitude.

Part I complete.

---

## March 5, 2026 — Class time + 8 hours outside of class
**Focus:** Part II — Prometheus Adaptation and PIC Data Generation Pipeline

This session was about setting up the machine learning side of the paper — adapting the Prometheus VAE framework (originally built for magnetic phase transitions) to instead detect the onset of collective behavior in a simulated particle beam, using a quantity called the structure factor as the input signal. The evening was mostly debugging the particle-in-cell simulations that generate the training data.

### In Class

Worked through the motivation for unsupervised detection and began the Prometheus architecture section.

### Outside of Class

A long session — the afternoon was focused on the framework adaptation; the evening was almost entirely PIC simulation setup and debugging.

**Why the Previous Architecture Failed:**

Section 8.1 documents the prior failure: a bespoke architecture combining graph neural networks, transformers, and neural ODEs. The latent space collapsed to a degenerate point regardless of training configuration. This is a structural failure — architectures without an explicit information bottleneck provide no mechanism to prevent the encoder from ignoring the small density variations that carry the phase transition signal. The $\beta$-VAE fixes this by design: the KL penalty with weight $\beta$ forces the encoder to commit only the information most essential for reconstruction into the latent code $z$.

**Input Representation — Structure Factor $S(q)$:**

Rather than raw particle positions (variable in number $N$, not translation/rotation invariant), we use:

$$S(\mathbf{q}) = \frac{1}{N}\left\langle\left|\sum_{j=1}^N e^{i\mathbf{q}\cdot\mathbf{r}_j}\right|^2\right\rangle_{\hat{q}}$$

averaged over $N_{\text{dir}} = 50$ randomly sampled unit vectors $\hat{q}$ at each magnitude $q$, giving a fixed 256-dimensional vector regardless of $N$. Three advantages: fixed dimension, translation/rotation invariance by construction, and the phase transition signatures predicted in Part I leave specific marks in $S(q)$ — the plasmon resonance as an enhancement at $|q| \sim \Omega_p/v_{\text{char}}$, and the Kohn anomaly as a cusp at $q = 2k_F$.

**Prometheus Architecture (1D Convolutions):**

The only architectural change from the 2D Ising application is replacing 2D convolutions with 1D convolutions to accommodate the $S(q)$ vector:

```
S(q) [256-dim]
  Conv1D (1→32, kernel 7, stride 2) → ReLU
  Conv1D (32→64, kernel 5, stride 2) → ReLU
  Conv1D (64→128, kernel 5, stride 2) → ReLU
  Conv1D (128→256, kernel 3, stride 2) → ReLU
  → μ, σ² ∈ ℝ²   [latent dim dz = 2]

Decoder: mirrored transposed convolutions

β-ELBO:  E_{q_φ(z|x)}[log p_θ(x|z)]  −  β · D_KL(q_φ(z|x) ∥ p(z))
```

$\beta$ swept over $\{0.1, 0.5, 1.0, 2.0, 4.0, 8.0\}$. The optimal $\beta^\star = 0.1$ is lower than $\beta = 1.0$ for the Ising model because $S(q)$ already encodes density-density correlations, requiring less regularization to isolate the phase signal.

**Order Parameter Extraction:**

After training, the encoder's mean output $\mu_\phi(x)$ serves as the order parameter estimator. For $M$ configurations drawn at density $n$:

$$\Phi(n) = \frac{1}{M}\sum_{i=1}^M \|\mu_\phi(x_i)\|^2$$

The encoder never observes the density label $n$ during training, so any density-dependence in $\Phi(n)$ is learned purely from the reconstruction objective — no label leakage.

**PIC Simulation Setup:**

Three distributions:
- **Fermi gas ($T \to 0$):** All states filled inside $|k| \leq k_F = (3\pi^2 n)^{1/3}$.
- **Gaussian beam:** $f_0(k) = (2\pi\sigma_p^2)^{-3/2}\exp(-|k - k_0|^2/(2\sigma_p^2))$.
- **Uniform shell:** Momenta drawn uniformly from $k_{\min} \leq |k| \leq k_{\max}$.

For each distribution: 20 densities logarithmically spaced over $n \in [10^8, 10^{12}]$ cm$^{-3}$, 1000 independent snapshots per density, equilibration time $t_{\text{eq}} = 5\tau_p$. Total: $3 \times 20 \times 1000 = 60{,}000$ configurations. Spent the evening verifying that $5\tau_p$ is sufficient by running short test PIC runs at three representative densities and confirming the pair correlation function $g(r)$ stabilizes within $3\tau_p$.

**Note on the Fermi Distribution:**

At $n \sim 10^{12}$ cm$^{-3}$ and $E \sim 10$ MeV, the Fermi energy $E_F = \hbar^2(3\pi^2 n)^{2/3}/(2m) \sim 10^{-5}$ eV is negligible compared to the kinetic energy — a realistic beam at these parameters is non-degenerate. The Fermi distribution is included as a theoretical comparison case, since it tests Prometheus's ability to distinguish distributions with qualitatively different critical behavior ($n_c = 0$ vs. $n_c > 0$), and the Lindhard function and Kohn anomaly are analytically tractable in this limit. Stated explicitly in Section 9.1 to preempt referee confusion.

**Dynamic Structure Factor Dataset:**

A second dataset for collective mode characterization: 10 selected densities (5 below, 5 above $n_c^{\text{theory}}$ for each distribution), PIC simulations run for 100 plasma periods. The dynamic structure factor:

$$S(\mathbf{q}, \omega) = \frac{1}{NT}\left|\int_0^T e^{i\omega t}\rho_{\mathbf{q}}(t)\,dt\right|^2$$

evaluated via FFT along the time axis. Will be used to verify the Ward identity prediction (distribution-independent $\Omega_p$) through dispersion fitting of $\omega^2_{\text{pk}}(q) = \Omega_p^2 + \beta_c v_{\text{char}}^2 q^2$.

**Training Protocol:**

Adam optimizer, lr $= 3\times10^{-4}$ with cosine annealing over 200 epochs, weight decay $10^{-5}$, batch size 128, KL annealing over first 20 epochs (linear ramp $0 \to \beta$), gradient clip norm 1.0, 5 random seeds. All hyperparameters match the published Prometheus pipeline — this is a zero-modification transfer to a new physical domain, which is itself the methodological claim.

---

## March 6, 2026 — Class time + 10 hours outside of class
**Focus:** Results, Validation, Discussion, and arXiv Preparation

With the theory written and the simulations done, this was the day of actually running the model and checking whether all six of the paper's theoretical predictions came out confirmed — and then getting the manuscript submitted to arXiv. A long day but a satisfying one.

### In Class

Ran initial training passes and began writing the results section.

### Outside of Class

Longest single working day on the paper. Training ran in the background through the afternoon; the evening was spent on the discussion section, appendices, and LaTeX cleanup.

**Phase Transition Detection:**

Order parameter $\Phi(n)$ evaluated on the held-out test set at each density:

- **Fermi gas:** $\Delta\Phi = 0.04$ (flat) across four orders of magnitude in density. Consistent with $n_c \to 0$ — collective modes exist at all densities, no transition to detect.
- **Gaussian beam:** $\Delta\Phi_{\text{Gauss}} = 0.69$, monotonically decreasing.
- **Uniform beam:** $\Delta\Phi_{\text{unif}} = 0.73$, monotonically decreasing.

The qualitative difference between the Fermi curve (flat) and the Gaussian/uniform curves (monotonically decreasing) emerges entirely from the unsupervised training objective — no label leakage possible.

**KL Divergence Analysis:**

For Gaussian and uniform distributions, $\langle D_{\text{KL}}\rangle$ decreases monotonically from $\approx 52$ at low density to $\approx 45$ at high density: the encoder becomes increasingly certain as the system moves deeper into the single-particle phase. The Fermi distribution maintains constant $\langle D_{\text{KL}}\rangle \approx 42$ — no phase ambiguity when collective modes exist at all densities.

**$\beta$ Sweep:**

Optimal $\beta^\star = 0.1$, peak order parameter magnitude 9.75. Lower than $\beta = 1.0$ for the Ising model, as expected: $S(q)$ already compresses density-density correlation information.

**Ward Identity Verification (Validation Check 6):**

From PIC dispersion analysis of $S(\mathbf{q}, \omega)$: extracted $\Omega_p$ at a reference density for all three distributions by fitting the zero-wavevector intercept of the dispersion relation. Result: $\Omega_p$ is identical across all three distributions, as Theorem 4 requires. Distribution-independence emerges from the simulation data itself.

**Kohn Anomaly:**

For the Fermi gas at $n > n_c$: pair correlation function fitted to $C(r) \sim \cos(2k_F r)/r^3$; extracted oscillation wavevector $q_{\text{fit}}$ agrees with $q_{\text{theory}} = 2k_F$. Kohn anomaly cusp detected in $S(q)$ at the correct wavevector. ✓

**Validation Summary:**

| Prediction | Result | Theory | Status |
|---|---|---|---|
| Fermi: no phase transition | $\Delta\Phi = 0.04$ (flat) | $n_c \to 0$ | ✓ |
| Gaussian: phase transition | $\Delta\Phi = 0.69$ (decreasing) | $n_c \sim 10^7$ | ✓ |
| Uniform: phase transition | $\Delta\Phi = 0.73$ (decreasing) | $n_c \sim 10^7$ | ✓ |
| Ward identity | $\Omega_p$ identical for all distributions | $\Omega_p^2 = ne^2/(m\epsilon_0)$ | ✓ |
| Distribution discrimination | Three distinct $\Phi(n)$ curves | Different $n_c$ | ✓ |
| Kohn anomaly (Fermi) | Cusp at $q = 2k_F$ | Friedel oscillations | ✓ |

All six pass.

**Discussion — Limitations:**

Stated clearly: finite-size scaling to extract critical exponents would require $N > 10{,}000$ — the current $N \leq 4000$ gives only $\Delta\Phi \sim 0.7$ dynamic range over four orders of magnitude. Damping rate measurements would benefit from higher temporal resolution. The 3D Ising universality class assignment assumes effective equilibrium near $n_c$.

**Experimental Outlook:**

Most accessible signature is anomalous beam broadening. Measuring $\langle\Delta p_\perp^2\rangle$ as a function of beam density $n$ should reveal a kink at $n_c$ with $\sqrt{n - n_c}$ scaling — testable at existing facilities without a dedicated new experiment.

**Evening — arXiv Submission Prep:**

Spent roughly three hours on arXiv preparation: metadata, abstract optimization for the hep-ex/physics.acc-ph cross-list, author affiliations, acknowledgments, and checking the compiled PDF rendering of all tables and figures. The appendices (mathematical formulation of field-theoretic methods, Lindhard function details, RG analysis, numerical methods, convergence tables, Prometheus hyperparameters, PIC simulation parameters, code availability) all needed a final consistency pass. GitHub repositories at `YCRG-Labs/beam-qft` and `YCRG-Labs/prometheus-beam` set up and linked in Appendix H.

Submitted to arXiv.

---

## March 7, 2026 — 12 hours outside of class
**Focus:** Appendices — Mathematical Formulation, Lindhard Derivations, and RG Analysis

The appendices are the technical backbone of the paper — self-contained derivations that make the main claims verifiable without asking the reader to chase down a dozen other references. This day was spent writing them out in full: the field theory toolkit, the three Lindhard function calculations, and the renormalization group analysis that places the beam-plasma transition in a known universality class.

No class (Saturday). Full day on the paper appendices, which had been deferred to keep the main text moving.

### Appendix A — Mathematical Formulation of Field-Theoretic Methods

Appendix A is a self-contained mathematical reference for the field-theoretic tools used throughout Part I, written assuming no prior QFT exposure. This took most of the morning to get right — the challenge is being precise without being inaccessible. Covered:

**Function spaces and the action:** A field $\psi \in C^\infty(\mathbb{R}^{3+1}, \mathbb{C}^k)$; the action $S[\psi] = \int \mathcal{L}(\psi, \partial_\mu\psi)\,d^4x$; classical solutions as critical points $\delta S/\delta\psi = 0$.

**Gaussian functional integrals:** The central formula for real bosons:

$$\int_{\mathbb{R}^N} \exp\!\left(-\tfrac{1}{2}x^T A x + J^T x\right) d^N x = \frac{(2\pi)^{N/2}}{\sqrt{\det A}}\exp\!\left(\tfrac{1}{2}J^T A^{-1}J\right)$$

extended to the field-theoretic continuum. The Grassmann (fermionic) version gives $\det A$ rather than $(\det A)^{-1/2}$.

**Saddle-point expansion and one-loop approximation:** Expanding around the classical solution $\phi_0$ and retaining only the quadratic fluctuation term:

$$S^{(1)}_{\text{eff}} = S[\phi_0] + \frac{i\hbar}{2}\text{Tr}\ln\hat{D}$$

where $\hat{D} = \delta^2 S/\delta\phi^2|_{\phi_0}$. This is the formula underlying Theorem 1.

**Perturbative expansion and Feynman rules:** The $n$-point correlators via $e^{iS_{\text{int}}/\hbar}$ expansion. Each diagram $G = (V, E)$ contributes vertex factors and propagators $\hat{D}^{-1}(x,y)$; $L$ independent loops give the $L$-loop contribution.

**Ultraviolet divergences and renormalization:** Dimensional regularization (analytically continue $d = 4 \to 4 - \varepsilon$); the one-loop integral formula:

$$\int \frac{d^dk}{(2\pi)^d}\frac{1}{(k^2 + m^2)^n} = \frac{1}{(4\pi)^{d/2}}\frac{\Gamma(n - d/2)}{\Gamma(n)}\frac{1}{(m^2)^{n-d/2}}$$

Renormalization absorbs poles in $\varepsilon$ into counterterms.

**Ward identities from symmetry:** The change of variables $\psi \to \psi + \delta\psi$ in the functional integral yields $\langle\delta S/\delta\psi \cdot \delta\psi\rangle = 0$. For $U(1)$ gauge symmetry: $q_\mu\Gamma^\mu(q,p) = G^{-1}(p+q) - G^{-1}(p)$.

### Appendix B — Lindhard Function Calculation

Detailed derivation of $\Pi^{00}_0(q^\mu)$ for all three distributions, expanding on the summary in Section 4. The degenerate Fermi gas calculation is reproduced in full using the Sokhotski–Plemelj formula:

$$\lim_{\eta \to 0^+} \frac{1}{\omega + i\eta - x} = \mathcal{P}\frac{1}{\omega - x} - i\pi\delta(\omega - x)$$

to decompose real and imaginary parts. The Gaussian beam calculation via saddle-point expansion, and the uniform distribution calculation involving the dilogarithm identity $\text{Li}_2(e^{i\theta}) = \pi^2/6 - \theta(\pi - \theta)/2 + i(\text{Cl}_2(\theta)/2)$.

### Appendix C — Renormalization Group Analysis

Full two-loop RG calculation. The effective action at scale $\mu$:

$$S_{\text{eff}}[\Phi;\mu] = \int d^4x\left[\frac{1}{2}(\partial_\mu\Phi)^2 + \frac{\lambda(\mu)}{4!}\Phi^4 + \ldots\right]$$

Two-loop beta function: $\beta_{\lambda_4} = -\varepsilon\lambda_4 + 3\lambda_4^2/(4\pi)^2 - 17\lambda_4^3/(3(4\pi)^4)$.

Fixed points: (i) Gaussian: $\lambda_4^\star = 0$ (unstable); (ii) Wilson–Fisher: $\lambda_4^\star = (4\pi)^2\varepsilon(1 + 17\varepsilon/27)/3$ (fully stable). Anomalous dimensions at the WF fixed point: $\eta = \varepsilon^2/54$; $\Delta_{\Phi^2} = 2 - 2\varepsilon/3 - 8\varepsilon^2/81$ giving $\nu \approx 0.63$. The stress tensor trace $T^\mu_\mu = \beta(\lambda)\Phi^4$ vanishes at $\lambda = \lambda^\star$, confirming emergent conformal invariance at the critical density. Universality class: 3D Ising $(O(1))$.

---

## March 8, 2026 — 14 hours outside of class
**Focus:** Appendices — Numerical Methods, Convergence Tables, and Code Repository

This session was about proving that the numerical methods underlying the simulations are mathematically sound, documenting every hyperparameter and simulation setting so the results are fully reproducible, and making the code publicly available. It is the kind of work that rarely gets noticed but that makes or breaks a paper's long-term credibility.

No class (Sunday). The longest single day. Completed the numerical appendices and set up the public code repositories.

### Appendix D — Numerical Methods

Derived the stability and convergence guarantees for the finite-difference scheme used in the PIC code. The leapfrog discretization:

$$\frac{\Phi^{n+1}_i - 2\Phi^n_i + \Phi^{n-1}_i}{(\Delta t)^2} = c^2 \frac{\Phi^n_{i+1} - 2\Phi^n_i + \Phi^n_{i-1}}{h^2} + \text{source terms}$$

**CFL stability condition** from von Neumann analysis: $c\Delta t/h \leq 1/\sqrt{d}$.

**Energy stability:** Multiplying by $D_t\Phi$ and summing over grid points gives, via discrete integration by parts and Grönwall's lemma: $E^n \leq (E^0 + C_F T)e^{LT}$.

**Convergence:** Truncation error $|\tau^n_i| \leq C_\tau((\Delta t)^2 + h^2)$; applying the stability estimate to the error equation gives $\|\Phi^h - \Phi\|_\infty \leq Ch^2$ (second-order convergence).

For smooth configurations, spectral methods using Chebyshev polynomials achieve exponential convergence $\|\Phi^N - \Phi\|_{L^2} \leq C(T)\rho^{-N}$ if $\Phi(\cdot, t)$ is analytic in the Bernstein ellipse $E_\rho$ with $\rho > 1$.

### Appendices E — Finite Difference Convergence Tables

Verified the theoretical $O(h^2)$ error bound numerically by solving $\ddot{\Phi} = c^2\Delta\Phi - \lambda\Phi^3$ on $\Omega = [0,\pi]^d$ with Dirichlet boundary conditions using the leapfrog scheme. Parameters: $c = 1$, $\lambda = 1$, $T = 1$, CFL number $\nu = 0.9/\sqrt{d}$.

**$d = 1$ (Table 9):**

| $N$ | $\|e\|_\infty$ | Rate | $\|e\|_{L^2}$ | Rate |
|---|---|---|---|---|
| 16 | $4.54\times10^{-5}$ | — | $5.86\times10^{-5}$ | — |
| 32 | $6.62\times10^{-6}$ | 2.78 | $8.67\times10^{-6}$ | 2.76 |
| 64 | $1.33\times10^{-6}$ | 2.31 | $1.76\times10^{-6}$ | 2.30 |
| 128 | $3.48\times10^{-7}$ | 1.94 | $4.59\times10^{-7}$ | 1.94 |
| 256 | $6.49\times10^{-8}$ | 2.42 | $8.59\times10^{-8}$ | 2.42 |

**$d = 2$ (Table 10):**

| $N$ | $\|e\|_\infty$ | Rate | $\|e\|_{L^2}$ | Rate |
|---|---|---|---|---|
| 8 | $2.31\times10^{-4}$ | — | $3.80\times10^{-4}$ | — |
| 16 | $5.22\times10^{-5}$ | 2.15 | $8.44\times10^{-5}$ | 2.17 |
| 32 | $1.13\times10^{-5}$ | 2.21 | $1.82\times10^{-5}$ | 2.21 |
| 64 | $2.12\times10^{-6}$ | 2.42 | $3.41\times10^{-6}$ | 2.42 |

**$d = 3$ (Table 11):**

| $N$ | $\|e\|_\infty$ | Rate | $\|e\|_{L^2}$ | Rate |
|---|---|---|---|---|
| 8 | $2.51\times10^{-4}$ | — | $5.17\times10^{-4}$ | — |
| 16 | $4.79\times10^{-5}$ | 2.39 | $1.06\times10^{-4}$ | 2.29 |

All observed rates consistent with the theoretical $p = 2$ bound.

### Appendix F — Prometheus Hyperparameters (Table 12)

Compiled the complete hyperparameter specification. Values marked $\dagger$ taken without modification from the published Prometheus pipeline:

| Hyperparameter | Value | Notes |
|---|---|---|
| Latent dimension $d_z$ | 2$^\dagger$ | Matches Ising application |
| $\beta$ (nominal) | 1.0$^\dagger$ | Swept over $\{0.1, 0.5, 1.0, 2.0, 4.0, 8.0\}$ |
| Input dimension | 256 | $S(q)$ grid size |
| Encoder channels | 1, 32, 64, 128, 256 | 1D conv, stride 2 |
| Encoder kernel sizes | 7, 5, 5, 3 | |
| Decoder channels | 256, 128, 64, 32, 1 | 1D transposed conv |
| Activation | ReLU$^\dagger$ | All hidden layers |
| Optimiser | Adam$^\dagger$ | |
| Learning rate | $3\times10^{-4}$$^\dagger$ | |
| Weight decay | $10^{-5}$$^\dagger$ | |
| LR schedule | Cosine annealing$^\dagger$ | 200 epochs |
| Batch size | 128$^\dagger$ | |
| Epochs | 200$^\dagger$ | |
| KL annealing epochs | 20$^\dagger$ | Linear ramp $0 \to \beta$ |
| Gradient clip norm | 1.0$^\dagger$ | |
| Random seeds | 5 | For error bars |

### Appendix G — PIC Simulation Parameters (Table 13)

| Parameter | Value | Units |
|---|---|---|
| Particle number $N$ | 2000 (static) | — |
| Box geometry | Cubic, PBC | — |
| Density range | $10^8$–$10^{12}$ | cm$^{-3}$ |
| Densities per sweep | 20 (log-spaced) | — |
| Equilibration time $t_{\text{eq}}$ | $5\tau_p$ | — |
| Snapshots per density | 1000 | — |
| $S(q)$ grid points $N_q$ | 256 | — |
| Angular averaging $N_{\text{dir}}$ | 50 | — |
| Dynamic run duration | $100\tau_p$ | — |
| Dynamic snapshot interval | $0.1\tau_p$ | — |

The time step satisfies $\Delta t < 0.05\,\Omega_p^{-1}$ at all densities.

### Appendix H — Code Availability

Set up both public repositories and verified they are accessible:

- **Theoretical calculations:** https://github.com/YCRG-Labs/beam-qft
- **PIC simulations and Prometheus training:** https://github.com/YCRG-Labs/prometheus-beam

Uploaded all code, wrote README files for both repositories, added a `requirements.txt` with pinned dependencies, and verified that the Prometheus training script runs end-to-end from a fresh clone. The data generation scripts, trained model checkpoints, and a Jupyter notebook reproducing the main figures are all included.

### End-to-End Manuscript Check

Spent the final three hours of the day doing a full end-to-end read of the compiled PDF — all 27 pages plus appendices. Caught several small issues: two cross-references to equation numbers that had shifted during revision, a mismatch between the proof sketch of Theorem 2 and the caption of Table 3, and a missing factor of 2 in the energy-loss sum rule formula (Eq. 27) that had been correct in a draft but dropped in a copy-paste. All fixed. The manuscript is clean.

---

## March 10, 2026 — Absent

---

## March 11, 2026 — Excused Absence (CTSEF)
**Focus:** Connecticut Science and Engineering Fair

Excused from class to attend the Connecticut Science and Engineering Fair.

---

## March 12, 2026 — Class time
**Focus:** CERN BL4S Submission Video

The BL4S video is the pitch to CERN's beam line for schools program — a short video explaining what experiment we want to run, why the physics is interesting, and what we would actually measure. The challenge is taking a paper full of quantum field theory and making the core argument land for a technical but non-specialist audience.

Spent today's session completing the CERN BL4S submission video for the beam-plasma collective oscillations experiment: **https://www.youtube.com/watch?v=Sk7OsiN0t6g**

