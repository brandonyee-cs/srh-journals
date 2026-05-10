# Research Journal 14
*NeurIPS 2026 Submission Sprint: Prometheus II, The Second Brain, MoltBook E&D*

---

## April 27, 2026 — Class time + 8 hours outside of class
**Focus:** Sprint Launch — Triage Across Three Papers

NeurIPS 2026 abstract deadline locked in for May 1, full submission May 6–7. Three papers in flight simultaneously: Prometheus II (dual-signal phase detection), The Second Brain (microbiome diffusion with River), and the MoltBook emergent coordination paper targeting the E&D track. CFGD-XGB is confirmed out of this cycle — moving to TMLR — so all bandwidth goes to these three. Laid out the honest state of each.

**Prometheus II:** The architecture is implemented and the 2D Ising, TFIM, and Vlasov–Poisson experiments are done. The J1-J2 finite-size slope diagnostic is underpowered and statistically inconclusive, which I decided early on to keep in the paper rather than hide. The Proposition 1 proof is complete. Outstanding: the abstract and introduction still frame the claim too strongly relative to what the experiments actually support — need to pull back to the dual self-consistency angle rather than claiming detection superiority. Also need to add Schindler et al. [2017] and Zhang et al. [2020] to the bib; they were cited in the submitted draft but somehow dropped from the BibTeX file.

**The Second Brain:** The main AGP result is solid — 1.4% sparsity deviation in the main comparison, 2.6% ± 0.5% across three seeds. SparseDOSSA2 is a stronger sparsity-only baseline (0.7%), and MIDASim leads on ecological distance metrics; the paper has to be honest about that hierarchy rather than overclaiming. The outstanding issue River flagged is that the SparseDOSSA2 NA warnings in the pipeline might be confounding the sparsity numbers — need to audit whether those NA values in the simulated count matrix are being treated as zeros or dropped. The PERMANOVA F = 64.29 is going in as an explicit limitation, not buried.

**MoltBook:** The evaluation framework is complete — three benchmark tasks with clean quantitative baselines. The silhouette inflation issue (0.91 is heavily driven by the 93.5% peripheral cluster) is stated explicitly in Section 2.2 as a known methodological constraint. The Cohen's d = −0.88 deficit in collaborative task resolution is the paper's most interesting result; it frames it as a hard benchmark target rather than a failure. This is targeting the E&D track specifically, so the framing is evaluation methodology as an object of study. Main outstanding item: the NeurIPS checklist needs a complete pass, and the compute resources section (Question 8) is currently [No] — need to add a one-sentence justification for the camera-ready note.

Wrote a task list for each paper and blocked the next eight days.

---

## April 28, 2026 — Class time + 9 hours outside of class
**Focus:** Prometheus II — Abstract Rewrite, Claim Calibration, Missing Bibliography

Spent most of the day on Prometheus II. The core problem was that earlier drafts were making too broad a claim: "Prometheus II detects phase transitions more accurately than simpler detectors." The experiments don't support that — PCA is nearly optimal on 2D Ising, and the value of the dual-branch design is the self-consistency criterion, not accuracy. Rewrote the abstract and introduction around the narrower and actually defensible claim.

The rewritten abstract centers on three things: (1) the architecture couples a β-VAE and an MAE through a shared encoder, producing two candidate transition estimates from one representation; (2) the dual-signal disagreement statistic $\Delta = |\hat{\theta}_c^{\text{VAE}} - \hat{\theta}_c^{\text{MAE}}|$ serves as an internal validation criterion; (3) on five benchmarks spanning lattice, field, and graph modalities, the two branches agree within a fixed finite-size tolerance on benchmarks with known transitions or instability regions, while the J1-J2 finite-size slope diagnostic is explicitly statistically inconclusive. That last clause is load-bearing — reviewers will check whether the paper acknowledges the negative result.

The Proposition 1 proof in Appendix A needed tightening. The claim is intentionally narrow: a constant encoder cannot be a global minimizer of the VAE reconstruction-KL objective under non-degenerate reconstruction assumptions. The proof sketch in Section 4 establishes this via a simple argument — a decoder receiving input-independent latents can only reconstruct an average or stochastic mixture, leaving positive expected squared reconstruction error whenever the data distribution has at least two distinct configurations with positive probability. An input-dependent posterior achieves strictly lower reconstruction loss with finite KL cost, so the constant solution is not globally optimal. The proof does not claim that stochastic optimization cannot converge to poor local optima, which would be false.

Added the two missing bibliography entries. Schindler, Regnault, and Neupert (2017), "Probing many-body localization with neural networks," Physical Review B; Zhang, Ginsparg, and Kim (2020), "Interpreting machine learning of topological quantum phase transitions," Physical Review Research. Both were cited in the submitted draft body but had been dropped from the BibTeX at some earlier revision. Verified the full citations against the published papers.

Ran the ablation experiments one more time on 2D Ising to verify the β sensitivity results in Table 4. β = 1 gives weak regularization and the VAE readout moves away from the MAE readout; β = 16 produces excessive compression and also increases disagreement; β = 4 gives the smallest ∆ among tested settings. Numbers consistent with earlier runs. The ablation story is clean: the self-consistency statistic is sensitive to representation quality and can be used for model selection, which is itself a concrete methodological contribution.

---

## April 29, 2026 — Class time + 10 hours outside of class
**Focus:** The Second Brain — SparseDOSSA2 NA Audit, Baseline Table Verification, PERMANOVA Framing

River and I had been back-and-forth on the SparseDOSSA2 NA warnings for about a week. Today I ran the audit properly.

The issue: SparseDOSSA2 generates count data using a negative binomial model fitted to real data. In some configurations, particularly with rare taxa that have very low prevalences, the fitting procedure produces NA values in the simulated count matrix. The question was whether our pipeline was treating those NAs as zeros (which would artificially inflate our sparsity deviation) or dropping those taxa from the comparison (which would give SparseDOSSA2 an unfair advantage). 

Dug into the pipeline code:

```python
# audit_sparsedossa2_nas.py
import numpy as np
import pandas as pd

def audit_sparsedossa2_output(sim_path):
    df = pd.read_csv(sim_path, index_col=0)
    
    na_count = df.isna().sum().sum()
    na_by_taxon = df.isna().sum(axis=0)
    
    print(f"Total NA entries: {na_count}")
    print(f"Taxa with any NA: {(na_by_taxon > 0).sum()}")
    print(f"Taxa entirely NA: {(na_by_taxon == df.shape[0]).sum()}")
    
    # Check what our pipeline does
    df_filled = df.fillna(0)
    df_dropped = df.dropna(axis=1)
    
    sparsity_fill = (df_filled == 0).mean().mean()
    sparsity_drop = (df_dropped == 0).mean().mean()
    
    return {
        'na_count': na_count,
        'sparsity_fill_zeros': sparsity_fill,
        'sparsity_drop_taxa': sparsity_drop
    }
```

Result: the pipeline was filling NAs with zero before computing sparsity metrics, which slightly inflates SparseDOSSA2's apparent sparsity deviation relative to filling-and-dropping. With fill-zeros, SparseDOSSA2 achieves 0.7% deviation; with drop-taxa, it achieves 0.6%. The difference is small enough that it does not change any claims in the paper — SparseDOSSA2 is the stronger baseline on sparsity either way. Added a footnote to Table 1 disclosing the NA treatment and the sensitivity. This is the kind of thing reviewers notice if it's not mentioned.

Verified the baseline table for MIDASim. The ecological distance scores in Table 2 — Bray-Curtis 0.0054, Jaccard 0.0012, UniFrac 0.0115 — are all markedly better than our model (0.0485, 0.0678, 0.0400) and SparseDOSSA2 (0.0495, 0.0367, 0.0435). The narrative in Section 5.2 now explicitly states that MIDASim leads on ecological distance metrics, SparseDOSSA2 leads on sparsity deviation and Jaccard distance, and our model achieves the best prevalence correlation (0.996) and competitive Bray-Curtis and UniFrac discrepancies. That three-way differentiation is what makes the comparison honest.

The PERMANOVA F = 64.29 required a careful rewrite. Earlier versions of the paper were softening this result in a way that read as evasive. The current framing in Section 5.2 and Section 6 explicitly states: "PERMANOVA remains able to distinguish generated from real AGP samples (F = 64.29), which we treat as an important limitation rather than evidence of indistinguishability." For contrast, SparseDOSSA2 achieves F = 5.37 and MIDASim achieves F = 3.09 — so the parametric simulators are considerably closer to indistinguishability. The narrow takeaway the paper makes is that the deep generative model matches parametric-level sparsity preservation and achieves the best prevalence correlation, not that it achieves distributional indistinguishability.

Wrote the Appendix E multi-seed section. Per-seed deviations: seed 42 gives 2.8%, seed 7 gives 1.9%, seed 123 gives 3.2%, for a mean of 2.6% ± 0.5%. These are run on the 15.2M-parameter full model; the ablation tables use the 593K-parameter comparison model and are clearly labeled as such throughout.

**Hours today:** 10 outside of class.

---

## April 30, 2026 — Class time + 10 hours outside of class
**Focus:** MoltBook E&D — Task 3 Extended Results, Section 7 Discussion, Checklist Pass

The MoltBook paper had the cleanest results of the three but needed the most work on framing, since the E&D track requires particularly careful self-reflection on evaluation methodology.

The Task 3 result is genuinely interesting: from 60,045 technical threads, only 164 collaborative events were identified, of which 11 achieved quality ≥ 0.5. Cohen's d = −0.88 against the single-agent baseline. The paper reframes this as a rigorous empirical target rather than a failure of the agents, which is the right framing for an evaluation methodology paper — we're not claiming agents are bad at collaboration, we're establishing a hard benchmark challenge that future coordination protocols must attempt to beat.

Filled in Table 10, the quality score component breakdown for successful vs. unsuccessful collaborative events. Successful events (n = 11) achieve: code presence 0.82, comment quality 0.65, test inclusion 0.45, syntax validity 0.73, total 0.68. Unsuccessful (n = 153): 0.14, 0.28, 0.02, 0.08, 0.15. The weight structure (code presence 0.30, comment quality 0.30, test inclusion 0.20, syntax validity 0.20) reflects the judgment that functional code output matters more than discussion quality in a software engineering context — stated explicitly in Section 5.1 so reviewers can assess whether they agree with the weighting.

Table 11 is the thread feature correlation table. Participant count correlates most strongly with success (Cohen's d = 1.18, p < 0.001), duration has a medium positive effect (0.72, p = 0.022), and network density has a small negative effect (−0.36, p = 0.254, not significant). Successful collaborations averaged 21.0 participants versus 9.4 for unsuccessful ones; mean duration 3.2 hours versus 1.7 hours. The positive participant count association suggests a critical mass of contributors may be necessary for coordination to yield benefits, which connects to Woolley et al.'s collective intelligence factor finding.

Section 7.2 on benchmark validity and generalizability needed significant expansion. Added explicit treatment of both construct validity (each of the three tasks is grounded in an established theoretical construct — role differentiation, contagion theory, cooperative task performance) and external validity (the MoltBook platform has idiosyncratic features like karma voting and Reddit-style community structure that may not generalize; baselines are platform-specific reference points rather than universal constants). Also added an explicit invitation for the community to apply the framework to other emerging agent interaction platforms.

Section 7.3 limitations is now a dedicated subsection with four bullet points: short observation window (three weeks insufficient for long-run processes), platform-specific mechanics, unobserved model identities (model architecture not recorded, prevents attribution of behavioral heterogeneity), and measurement assumptions (silhouette score conflates cluster separation with role diversity; n-gram cascade extraction misses semantic adoption; quality score weights are heuristic).

Ran the full NeurIPS checklist pass. All questions answered with one-to-two-sentence justifications. The [No] for Question 8 (compute resources) now reads: "Compute details are not explicitly reported. All analyses use standard statistical methods on tabular data. Full details will be added in the camera-ready version." That's an acceptable justification per the NeurIPS guidelines.

**Hours today:** 10 outside of class.

---

## May 1, 2026 — Abstract Deadline — 11 hours outside of class
**Focus:** Abstract Submissions, Final Structural Audit Across All Three Papers

Abstract submissions due today. Submitted all three to OpenReview.

Prometheus II abstract went through four drafts before I was happy with it. The key constraint is that it cannot claim detection superiority — that's not what the experiments show, and a reviewer will catch it immediately. Final version centers on: (1) the dual-branch self-supervised architecture with one shared encoder, (2) the disagreement statistic $\Delta$ as an internal validation criterion, (3) agreement on known benchmarks within finite-size tolerance, (4) explicit acknowledgment that the J1-J2 slope diagnostic is statistically inconclusive. Abstract is 192 words.

The Second Brain abstract was already tighter from River's earlier revisions. Final structure: the sparsity barrier motivates the work, the two sparsity mechanisms are named (prevalence-aware bias initialization, hard sparsity loss), the full model achieves parametric-level sparsity preservation (1.4% deviation in main comparison, 2.6% ± 0.5% across three AGP seeds), the three-way comparison is stated honestly (SparseDOSSA2 best on sparsity deviation, MIDASim best on ecological distance metrics, our model best on prevalence correlation). PERMANOVA distinguishability is acknowledged as an important limitation in the abstract itself, not just the conclusion. Abstract is 210 words.

MoltBook abstract: the key is that the primary contribution is the evaluation methodology itself, not a new multi-agent architecture. The three tasks and their baselines are summarized quantitatively: silhouette 0.91 (with acknowledgment of inflation), α = 2.57, Cohen's d = −0.88. The abstract explicitly frames the negative result on Task 3 as a "hard benchmark challenge" and the paper as enabling the community to compare future multi-agent protocols. 189 words.

After submitting, did a full structural cross-check across all three papers to catch inconsistencies. Found two in The Second Brain: the non-Western compendium row in Table 3 uses the 593K-parameter comparison model, but an earlier paragraph was ambiguous about this. Added a dagger footnote and explicit note in the caption. The ablation tables also use the 593K model and this is now consistently flagged throughout with a note that the ablation ∆MFD values should not be compared numerically with Table 3.

No inconsistencies found in Prometheus II or MoltBook. Both papers have consistent notation throughout.

**Hours today:** 11 outside of class.

---

## May 2–4, 2026 — 8 hours total outside of class
**Focus:** Final Figure and Table Passes, Related Work Expansions

Spread across three days, worked through remaining lower-priority items on all three papers.

**Prometheus II:** Verified all five benchmarks produce reproducible results with the specified hyperparameters. The default training configuration is Adam, lr = 10⁻⁴, batch size 256, 100 epochs, d = 128, dz = 16, β = 4, λ = 1. The J1-J2 multi-size results in Table 5 report mean ± one standard deviation across three seeds at each of L ∈ {16, 32, 64}. At L = 64, $\hat{\theta}_c^{\text{VAE}} = 0.497 \pm 0.082$ and $\hat{\theta}_c^{\text{MAE}} = 0.503 \pm 0.089$, giving ∆ = 0.087 ± 0.060, which is below the ε = 0.1 tolerance. The finite-size disagreement slope is d∆/d log L = −0.020 ± 0.058 with 95% CI = [−0.113, 0.063]; because this interval includes both signs, the slope-based transition-order classification is inconclusive and is stated as such.

Added a paragraph to the related work section clarifying the distinction between the current method and ordinary ensembling. An ensemble of identical autoencoders mainly measures optimization variability. The two branches in Prometheus II induce different failure modes — the VAE branch compresses global information into a stochastic latent code and is sensitive to changes in the geometry of the posterior means; the MAE branch measures the ability to infer missing local information from context and is sensitive to scale-dependent reconstructability. Agreement is meaningful precisely because the two branches are not interchangeable copies.

**The Second Brain:** Added the cross-population robustness subsection in more detail. The non-Western compendium has substantially higher held-out sparsity (0.899 vs. 0.646) and lower alpha diversity (1.95 vs. 2.88), representing a qualitatively harder generalization target. A model from the same sparsity-preserving framework, retrained on the compendium, achieves sparsity 0.939 (real: 0.899), alpha diversity 2.01 (real: 1.95), and prevalence correlation 0.788. The AGP row in the cross-population table uses the smaller comparison model (593K parameters), not the full 15.2M-parameter model — this distinction is now clearly flagged in the table caption with a dagger.

Section 5.6 on the alpha diversity gap now includes a concrete hypothesis about mechanism: the log-normal decoder yields abundances that are too uniform across present taxa, under-representing the heavy-tailed dominance patterns seen in real samples. Temperature scaling and dominance regularization were attempted and failed. Heavier-tailed abundance parameterizations, normalizing-flow decoders, or hierarchical discrete latents are identified as the most promising future directions.

**MoltBook:** The related work section was expanded with tighter integration of the theoretical constructs underlying each task. Task 1 operationalizes sociological role differentiation through network clustering; Task 2 operationalizes contagion theory through survival analysis; Task 3 operationalizes cooperative task performance through a composite quality score. Cited Borgatti and Everett (2000) on core-periphery structures, Centola and Macy (2007) on complex contagion, Dodds and Watts (2004) on the saturating contagion model, and Woolley et al. (2010) on collective intelligence factors. These aren't just decorative citations — each is doing work in motivating why the corresponding benchmark task measures a meaningful construct.

Verified the power-law parameter estimation details in Appendix B. Maximum likelihood estimation yields α̂ = 2.569 ± 0.016, xmin = 104, KS statistic D = 0.020. Likelihood ratio tests comparing power-law to exponential: log-LR = 6109.6, p < 0.001 (power-law strongly favored). vs. lognormal: log-LR = 0.22, p = 0.030 (power-law marginally favored). vs. truncated power-law: inconclusive. These numbers are stable and consistent with the methodology of Clauset, Shalizi, and Newman (2009).

---

## May 5–7, 2026 — AP Examinations (Out of Class)
**Focus:** Final submission cleanup and checklist verification outside of class.

Away from class May 5, 6, and 7 for AP exams. Continued working outside of class across these days on final submission verification — confirming OpenReview metadata, checking that all supplementary materials compiled correctly, and verifying the NeurIPS checklist responses across all three papers. Papers were in submittable state by May 4; the remaining work during the AP days was cleanup and confirmation rather than substantive revision.

---

## Submission Notes (Filed May 4, 2026)

**Prometheus II: "Dual-Signal Self-Consistency: A Principle for Self-Validating Unsupervised Phase Detection"**

Submitted to NeurIPS 2026 main track. The paper's central claim is narrow and deliberately calibrated: internal dual-signal consistency supplies a check for unsupervised scientific representation learning that a single-branch detector cannot provide. The paper does not claim that Prometheus II is always more accurate than simpler detectors — on 2D Ising, PCA is nearly optimal because its leading component closely tracks magnetization, and Prometheus II is not presented as a replacement for that known solution. The value is the presence of two agreeing estimates. On the frustrated J1-J2 benchmark, both branches concentrate near J2/J1 ≈ 0.5 while PCA on the same configurations is displaced to 0.636 ± 0.066. The finite-size disagreement slope is insufficient to resolve transition order. All of this is in the paper.

The checklist is clean. Theory and proofs: Proposition 1 is clearly stated with explicit assumptions and a full proof in Appendix A. Reproducibility: the encoder architecture (four-block convolutional, stride-two convolutions, batch normalization, ReLU, global average pooling for lattice/field data; five-layer GIN with sum pooling for graphs) is specified in sufficient detail for reimplementation. Training: Adam, lr = 10⁻⁴, batch size 256, 100 epochs, d = 128, dz = 16, β = 4, λ = 1. Statistical significance: J1-J2 results report mean ± standard deviation across three seeds at each of three system sizes; the finite-size slope includes a bootstrap 95% CI. Compute: training requires approximately 20 hours per benchmark on a single A100 GPU.

---

**The Second Brain: "The Second Brain: Diffusion Models for Realistic Human Microbiome Generation"**

Submitted to NeurIPS 2026 main track. The central claim is narrow: first deep generative model to achieve parametric-level sparsity preservation on the American Gut Project, matching the operational threshold previously met by specialized simulators SparseDOSSA2 and MIDASim. The paper does not claim the model is the sparsest overall method — SparseDOSSA2 achieves 0.7% deviation versus our 1.4% in the main comparison — and does not claim distributional indistinguishability, since PERMANOVA F = 64.29 shows generated communities remain statistically distinguishable from real ones.

The three-way comparison is: among methods that pass the 10% operational sparsity threshold, MIDASim achieves the best ecological distance scores, SparseDOSSA2 is best on sparsity deviation, and our model achieves the best prevalence correlation (0.996) and competitive Bray-Curtis and UniFrac discrepancies. That differentiated comparison is what makes the contribution honest.

Code will be released upon acceptance. Data (American Gut Project, Human Microbiome Compendium) are publicly available. All baselines are cited and implemented. No human subjects research was conducted — data are pre-existing publicly available de-identified datasets.

---

**"Benchmarking Emergent Coordination in Large-Scale LLM Populations: An Evaluation Framework on the MoltBook Archive"**

Submitted to NeurIPS 2026 Evaluations and Datasets track. The primary contribution is the evaluation methodology itself, not a new multi-agent architecture. Three benchmark tasks — structural role evaluation (Task 1), information diffusion auditing (Task 2), decentralized collaboration benchmarking (Task 3) — are applied to the MoltBook Observatory Archive, a dataset of 2.73M interactions among 90,704 autonomous agents.

The quantitative baselines established: silhouette coefficient 0.91 (with inflation acknowledged), cascade power-law exponent α = 2.57 (95% CI [2.54, 2.60]), Cox hazard ratio 0.528 for time-to-adoption, and Cohen's d = −0.88 for collaborative vs. single-agent task resolution. The negative Task 3 result is framed as a rigorous empirical target for future coordination protocols rather than a failure — this framing is appropriate for an evaluation methodology paper.

Methodological constraints are stated explicitly in Section 2.2: silhouette inflation from the homogeneous peripheral cluster, baseline selection confound in Task 3 (collaborative threads likely represent intrinsically harder tasks), and cascade identification ambiguity (n-gram extraction captures textual repetition but misses semantic adoption). The three-week observation window is insufficient for long-run coordination dynamics, and model identities are not recorded so architectural effects cannot be separated from platform dynamics. All of this is in the paper.

Data: MoltBook Observatory Archive is publicly available on Hugging Face. Evaluation framework code is in supplementary materials and will be open-sourced upon publication. The dataset contains only AI agent interactions with no human subjects or personally identifiable information.
