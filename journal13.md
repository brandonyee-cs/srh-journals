# Research Journal 13
*PAI26 Submission Sprint*

---

## April 13, 2026 — Class time
**Focus:** Prometheus — Introduction and Methods Trimming

Identified three YCRG papers to submit to PAI26 (deadline May 1, Paul Brest Hall, Stanford June 10–12). Started on Prometheus since it needs the most work — currently 5 pages, PAI26 cap is 4.

Spent class time on the Introduction and Methods sections. Compressed the opening motivation from two paragraphs to one, keeping the core contrast: supervised ML requires labels, can't extract exponents, can't identify novel phases. Cut the 3D convolutional architecture prose to two sentences pointing at the key numbers; the Q-VAE fidelity loss stayed since it's the methodological novelty.

---

## April 14, 2026 — Class time
**Focus:** Prometheus — Related Work and Results Prose

Cut the Related Work section from three separate quantum ML paragraphs into one. Preserved the key contrast — IRFP detection had never been demonstrated without explicit SDRG or quantum Monte Carlo with manual scaling analysis — because it's doing real scientific work.

Trimmed results prose by removing transition sentences that restated what the tables already showed. All numbers untouched: Tc/J = 4.511 ± 0.005, r = 0.997, ψ = 0.48 ± 0.08, ∆χ² = 12.3, p < 0.001. Refused to cut one sentence: "this detection was achieved without providing the model any information about the expected functional form or the existence of IRFP physics." That's the thesis of the paper.

Hit exactly 4.0 pages.

---

## April 15, 2026 — Class time
**Focus:** Simulation Budget Fallacy — Perspectives Track Framing

Paper was already 4 pages so the work was framing, not cutting. Rewrote the abstract opening to make the physical context explicit — reservoir engineers, climate model emulators, and manufacturing simulation teams are the practitioners being failed by the ϱ = 0 evaluation regime.

Added a half-sentence in Section 3 connecting the Timescale Alignment Principle to Lie-Trotter splitting, which physicists already use in their numerical PDE codes. Tables 1, 2, and 3 are all load-bearing and needed no changes.

---

## April 17, 2026 — Class time
**Focus:** MSAT — Proposition Verification

Worked through Propositions 4.1 and 4.2. For 4.1 (FNO error on irregular domains): κ jump discontinuities → |c_k| = O(k⁻¹) → Parseval gives Ω(κ/K). With κ = 18 and K = 16 on Heat2D-CG, error is bounded away from zero regardless of model depth — explains the 3.6× empirical gap. Tight.

For 4.2 (attention error): rewrote the proof sketch to make explicit that exponential convergence requires Sobolev regularity in the interior as an assumption, not a derived result. Verified Conjecture 4.3 is labeled as a conjecture throughout.

---

## April 20, 2026 — Class time
**Focus:** Jargon Audit

PAI26 requires papers to be approachable by non-experts. Did a pass on all three.

Prometheus: defined "universality class" parenthetically, added a half-sentence on "Binder cumulant crossing," distinguished "latent susceptibility" from thermodynamic susceptibility.

Simulation Budget Fallacy: added a parenthetical defining "masked autoencoder," clarified "operator splitting" as the numerical PDE technique physicists already use.

MSAT: defined "Gibbs phenomenon" inline in the Proposition 4.1 proof sketch, added one-clause characterizations of the Kuramoto-Sivashinsky equation and lid-driven cavity benchmark.

About 15 fixes across three papers, none touching any scientific claim.

---

## April 23, 2026 — Class time
**Focus:** Anonymization and OpenReview Setup

Verified anonymization on all three papers — author blocks already set to "Anonymous Author(s)" from the NeurIPS cycle, no repo links, no acknowledgments. Self-citation check: Prometheus keeps "Yee et al., 2026" because omitting it makes the paper incomprehensible; Simulation Budget Fallacy already uses "Anonymous [2026]" for the PI-JEPA concurrent preprint.

Registered on PAI26 OpenReview under my Stanford affiliation. Checked the organizing committee for conflicts — none. Track assignments: Prometheus → Research, Simulation Budget Fallacy → Perspectives, MSAT → Research.

---

## April 24, 2026 — Class time
**Focus:** Final Read and Submission

Cold read on each paper. Prometheus holds — the DTFIM Table 2 tells a clean monotonic story and the "first unsupervised detection" claim is defensible. Simulation Budget Fallacy argument flows cleanly; limitation about the missing timescale-agnostic SSL baseline is disclosed. MSAT's main finding and ablation are clearly stated, KS physics-constraint result flagged as open.

Submitted all three to PAI26 OpenReview. Notifications May 10.
