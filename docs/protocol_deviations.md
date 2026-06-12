# Protocol V3 — deviation log

Required by `docs/protocol_v3_evaluation.md` (header): any deviation
from the pre-registered protocol is logged here with a justification.

---

## D1 — Task 8: per-scenario physics-seed noise amplitude (kelvin_helmholtz)

**Pre-registered text (§1.1):** "Physics seeds: ≥ 5 per (scenario, Re)
— perturbed initial conditions (amplitude noise at the V1 perturbation
level)."

**Implementation:** seeded band-limited Gaussian noise (spectral
low-pass, modes |k| ≤ 8, std normalised) added to (vx, vy), amplitude
0.1 = `init_kelvin_helmholtz` `noise_amplitude` (the V1 perturbation
level), followed by div-free projection. Band-limiting is required
because the V1 spectral projector is exact only away from the Nyquist
modes, and matches the large-scale structured character of V1's own
perturbation.

**Deviation:** for `kelvin_helmholtz` only, the amplitude is lowered
to **0.02** (other 7 scenarios keep 0.1).

**Justification:** at amplitude 0.1 the injected noise carries ≈ 5×
the fluctuation energy of KH's structured 0.1·sin perturbation
(noise on both velocity components over the full field vs an
enveloped single-component term). The phase-1b KH validation check —
fluctuating-KE growth > 1.1× over t ∈ [0, 1] — then measures the
viscous decay of the injected noise instead of the instability growth
and reads flat (0.99–1.00×) at every Re, despite physically sound runs
(div B ≈ 1e-4, monotone energy). Lowering the KH seed amplitude to
0.02 (≈ 20% of the structured perturbation's energy scale) keeps the
pre-registered validation observable meaningful while still producing
genuinely distinct trajectories. The four KH seed-1 trajectories
generated at 0.1 were discarded and re-generated at 0.02.

**Date:** 2026-06-12. **Decided by:** user (option selected from
explicit alternatives, including keeping 0.1 with documented failures).
**Scope:** data generation only; no metric, label, split or subset
changed.
