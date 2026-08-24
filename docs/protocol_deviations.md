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

**Addendum 1 (same day):** the initial mechanism stated above
("injected noise masks the check observable") was incomplete: the
0.02 rerun still read growth ≈ 1.00, which exposed the check bug
logged as D2 below. With the CORRECTED observable (D2), the amplitude
effect is confirmed independently at N=64/Re=400: seed-0 growth =
1.42×, seed-1 at amplitude 0.02 = 1.13× (passes), seed-1 at amplitude
0.1 = 0.82× (fails — injected fluctuation energy ≈ 20× the structured
KH perturbation, growth diluted below threshold).

**Addendum 2 (same day, final):** at N=256 the corrected observable
reads seed-0 = 1.41–1.43× and seed-1 at amplitude 0.02 = 1.05–1.08× —
marginally below the 1.1 threshold (dilution model calibrated on the
data: injected energy a², structured ≈ 2.5e-4, noise decay ≈ 0.84).
**Final D1 value: KH amplitude = 0.005** (predicted growth ≈ 1.37×,
comfortable margin for the future ≥5-seed extension of §1.1; same
level as V1's `init_lamb_oseen_vortex` noise_amplitude=0.005, so the
value has V1 provenance). User-selected from explicit alternatives
(0.005 / 0.01 / keep 0.02 with documented failures). The 0.02 KH
seed-1 trajectories were discarded and re-generated at 0.005.

---

## D2 — Phase-1b KH validation check: corrected observable (v3 copy)

**Pre-registered text (§1.1):** trajectories "validated by phase 1b
checks (div B, OT energy-decay window, KH growth)".

**Bug found (V2 code, `phase1b_dns_validation.fluctuating_KE`):** the
fluctuating kinetic energy subtracts the mean over **axis 1 (Y)** —
but the KH base flow `v_flow(Y)` varies along Y and is invariant along
X (axis 0), so the subtraction removes nothing of the base profile.
Ep is then dominated by the base-flow variance (≈ 0.341 vs a true
perturbation energy ≈ 2.5e-4 at t=0), and the growth ratio reads
≈ 1.00 for ANY trajectory — including unperturbed seed-0 runs
(verified: fresh N=64 seed-0 run reads 0.994× with the 1b observable
and 1.42× with the corrected one). The check as published could never
detect KH growth; its past "pass/fail" status is uninformative.

**Repair (per §8.2, V2 read-only):** `study/v3/t8_dns_extension.py`
implements `fluctuating_ke_fixed` (subtract the X-average, axis 0) and
`check_kh_fixed` (identical windows t ∈ [0,0.2] vs [0.8,1.2] and
identical criterion growth > 1.1×; only the observable is fixed).
`phase1b_dns_validation.py` is untouched and still runnable for
regression. All Task-8 KH validations use the corrected check; a
`--validate-only` mode re-validates existing files.

**Scope:** validation observable implementation only; the check's
pre-registered intent (KH growth), windows and threshold are
unchanged. No metric, label, split or subset of the evaluation layer
is affected.

**Date:** 2026-06-12.

---

## D3 — Task 6 (vérité terrain dynamique) : quatre écarts, tous assumés

**Statut : appliqué.** Module `study/pipeline/dynamic_patch_labels.py`.

| point | protocole | ici | raison |
|---|---|---|---|
| chemin | `study/v3/t6_dynamic_gt.py` | `study/pipeline/dynamic_patch_labels.py` | le dépôt a été réorganisé depuis la rédaction du protocole ; le `study/phase2_hard_patches.py` qu'il cite est aujourd'hui `study/pipeline/hard_patch_labels.py` |
| résolution du pilote | N=128 | **N=96** | aucun artefact DNS N=128 n'existe dans le dépôt (N ∈ {64, 96, 256}) ; N=96 avec dim=8 respecte la contrainte `dim ≤ N/8` |
| format de sortie | « mirrors the phase-2 format so phase-11 builders accept it as a **drop-in** label source » | clés explicites `d_errors` / `d0_errors` / `amplification` ; `l2_errors` contient le label **statique** | écrire le label dynamique sous la clé `l2_errors` produirait un artefact de la forme phase 2 dont une clé désigne autre chose que son nom — la classe de défaut que `CODE_REVIEW.md` retient comme la seule qui compte. Un consommateur qui veut le label dynamique le nomme. |
| nom de fichier | non spécifié | `δt` inclus dans le nom | δt décide si le label dit quelque chose (ρ = 0,982 à 0,1, 0,596 à 2,0 sur OT) ; deux horizons partageant un nom s'écraseraient en silence |

### Et un écart que la mesure impose au protocole, pas l'inverse

Le protocole fixe `δt = one hybrid step (0.1)` et pose comme seul critère
d'acceptation *« sanity check Spearman(d_i, e_i) > 0 reported »*.

**Le critère est satisfait et ne contrôle rien** : à cet horizon ρ ≥ 0,98 sur
les quatre scénarios, c'est-à-dire que le label dynamique est une
renumérotation monotone du label statique. Un contrôle qu'un label redondant
passe haut la main n'est pas un contrôle.

L'horizon défendable n'est pas un nombre de pas hybrides mais une échelle
physique — le temps de traversée d'un patch,
`t_x = 2π / (dim · (v+b)_rms)` — soit 0,41 à 0,88 à `dim = 8`. À δt = 0,1 la
perturbation parcourt 0,11 à 0,25 d'une largeur de patch.

**Rien n'est décidé ici** : changer l'horizon du protocole est une décision de
campagne. La mesure est publiée dans `RESULTS.md` et le module accepte
`--delta-t`.

