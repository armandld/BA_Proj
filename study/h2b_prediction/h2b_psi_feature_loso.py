#!/usr/bin/env python3
"""
V3 Task 5 - Rerun corrige de phase 11E (protocole v3, prerequis de la
section 3 ; retire ou ressuscite psi). Copie de phase11e_v1h_loso.py
avec quatre changements :

  (1) parametres V1 de study/config.py TRAINED_* (essai Optuna #4 :
      beta=9.94, thr=0.1496) ; l'execution codee en dur a l'essai #85
      (beta=0.5495, thr=0.3044) est CONSERVEE comme bras de comparaison,
      et la divergence est journalisee ;
  (2) psi SIGNE : score_aug = clip(score + 0.5*sin(psi), 0, 1)
      (signe preserve : un amortissement reduit l'urgence). La
      combinaison des deux composantes d'arete preserve le signe :
      psi = celle de plus grande amplitude (analogue signe du
      max(|psi_h|, |psi_v|) de phase 11e). S'y ajoute la variante sans
      parametre compute_psi_v2 (HamiltParams_v2), appliquee a chaque
      composante puis combinee de la meme facon. Le bras "legacy"
      reproduit la formule |psi| de phase 11e (jamais utilisee au-dela
      de ce bras de comparaison) ;
  (3) les deux agregations : block_avg (B1, convention V1) et block_max
      (B2, convention V2) du MEME champ de score fin ;
  (4) bootstrap par blocs trajectoire (Task 3) : F1 par trajectoire
      (scenario, Re), chaque trajectoire evaluee sous le fold LOSO qui
      tient son scenario ; deltas apparies vs v1-classique (sans psi),
      CI 95 % par percentiles, fraction de trajectoires avec delta > 0.

Le label, les folds, l'ordre des donnees, les grilles de seuil et le
pool d'entrainement sont ceux de phase 11e (le bras legacy block_avg a
beta=0.5495 doit reproduire les F1_v1_psi publies).

Metriques : F1 uniquement (copie fidele de phase 11e) ; le test de psi
en metriques continues CE(b) arrive au niveau 2 (Task 7).

Sortie : results/t5_v1_psi_loso_N{N}_dim{D}.npz
Usage :
  python study/v3/t5_v1_psi_loso.py --N 256 --dim 4
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from h2b_feature_selection import git_commit_hash
from stats import paired_delta_bootstrap

# essai Optuna #4 (study/config.py TRAINED_*) vs essai #85 (code en dur
# dans phase11e_v1h_loso.py) — divergence journalisee dans main()
from config import TRAINED_BETA as BETA_TRIAL4          # 9.94
from config import TRAINED_THRESHOLD as THR_TRIAL4      # 0.1496
BETA_TRIAL85 = 0.5495366256460598
THR_TRIAL85 = 0.304445558422031


# -------------------------------------------------------------------
# Helpers purs (numpy seul, testables sans qiskit)
# -------------------------------------------------------------------

def signed_combine(psi_h, psi_v):
    """Combinaison preservant le signe : composante de plus grande
    amplitude (analogue signe du max(|psi_h|, |psi_v|) de phase 11e ;
    egalite -> composante horizontale)."""
    return np.where(np.abs(psi_h) >= np.abs(psi_v), psi_h, psi_v)


def psi_signed_v1(dphi_h, dphi_v, beta):
    """psi signe, machinerie V1 : psi_{h,v} = (pi/2) tanh(beta dPhi/avg),
    normalisation partagee avg = <|dphi_h|+|dphi_v|>/2 (phase 11e)."""
    avg = float(np.mean(np.abs(dphi_h) + np.abs(dphi_v))) / 2.0
    if avg < 1e-12:
        return np.zeros_like(dphi_h)
    psi_h = (np.pi / 2.0) * np.tanh(beta * dphi_h / avg)
    psi_v = (np.pi / 2.0) * np.tanh(beta * dphi_v / avg)
    return signed_combine(psi_h, psi_v)


def psi_abs_v1(dphi_h, dphi_v, beta):
    """|psi| de phase 11e (bras de comparaison legacy uniquement)."""
    avg = float(np.mean(np.abs(dphi_h) + np.abs(dphi_v))) / 2.0
    if avg < 1e-12:
        return np.zeros_like(dphi_h)
    psi_h = (np.pi / 2.0) * np.tanh(beta * dphi_h / avg)
    psi_v = (np.pi / 2.0) * np.tanh(beta * dphi_v / avg)
    return np.maximum(np.abs(psi_h), np.abs(psi_v))


def score_aug_signed(score, psi):
    """Formule pre-enregistree : clip(score + 0.5*sin(psi), 0, 1)."""
    return np.clip(score + 0.5 * np.sin(psi), 0.0, 1.0)


def score_aug_legacy(score, psi_abs):
    """Formule phase 11e : clip(score + |psi|/(pi/2), 0, +inf)."""
    return np.clip(score + psi_abs / (np.pi / 2.0), 0.0, None)


def block_agg(f, dim, how):
    """Agregation (N, N) -> (dim, dim) : 'avg' (B1) ou 'max' (B2)."""
    ps = f.shape[0] // dim
    blocks = f.reshape(dim, ps, dim, ps)
    return blocks.mean(axis=(1, 3)) if how == "avg" \
        else blocks.max(axis=(1, 3))


# noms des variantes ; cellules 2x2x2 = {signed, legacy} x {b4, b85}
# (x {avg, max} a l'evaluation) ; psi_v2 est sans parametre
VARIANTS = [
    "v2-classical",
    "v1-classical (no psi)",
    "v1+psi signed b=9.94 (trial4)",
    "v1+psi signed b=0.5495 (trial85)",
    "v1+psi legacy-abs b=9.94 (trial4)",
    "v1+psi legacy-abs b=0.5495 (11e repro)",
    "v1+psi_v2 signed (param-free)",
]
AGGS = ("avg", "max")


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def _gather(by_scene, dim, max_snaps):
    """{scenario: {re: list par snapshot de dict(y, scores)}} ;
    scores = {(variant, agg): (dim*dim,)}. Ordre temporel par config,
    psi calcule entre snapshots consecutifs de la sous-sequence
    (convention phase 11e : pj = idx[k-1])."""
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.HamiltParams_v2 import compute_psi_v2
    from exact_diagonalisation import build_patch_hamiltonian
    from h2b_v1_hamiltonian_loso import v1_state

    am = AngleMapper()
    out = {}
    for sc, rows in by_scene.items():
        out[sc] = {}
        for re, dns_path, patches_path in rows:
            dns = np.load(dns_path)
            patches = np.load(patches_path)
            vx_all = dns["vx"].astype(np.float64)
            vy_all = dns["vy"].astype(np.float64)
            Bx_all = dns["Bx"].astype(np.float64)
            By_all = dns["By"].astype(np.float64)
            N = vx_all.shape[1]
            l2_all = patches["l2_errors"]
            l2_thr = float(patches["l2_threshold"])
            n_snaps = len(vx_all)
            step = max(1, n_snaps // max_snaps)
            idx = list(range(0, n_snaps, step))[:max_snaps]

            cfg_snaps = []
            phi_prev = None
            for si in idx:
                vx, vy, Bx, By = (vx_all[si], vy_all[si],
                                  Bx_all[si], By_all[si])
                state = v1_state(vx, vy, Bx, By)
                cls = AngleMapper.classical_score(state)
                phi = am.compute_stress_flux(state)

                # champ de score fin par variante
                _, _, v2_full = build_patch_hamiltonian(
                    vx, vy, Bx, By, N, dim, re,
                    threshold_amr=0.15, use_v2=True, c_bias=1.0)
                fields = {"v2-classical": v2_full,
                          "v1-classical (no psi)": cls}
                if phi_prev is None:
                    z = np.zeros_like(cls)
                    fields[VARIANTS[2]] = score_aug_signed(cls, z)
                    fields[VARIANTS[3]] = score_aug_signed(cls, z)
                    fields[VARIANTS[4]] = score_aug_legacy(cls, z)
                    fields[VARIANTS[5]] = score_aug_legacy(cls, z)
                    fields[VARIANTS[6]] = score_aug_signed(cls, z)
                else:
                    dph = phi["phi_horizontal"] - phi_prev["phi_horizontal"]
                    dpv = phi["phi_vertical"] - phi_prev["phi_vertical"]
                    fields[VARIANTS[2]] = score_aug_signed(
                        cls, psi_signed_v1(dph, dpv, BETA_TRIAL4))
                    fields[VARIANTS[3]] = score_aug_signed(
                        cls, psi_signed_v1(dph, dpv, BETA_TRIAL85))
                    fields[VARIANTS[4]] = score_aug_legacy(
                        cls, psi_abs_v1(dph, dpv, BETA_TRIAL4))
                    fields[VARIANTS[5]] = score_aug_legacy(
                        cls, psi_abs_v1(dph, dpv, BETA_TRIAL85))
                    psi_v2 = signed_combine(
                        compute_psi_v2(phi_prev["phi_horizontal"],
                                       phi["phi_horizontal"]),
                        compute_psi_v2(phi_prev["phi_vertical"],
                                       phi["phi_vertical"]))
                    fields[VARIANTS[6]] = score_aug_signed(cls, psi_v2)
                phi_prev = phi

                scores = {(v, a): block_agg(fields[v], dim, a).ravel()
                          for v in VARIANTS for a in AGGS}
                cfg_snaps.append(dict(
                    y=(l2_all[si] >= l2_thr).ravel().astype(int),
                    scores=scores,
                ))
            out[sc][re] = cfg_snaps
    return out


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 5: fixed phase-11E rerun (signed psi, "
                    "trial-4 params, dual aggregation, traj bootstrap)")
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N
    from h2b_ceiling_random_split import best_threshold_f1

    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 5: phase-11E rerun, fixed (signed psi; trial-4 vs "
          "trial-85; dual aggregation)")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"seed={args.seed}  n-boot={args.n_boot}")
    print("=" * 88)
    print("\n  [param discrepancy log] trial #4 (config.TRAINED_*): "
          f"beta={BETA_TRIAL4}, thr={THR_TRIAL4}")
    print("                          trial #85 (phase11e hardcoded):  "
          f"beta={BETA_TRIAL85:.4f}, thr={THR_TRIAL85:.4f}")
    print()

    by_scene = {}
    for sc in args.scenario:
        rows = []
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                              f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                rows.append((re, dp, pp))
        if rows:
            by_scene[sc] = rows
    if len(by_scene) < 2:
        print("need >=2 scenarios with data."); return

    print("  building per-snapshot score variants...")
    t0 = time.time()
    data = _gather(by_scene, args.dim, args.max_snaps)
    print(f"  done in {time.time() - t0:.1f}s\n")

    scenarios = list(data.keys())

    def pool(scs, key):
        """Concatene y ou un score (variant, agg) sur des scenarios."""
        chunks = []
        for sc in scs:
            for re in sorted(data[sc]):
                for d in data[sc][re]:
                    chunks.append(d["y"] if key == "y" else d["scores"][key])
        return np.concatenate(chunks)

    # ---- LOSO : F1 par fold + F1 par trajectoire (scenario, Re) ----
    f1_fold = {}    # (variant, agg) -> {held: f1}
    f1_traj = {}    # (variant, agg) -> {(held, re): f1}
    for held in scenarios:
        others = [s for s in scenarios if s != held]
        Ytr = pool(others, "y")
        Yva = pool([held], "y")
        for v in VARIANTS:
            for a in AGGS:
                Str = pool(others, (v, a))
                Sva = pool([held], (v, a))
                if v == "v2-classical":
                    thr, _ = best_threshold_f1(Str, Ytr)
                else:
                    thr, _ = best_threshold_f1(
                        Str, Ytr,
                        grid=np.linspace(Str.min(), Str.max(), 201))
                f1_fold.setdefault((v, a), {})[held] = float(f1_score(
                    Yva, (Sva > thr).astype(int), zero_division=0))
                for re in sorted(data[held]):
                    y_t = np.concatenate(
                        [d["y"] for d in data[held][re]])
                    s_t = np.concatenate(
                        [d["scores"][(v, a)] for d in data[held][re]])
                    f1_traj.setdefault((v, a), {})[(held, re)] = float(
                        f1_score(y_t, (s_t > thr).astype(int),
                                 zero_division=0))

    # ---- tables par agregation ----
    for a in AGGS:
        print(f"\n  [aggregation = block_{a}]"
              f"  (per-fold prevalence 0.250, refine-all floor F1=0.400)")
        head = (f"  {'variant':<42} "
                + " ".join(f"{sc[:8]:>8}" for sc in scenarios)
                + f"  {'mean':>7}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for v in VARIANTS:
            pf = f1_fold[(v, a)]
            cells = " ".join(f"{pf[sc]:>8.3f}" for sc in scenarios)
            print(f"  {v:<42} {cells}  "
                  f"{np.mean(list(pf.values())):>7.3f}")

    # ---- resume 2x2x2 (params x psi-handling x aggregation) ----
    print("\n  [2x2x2 summary: mean LOSO F1]")
    print(f"  {'psi-handling':<12} {'params':<10} {'block_avg':>10} "
          f"{'block_max':>10}")
    cell = {("signed", "trial4"): VARIANTS[2],
            ("signed", "trial85"): VARIANTS[3],
            ("legacy-abs", "trial4"): VARIANTS[4],
            ("legacy-abs", "trial85"): VARIANTS[5]}
    for (h, prm), v in cell.items():
        m = {a: np.mean(list(f1_fold[(v, a)].values())) for a in AGGS}
        print(f"  {h:<12} {prm:<10} {m['avg']:>10.3f} {m['max']:>10.3f}")
    m = {a: np.mean(list(f1_fold[(VARIANTS[6], a)].values()))
         for a in AGGS}
    print(f"  {'signed v2':<12} {'param-free':<10} {m['avg']:>10.3f} "
          f"{m['max']:>10.3f}")

    # ---- bootstrap trajectoire (Task 3) : delta vs v1-classique ----
    traj_keys = [(sc, re) for sc in scenarios for re in sorted(data[sc])]
    traj_ids = np.arange(len(traj_keys))
    print(f"\n  [trajectory bootstrap, n_traj={len(traj_keys)}, "
          f"B={args.n_boot}]  delta = F1(variant) - F1(v1-classical), "
          "paired per trajectory")
    print(f"  {'variant':<42} {'agg':<4} {'mean_d':>8} "
          f"{'CI95':>18} {'frac>0':>7}")
    boot_rows = []
    for v in VARIANTS[2:]:
        for a in AGGS:
            va = np.array([f1_traj[(v, a)][k] for k in traj_keys])
            vb = np.array([f1_traj[("v1-classical (no psi)", a)][k]
                           for k in traj_keys])
            r = paired_delta_bootstrap(va, vb, traj_ids, B=args.n_boot,
                                       seed=args.seed)
            boot_rows.append((v, a, r))
            print(f"  {v:<42} {a:<4} {r['mean_delta']:>+8.3f} "
                  f"[{r['ci_low']:>+7.3f},{r['ci_high']:>+7.3f}] "
                  f"{r['frac_positive']:>7.2f}")
    # reference : v1-classique vs v2-classique (effet Lohner)
    for a in AGGS:
        va = np.array([f1_traj[("v1-classical (no psi)", a)][k]
                       for k in traj_keys])
        vb = np.array([f1_traj[("v2-classical", a)][k]
                       for k in traj_keys])
        r = paired_delta_bootstrap(va, vb, traj_ids, B=args.n_boot,
                                   seed=args.seed)
        print(f"  {'(ref) v1-classical - v2-classical':<42} {a:<4} "
              f"{r['mean_delta']:>+8.3f} "
              f"[{r['ci_low']:>+7.3f},{r['ci_high']:>+7.3f}] "
              f"{r['frac_positive']:>7.2f}")

    # ---- sauvegarde ----
    out = os.path.join(RESULTS_DIR,
                       f"t5_v1_psi_loso_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenarios=np.array(scenarios),
        variants=np.array(VARIANTS),
        aggs=np.array(AGGS),
        f1_fold=np.array([[[f1_fold[(v, a)][sc] for sc in scenarios]
                           for a in AGGS] for v in VARIANTS]),
        traj_keys=np.array([f"{sc}|Re{re}" for sc, re in traj_keys]),
        f1_traj=np.array([[[f1_traj[(v, a)][k] for k in traj_keys]
                           for a in AGGS] for v in VARIANTS]),
        boot_variant=np.array([f"{v}|{a}" for v, a, _ in boot_rows]),
        boot_mean=np.array([r["mean_delta"] for _, _, r in boot_rows]),
        boot_ci_low=np.array([r["ci_low"] for _, _, r in boot_rows]),
        boot_ci_high=np.array([r["ci_high"] for _, _, r in boot_rows]),
        boot_frac_pos=np.array([r["frac_positive"]
                                for _, _, r in boot_rows]),
        beta_trial4=BETA_TRIAL4, thr_trial4=THR_TRIAL4,
        beta_trial85=BETA_TRIAL85, thr_trial85=THR_TRIAL85,
        seed=args.seed,
        git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV3 Task 5 complete.")


if __name__ == "__main__":
    main()
