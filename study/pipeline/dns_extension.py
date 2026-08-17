#!/usr/bin/env python3
"""
V3 Task 8 - Extension des donnees (protocole v3, section 1.1).

Enveloppe (wrapper) de `study/dns_sweep.py` — JAMAIS modifie en
place. Ajoute :

  (1) les 4 scenarios supplementaires, deja implementes dans le module
      scenario V1 (src/Simulation/solver.py) : lamb_oseen,
      island_coalescence, double_tearing, magnetic_twist ;
  (2) `--phys-seed k ...` : perturbation des conditions initiales —
      bruit gaussien additif sur (vx, vy), amplitude par defaut 0.1
      (= `noise_amplitude` de `init_kelvin_helmholtz`, le "niveau de
      perturbation V1" ; --noise-amplitude pour ajuster), puis
      projection div-free. seed=0 = AUCUNE perturbation : reproduit
      exactement le dataset mono-graine existant (garde de
      cross-check, section 1.1) ;
  (3) regeneration des labels phase 2 (reutilise
      `phase2_hard_patches.analyze_dns_file`) avec nommage conscient de
      la graine ;
  (4) validation phase 1b sur CHAQUE nouvelle trajectoire (reutilise
      `phase1b_dns_validation.analyse_one` + les checks publies) :
      div B et non-divergence pour toutes ; E(t) non croissante pour
      toutes (systemes dissipatifs non forces) ; checks specifiques
      reutilises : OT (fenetre de decroissance), KH (croissance),
      tearing (pic de <J^2>) — ce dernier applique aussi a
      double_tearing et island_coalescence (meme physique).

PARAMETRES NON PRE-ENREGISTRES (a confirmer avant tout lancement
complet ; ajustables en CLI, journalises dans le .npz) :
  - durees des 4 nouveaux scenarios (V3_SCENARIO_CONFIG ci-dessous) ;
  - amplitude de la perturbation (0.1, convention V1 KH).

Nommage : seed 0 -> noms standards (dns_{sc}_Re{re}_N{N}.npz),
seed k>0 -> suffixe _seed{k}. Les fichiers existants sont SAUTES par
defaut (--no-skip-existing pour forcer).

Usage :
  python study/v3/dns_extension.py --dry-run           # plan + cout
  python study/v3/dns_extension.py                     # tout (heures)
  python study/v3/dns_extension.py --scenario lamb_oseen --re 400 \
      --phys-seed 0 1                                     # sous-ensemble
"""
import argparse, json, os, sys, time
import numpy as np

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
from Simulation.grid import AXIS_X, AXIS_Y

# scenarios V3 = les 4 V2 + les 4 nouveaux (section 1.1)
EXTRA_SCENARIOS = ["lamb_oseen", "island_coalescence",
                   "double_tearing", "magnetic_twist"]

# durees proposees (NON pre-enregistrees ; memes conventions que
# config.SCENARIO_CONFIG : snapshot_dt=0.10, t_max assez long pour le
# developpement de l'instabilite — tearing-like alignes sur
# harris_tearing, vortex/twist sur orszag_tang)
V3_SCENARIO_CONFIG = {
    "lamb_oseen":         {"warmup_steps": 100, "t_max": 3.0,
                           "snapshot_dt": 0.10},
    "island_coalescence": {"warmup_steps": 80, "t_max": 2.0,
                           "snapshot_dt": 0.10},
    "double_tearing":     {"warmup_steps": 80, "t_max": 2.0,
                           "snapshot_dt": 0.10},
    "magnetic_twist":     {"warmup_steps": 80, "t_max": 2.0,
                           "snapshot_dt": 0.10},
}

V1_NOISE_AMPLITUDE = 0.1   # init_kelvin_helmholtz(noise_amplitude=0.1)

# checks 1b specifiques reutilises par scenario (les autres : generiques)
TEARING_LIKE = {"harris_tearing", "double_tearing", "island_coalescence"}


def fluctuating_ke_fixed(vx, vy):
    """Energie cinetique de perturbation, observable CORRIGEE :
    soustraction de la moyenne sur X (axe 0), direction d'invariance du
    flot de base KH v_flow(Y). La version de phase 1b moyenne sur
    l'axe 1 (Y) et laisse le profil de base (variance ~0.34) dans Ep,
    masquant toute croissance — y compris a seed 0 (deviation D2 ;
    phase 1b reste intouchee, reparation cote v3 par copie)."""
    return 0.5 * ((vx - vx.mean(axis=0, keepdims=True)) ** 2
                  + (vy - vy.mean(axis=0, keepdims=True)) ** 2).mean()


def mean_sq_current_fixed(Bx, By):
    """<J_z^2> avec J_z = dBy/dx - dBx/dy, observable CORRIGEE.

    La version de phase 1b (`dns_validation.mean_sq_current`) prend ses deux
    differences sur l'axe oppose a la convention du depot
    (`grid.AXIS_X = 0`, `AXIS_Y = 1`). Elle calcule donc
    dBy/dy - dBx/dx : une combinaison de DEFORMATION, pas un courant.

    Ce n'est pas un courant de signe oppose — le carre n'aurait rien
    rattrape. C'est son complementaire : sur une rotation solide et sur un
    cisaillement pur elle rend exactement zero, et sur une compression pure
    elle rend ce que le courant ne voit pas (deviation D3 ; phase 1b reste
    intouchee, reparation cote v3 par copie, comme pour D2).
    """
    dxBy = (np.roll(By, -1, axis=AXIS_X) - np.roll(By, 1, axis=AXIS_X)) * 0.5
    dyBx = (np.roll(Bx, -1, axis=AXIS_Y) - np.roll(Bx, 1, axis=AXIS_Y)) * 0.5
    return ((dxBy - dyBx) ** 2).mean()


def check_kh_fixed(dns_path):
    """check_kh de phase 1b (memes fenetres t in [0,0.2] vs [0.8,1.2],
    meme critere croissance > 1.1x) avec l'observable corrigee."""
    d = np.load(dns_path)
    t = d["t"].astype(np.float64)
    vx = d["vx"].astype(np.float64)
    vy = d["vy"].astype(np.float64)
    Ep = np.array([fluctuating_ke_fixed(vx[i], vy[i])
                   for i in range(len(t))])
    m0 = (t >= 0.0) & (t <= 0.2)
    m1 = (t >= 0.8) & (t <= 1.2)
    if not m0.any() or not m1.any():
        return dict(ok=False, reason="insufficient time coverage")
    e0 = float(Ep[m0].mean())
    e1 = float(Ep[m1].mean())
    return dict(Ep_early=e0, Ep_mid=e1,
                growth=float(e1 / max(e0, 1e-30)),
                ok=bool(e1 > 1.1 * e0))


# -------------------------------------------------------------------
# Helpers purs
# -------------------------------------------------------------------

def seeded_dns_path(results_dir, sc, re, N, seed):
    suf = "" if seed == 0 else f"_seed{seed}"
    return os.path.join(results_dir, f"dns_{sc}_Re{re}_N{N}{suf}.npz")


def seeded_patches_path(results_dir, sc, re, N, dim, seed):
    suf = "" if seed == 0 else f"_seed{seed}"
    return os.path.join(results_dir,
                        f"patches_{sc}_Re{re}_N{N}_dim{dim}{suf}.npz")


PERT_K_CUT = 8   # modes entiers conserves (grande echelle, comme la
                 # perturbation sin structuree de init_kelvin_helmholtz)


def _band_limited_noise(rng, N, k_cut):
    """Bruit gaussien filtre passe-bas spectral (modes |k| <= k_cut),
    normalise a ecart-type 1. Bande limitee pour deux raisons : (i) la
    perturbation V1 (KH) est grande echelle ; (ii) le projecteur
    div-free du solveur est exact hors mode de Nyquist — du bruit blanc
    pleine bande laisserait un residu de divergence O(1)."""
    eta = np.fft.fft2(rng.standard_normal((N, N)))
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    eta[np.sqrt(KX ** 2 + KY ** 2) > k_cut] = 0.0
    out = np.real(np.fft.ifft2(eta))
    return out / (out.std() + 1e-30)


def perturb_fields(vx, vy, seed, amplitude, k_cut=PERT_K_CUT):
    """Bruit additif seede, bande limitee, sur (vx, vy).
    seed=0 -> identite (le dataset mono-graine existant reste le
    cross-check, section 1.1)."""
    if seed == 0:
        return vx, vy
    rng = np.random.default_rng(seed)
    N = vx.shape[0]
    return (vx + amplitude * _band_limited_noise(rng, N, k_cut),
            vy + amplitude * _band_limited_noise(rng, N, k_cut))


def energy_non_increasing(E, tol=1e-3):
    """Invariant generique des systemes dissipatifs non forces
    (meme tolerance que le check OT de phase 1b)."""
    dE = np.diff(np.asarray(E, float))
    return bool((dE <= tol).all()), (float(dE.max()) if len(dE) else 0.0)


def presence_matrix(results_dir, scenarios, re_values, N, seeds):
    """{(sc, seed): nb de fichiers dns presents sur les Re demandes}."""
    out = {}
    for sc in scenarios:
        for seed in seeds:
            out[(sc, seed)] = sum(
                os.path.exists(seeded_dns_path(results_dir, sc, re,
                                               N, seed))
                for re in re_values)
    return out


# -------------------------------------------------------------------
# Wrapper phase 1 (monkeypatch runtime ; phase1 jamais edite)
# -------------------------------------------------------------------

def _extended_init(sim, scenario, seed, amplitude):
    """Carte d'init V1 etendue + perturbation phys-seed + reprojection."""
    inits = {
        "orszag_tang": sim.init_orszag_tang,
        "harris_tearing": sim.init_harris_tearing,
        "kelvin_helmholtz": sim.init_kelvin_helmholtz,
        "mhd_rotor": sim.init_mhd_rotor,
        "lamb_oseen": sim.init_lamb_oseen_vortex,
        "island_coalescence": sim.init_island_coalescence,
        "double_tearing": sim.init_double_tearing,
        "magnetic_twist": sim.init_magnetic_twist,
    }
    if scenario not in inits:
        raise ValueError(f"Unknown scenario: {scenario}")
    inits[scenario]()
    if seed != 0:
        sim.vx, sim.vy = perturb_fields(sim.vx, sim.vy, seed, amplitude)
        sim.enforce_incompressibility()   # restaure div v = 0


def run_one(scenario, re, N, seed, amplitude):
    """Reutilise phase1.run_dns via monkeypatch de init_scenario et
    extension runtime de SCENARIO_CONFIG (config jamais edite sur
    disque)."""
    import dns_sweep as p1
    p1.SCENARIO_CONFIG.update(V3_SCENARIO_CONFIG)   # runtime only
    p1.init_scenario = (lambda sim, sc:
                        _extended_init(sim, sc, seed, amplitude))
    snapshots, metadata = p1.run_dns(scenario, re, N=N)
    metadata["phys_seed"] = seed
    metadata["noise_amplitude"] = amplitude
    return snapshots, metadata


def save_seeded(snapshots, metadata, results_dir, cli_args):
    """Empaquetage identique a phase1.save_dns + nommage seede +
    hash git + arguments CLI (garde-fous v3)."""
    sc, Re, N = metadata["scenario"], metadata["Re"], metadata["N"]
    seed = metadata["phys_seed"]
    path = seeded_dns_path(results_dir, sc, Re, N, seed)

    n = len(snapshots)
    vx = np.zeros((n, N, N), dtype=np.float32)
    vy = np.zeros_like(vx); Bx = np.zeros_like(vx); By = np.zeros_like(vx)
    t = np.zeros(n); step = np.zeros(n, dtype=np.int32)
    for i, s in enumerate(snapshots):
        vx[i] = s["vx"].astype(np.float32)
        vy[i] = s["vy"].astype(np.float32)
        Bx[i] = s["Bx"].astype(np.float32)
        By[i] = s["By"].astype(np.float32)
        t[i] = s["t"]; step[i] = s["step"]

    np.savez_compressed(
        path, vx=vx, vy=vy, Bx=Bx, By=By, t=t, step=step,
        **{f"meta_{k}": v for k, v in metadata.items()
           if isinstance(v, (int, float, str, bool))},
        git_hash=git_commit_hash(), cli_args=json.dumps(cli_args),
    )
    print(f"  saved: {os.path.basename(path)} "
          f"({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def make_labels(dns_path, dims, results_dir, seed):
    """Labels phase 2 (reutilise analyze_dns_file), nommage seede."""
    from hard_patch_labels import analyze_dns_file
    results_by_dim, meta = analyze_dns_file(dns_path, dims)
    for n_p, res in results_by_dim.items():
        out = seeded_patches_path(results_dir, meta["scenario"],
                                  meta["Re"], meta["N"], n_p, seed)
        np.savez_compressed(
            out,
            l2_errors=res["l2_errors"],
            classical_scores=res["classical_scores"],
            is_hard=res["is_hard"],
            l2_threshold=res["l2_threshold"],
            t=meta["t"], scenario=meta["scenario"], Re=meta["Re"],
            N=meta["N"], n_patches=n_p, phys_seed=seed,
        )
        print(f"    labels: {os.path.basename(out)}")


def div_rel_max_fixed(dns_path):
    """max|div B| / rms|B| avec l'operateur qui GARANTIT la contrainte.

    D-73, meme famille que D-72. `analyse_one` (fichier GELE) calcule cette
    grandeur au SPECTRAL, et son commentaire porte la condition qui la
    justifiait : « should be O(eps_machine) WHEN THE FFT PROJECTION IS
    APPLIED ». Depuis D-25 elle ne l'est plus pour B : `PROJECT_B = False`,
    et B est solenoidal AUX DIFFERENCES FINIES par construction (induction
    en forme rotationnelle, cf. `solver.enforce_incompressibility`).

    Le portail rejetait donc des trajectoires saines. Mesure de bout en bout,
    DNS reellement generee a HEAD (harris_tearing, Re=400, N=64, seed=0) :

      div_rel_max spectral (avant)   1.6205e-02  -> FAIL contre div_tol=1e-3
      div_rel_max assorti  (apres)   5.0573e-06  -> OK

    Les artefacts DNS deja dans le depot passent dans les deux cas (mesure
    sur 8 : 1.4170e-05 a 5.6503e-05) : ils datent d'avant D-25, et ce qu'on y
    lit est le plancher de stockage float32, pas l'ecart d'operateur.

    DEUX ECARTS AU FICHIER GELE, tous deux volontaires ici :

    1. l'operateur — FD4 de V1 (`_fd_grad`, celui-la meme qui assemble
       `rhs_B`), au lieu du spectral ;
    2. le pas d'espace — `analyse_one` passe `dx = 1/N` alors que
       `PeriodicGrid` pose `L = 2*pi`, donc `dx = 2*pi/N`. Un facteur 6.2832
       sur toute divergence rapportee. Le gele n'est pas touche ; cette
       version emploie le `dx` de la grille.

    Le fichier gele reste intact : `dns_extension` est deja l'endroit ou
    vivent les observables corrigees (`mean_sq_current_fixed`,
    `fluctuating_ke_fixed`, `check_kh_fixed`), celle-ci les rejoint.
    """
    from Simulation.solver import MHDSolver

    d = np.load(dns_path)
    Bx = d["Bx"].astype(np.float64)
    By = d["By"].astype(np.float64)
    N = Bx.shape[1]
    dx = 2 * np.pi / N                      # celui de PeriodicGrid, cf. (2)
    worst = 0.0
    for i in range(Bx.shape[0]):
        g_Bx_x, _ = MHDSolver._fd_grad(Bx[i], dx)
        _, g_By_y = MHDSolver._fd_grad(By[i], dx)
        rms_B = np.sqrt((Bx[i] ** 2 + By[i] ** 2).mean()) + 1e-30
        worst = max(worst, float(np.abs(g_Bx_x + g_By_y).max() / rms_B))
    return worst


def validate_one(dns_path, scenario, div_tol=1e-3):
    """Checks phase 1b sur UNE trajectoire (fonctions 1b reutilisees).
    Retourne (liste d'echecs, lignes de log)."""
    from dns_validation import (
        analyse_one, check_ot, check_tearing)
    res = analyse_one(dns_path)
    name = os.path.basename(dns_path)
    fails, log = [], []

    if res["diverged"]:
        fails.append(f"{name}: solver diverged")
    # D-73 : la decision porte sur l'operateur assorti. La valeur spectrale
    # du fichier gele reste journalisee a cote, pour que l'ecart entre les
    # deux reste visible plutot que d'etre silencieusement remplace.
    div_rel = div_rel_max_fixed(dns_path)
    if div_rel > div_tol:
        fails.append(f"{name}: divB {div_rel:.1e}")
    log.append(f"div={div_rel:.1e} (spectral gele "
               f"{res['div_rel_max']:.1e})")

    mono_ok, max_dE = energy_non_increasing(res["E"])
    log.append(f"E mono={'OK' if mono_ok else f'WARN({max_dE:+.1e})'}")
    if not mono_ok:
        fails.append(f"{name}: E increases (max dE={max_dE:+.2e})")

    if scenario == "orszag_tang":
        chk = check_ot(res)
        log.append(f"OT decay={chk['frac_decay'] * 100:.1f}%")
        if not (chk["E0_ok"] and chk["frac_ok"]):
            fails.append(f"{name}: OT check (E0={chk['E0']:.3f}, "
                         f"decay={chk['frac_decay'] * 100:.1f}%)")
    elif scenario in TEARING_LIKE:
        chk = check_tearing(res)
        log.append(f"J2 amp={chk['amplification']:.2f}x "
                   f"@t={chk['t_peak']:.2f}")
        if not chk["ok"]:
            fails.append(f"{name}: tearing amp "
                         f"{chk['amplification']:.2f}x")
    elif scenario == "kelvin_helmholtz":
        chk = check_kh_fixed(dns_path)
        if "growth" in chk:
            log.append(f"KH growth(fixed obs)={chk['growth']:.2f}x")
        if not chk["ok"]:
            fails.append(f"{name}: KH check (fixed obs, growth="
                         f"{chk.get('growth', float('nan')):.2f}x)")
    return fails, log


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def _summary(args, results_dir, all_scenarios, all_fails):
    """Matrice de presence + bilan de validation + verdict acceptation."""
    print("\n" + "=" * 88)
    print("  presence matrix (files per scenario x seed, "
          f"over {len(args.re)} Re values):")
    pm = presence_matrix(results_dir, all_scenarios, args.re, args.N,
                         args.phys_seed)
    print(f"  {'scenario':<20} "
          + " ".join(f"{'seed' + str(s):>7}" for s in args.phys_seed))
    complete = True
    for sc in all_scenarios:
        cells = []
        for s in args.phys_seed:
            n = pm[(sc, s)]
            cells.append(f"{n:>7d}")
            if n < len(args.re):
                complete = False
        print(f"  {sc:<20} " + " ".join(cells))

    print()
    if all_fails:
        print(f"  VALIDATION: {len(all_fails)} FAILURE(S)")
        for f in all_fails:
            print(f"    - {f}")
    else:
        print("  VALIDATION: clean on all validated trajectories")
    print(f"  ACCEPTANCE ({len(all_scenarios)} scenarios x "
          f">= 2 seeds, clean log): "
          f"{'PASS' if complete and not all_fails else 'NOT YET'}")
    print("\nV3 Task 8 complete.")


def main():
    p = argparse.ArgumentParser(
        description="V3 Task 8: DNS extension (8 scenarios x phys seeds)")
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, VQA_DIMS

    all_scenarios = SCENARIOS + EXTRA_SCENARIOS
    p.add_argument("--scenario", nargs="+", default=all_scenarios)
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--phys-seed", nargs="+", type=int, default=[0, 1],
                   help="graines physiques (0 = IC non perturbees)")
    p.add_argument("--noise-amplitude", type=float,
                   default=V1_NOISE_AMPLITUDE,
                   help="amplitude de la perturbation IC (defaut 0.1 = "
                        "noise_amplitude V1 KH)")
    p.add_argument("--labels-dim", nargs="+", type=int, default=[4],
                   help=f"dims des labels phase 2 (V2 utilisait "
                        f"{VQA_DIMS})")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="re-genere meme si le fichier dns existe")
    p.add_argument("--dry-run", action="store_true",
                   help="affiche le plan et sort sans calculer")
    p.add_argument("--validate-only", action="store_true",
                   help="aucune generation : re-valide les fichiers "
                        "presents de la grille demandee (checks 1b + "
                        "observable KH corrigee)")
    p.add_argument("--seed", type=int, default=0,
                   help="enregistre (le pipeline DNS est deterministe ; "
                        "la graine physique est --phys-seed)")
    args = p.parse_args()

    print("=" * 88)
    print("  V3 Task 8: DNS extension — 8 scenarios x physics seeds")
    print(f"  N={args.N}  Re={args.re}  phys-seeds={args.phys_seed}  "
          f"noise={args.noise_amplitude}")
    print("  NOTE: run lengths of the 4 new scenarios and the noise "
          "amplitude are NOT pre-registered;")
    print("        defaults are logged in this output and in every .npz "
          "(adjust via CLI if needed).")
    for sc, cfg in V3_SCENARIO_CONFIG.items():
        print(f"        {sc:<20} t_max={cfg['t_max']}  "
              f"snapshot_dt={cfg['snapshot_dt']}")
    print("=" * 88)
    print()

    # ---- plan ----
    todo, skipped = [], []
    for sc in args.scenario:
        for re in args.re:
            for seed in args.phys_seed:
                path = seeded_dns_path(RESULTS_DIR, sc, re, args.N, seed)
                if os.path.exists(path) and not args.no_skip_existing:
                    skipped.append((sc, re, seed))
                else:
                    todo.append((sc, re, seed))
    print(f"  plan: {len(todo)} DNS runs to compute, "
          f"{len(skipped)} existing (skipped)")
    if args.dry_run:
        for sc, re, seed in todo:
            print(f"    - {sc} Re={re} seed={seed}")
        print("\n  dry-run: nothing computed.")
        return

    # ---- mode re-validation seule ----
    all_fails = []
    if args.validate_only:
        print("  [validate-only] re-validating existing files "
              "(corrected KH observable)")
        for sc in args.scenario:
            for re in args.re:
                for seed in args.phys_seed:
                    path = seeded_dns_path(RESULTS_DIR, sc, re,
                                           args.N, seed)
                    if not os.path.exists(path):
                        print(f"  MISSING {os.path.basename(path)}")
                        continue
                    fails, log = validate_one(path, sc)
                    tag = "OK" if not fails else "FAIL"
                    print(f"  [{tag:>4}] {os.path.basename(path):<46} "
                          + "  ".join(log))
                    all_fails += fails
        _summary(args, RESULTS_DIR, all_scenarios, all_fails)
        return

    # ---- runs + labels + validation ----
    new_paths = []
    cli = vars(args)
    for i, (sc, re, seed) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {sc} Re={re} seed={seed}")
        t0 = time.time()
        snaps, meta = run_one(sc, re, args.N, seed,
                              args.noise_amplitude)
        path = save_seeded(snaps, meta, RESULTS_DIR, cli)
        new_paths.append(path)
        make_labels(path, args.labels_dim, RESULTS_DIR, seed)
        fails, log = validate_one(path, sc)
        tag = "OK" if not fails else "FAIL"
        print(f"  validation [{tag}]: " + "  ".join(log))
        all_fails += fails
        print(f"  total {time.time() - t0:.0f}s")

    _summary(args, RESULTS_DIR, all_scenarios, all_fails)


if __name__ == "__main__":
    main()
