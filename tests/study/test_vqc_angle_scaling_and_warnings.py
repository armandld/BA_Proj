"""File ouverte de `COUVERTURE.md`, item 6/7
(`h2b_variational_classifier.py --reps-ansatz`) : les deux observations que
la passe precedente avait laissees NON MESUREES ("a instruire, pas a
croire") sont mesurees ici -- et une troisieme, plus lourde, decouverte en
mesurant la seconde.

(a) `run_vqc` enrobe `vqc.fit()` d'un `warnings.catch_warnings()` +
`simplefilter("ignore")` INCONDITIONNEL -- meme forme que D-38. Mesure sur
donnees reelles (`orszag_tang`/`harris_tearing`/`kelvin_helmholtz`/
`mhd_rotor`, Re 800-1600, N=96, dim=4, `--reps-fm 2 --reps-ansatz 2
--maxiter 40`), en appelant `run_vqc` REEL avec son `catch_warnings`
remplace par un enregistreur (pas une reimplementation) : **0 warning**
emis par `vqc.fit()` dans cet environnement (qiskit 2.5.2,
qiskit-machine-learning 0.9.1). La suppression est INERTE aujourd'hui --
verifie, pas suppose. Reste un motif a risque (large `ignore` plutot
qu'une categorie precise) si la version de qiskit change ; rien a corriger
maintenant faute de defaut demontre.

(b) Le commentaire dit *"Scale to [-pi, pi]"* ; la formule
(`/span * pi - pi/2`) donne en realite **[-pi/2, +pi/2]** pour
l'ENTRAINEMENT (verifie ci-dessous, deterministe -- ne depend d'aucun
tirage). La VALIDATION, avec les memes `lo`/`hi`/`span` tires du train,
n'est PAS bornee a [-pi/2, pi/2] -- elle peut deborder (mesure sur un
champ construit pour separer : 40 % d'etalement en plus sur la validation
suffit) -- et n'est clippee qu'a **+-pi**, deux fois plus large que la
plage d'entrainement reelle. Composante FORMULE : reelle, deterministe,
epinglee ci-dessous.

**(c) -- la decouverte principale de cet item, question 3 de VIGIL.md.**
En essayant de chiffrer la CONSEQUENCE de (b) sur le F1 (memes donnees
d'entrainement, meme `seed=0` -- seul le clip de validation cense
changer), deux mesures independantes ne s'accordaient pas : la premiere
donnait +0,050 (clip pi favorise), la seconde -0,006 (clip pi/2 favorise,
signe INVERSE) -- VIGIL.md : *"si les deux references different de plus
que l'effet cherche, la grandeur ne tranche rien, il faut le dire"*.
Cause trouvee en appelant `run_vqc` DEUX FOIS de suite avec des arguments
RIGOUREUSEMENT IDENTIQUES, `seed=0` compris : **F1 0,653 puis 0,639,
`p_va` different jusqu'a 0,487 sur un point** (la moitie de l'intervalle
[0,1]), alors que `run_qke`, la fonction soeur du meme fichier, USE bien
son `seed` (`SVC(random_state=seed)`, verifie par lecture). `run_vqc`
declare un parametre `seed` dans sa signature et ne le lit JAMAIS dans son
corps (verifie par `grep` sur le corps de la fonction) -- meme forme que
D-48 (`classical_warm_start_params`, arguments morts). C'est pour cette
raison, PAS pour le clip lui-meme, que l'observation (b) ne peut pas
recevoir de verdict numerique fiable : le bruit d'un VQC non graine
depasse largement l'effet du clip qu'on cherchait a mesurer.

Aucun nombre publie n'en depend : `results/` ne contient aucun `vqc_*.npz`
dans ce depot -- D-81 (`docs/RESULTS.md`) l'a deja etabli, la phase 12
n'a jamais tourne ici. Rapport seul dans `docs/COUVERTURE.md`, pas
`DEFAUTS.md` (regle d'arret : ni lecture publiee, ni blocage de la
reoptimisation) -- mais c'est le genre de defaut qu'une premiere execution
de phase 12 sous la campagne ferait decouvrir a la dure (deux runs
"identiques" qui ne se reproduisent pas).

Ce fichier epingle la FORMULE de (a)/(b) (rapide, deterministe, sans VQC)
et, marque `slow`, la non-reproductibilite (c) mesuree ci-dessus.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _scale(Xtr, Xva, clip_bound):
    """Mirroir exact des 3 lignes de mise a l'echelle de `run_vqc` /
    `run_qke` (`h2b_variational_classifier.py`), seul le clip est
    parametre pour comparer les deux lectures."""
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    Xtr_s = (Xtr - lo) / span * np.pi - np.pi / 2
    Xva_s = (Xva - lo) / span * np.pi - np.pi / 2
    Xva_s = np.clip(Xva_s, -clip_bound, clip_bound)
    return Xtr_s, Xva_s


def test_training_range_is_half_pi_not_pi_as_the_comment_says():
    """Le commentaire du module dit "Scale to [-pi, pi]". La formule
    rend en realite [-pi/2, +pi/2] pour les points d'ENTRAINEMENT --
    par construction (lo/hi sont le min/max de CE MEME tableau)."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(200, 4)) * 3.7 + 1.2  # echelle/offset arbitraires
    Xva = rng.normal(size=(80, 4)) * 3.7 + 1.2
    Xtr_s, _ = _scale(Xtr, Xva, clip_bound=np.pi)

    assert np.isclose(Xtr_s.min(), -np.pi / 2, atol=1e-9)
    assert np.isclose(Xtr_s.max(), np.pi / 2, atol=1e-9)
    # et PAS +-pi, contrairement au commentaire :
    assert not np.isclose(Xtr_s.max(), np.pi, atol=0.1)


def test_validation_can_exceed_training_range_and_clip_is_twice_as_wide():
    """La validation, mise a l'echelle avec le lo/hi du TRAIN, peut sortir
    de [-pi/2, pi/2] des que sa distribution differe un peu -- et le clip
    du code (+-pi) est deux fois plus large que cette plage d'entrainement,
    donc laisse passer un debordement au lieu de le ramener dedans."""
    rng = np.random.default_rng(1)
    # Val legerement plus etale que train sur un axe, pour separer les deux
    # lectures (VIGIL.md : choisir un champ qui separe).
    Xtr = rng.normal(size=(200, 4))
    Xva = rng.normal(size=(80, 4)) * 1.4  # 40% plus etale

    Xtr_s, Xva_s_wide = _scale(Xtr, Xva, clip_bound=np.pi)
    _, Xva_s_narrow = _scale(Xtr, Xva, clip_bound=np.pi / 2)

    train_half_range = np.pi / 2
    n_exceeds_train_range = int(
        np.sum(np.abs(Xva_s_wide) > train_half_range + 1e-9))
    assert n_exceeds_train_range > 0, (
        "precondition : ce tirage doit produire au moins une entree de "
        "validation hors de la plage d'entrainement pour separer les deux "
        "lectures -- sinon changer la graine/l'etalement")

    # Avec le clip actuel (+-pi), ces entrees survivent SANS etre ramenees
    # dans la plage vue a l'entrainement ; avec le clip +-pi/2 (ce que le
    # commentaire annonce), elles y seraient forcees.
    still_out_with_pi_clip = np.sum(
        np.abs(Xva_s_wide) > train_half_range + 1e-9)
    still_out_with_half_pi_clip = np.sum(
        np.abs(Xva_s_narrow) > train_half_range + 1e-9)
    assert still_out_with_pi_clip == n_exceeds_train_range
    assert still_out_with_half_pi_clip == 0
    assert not np.allclose(Xva_s_wide, Xva_s_narrow)


def _real_dataset_for_vqc(n_train=200, n_val=100):
    results_dir = os.path.join(_REPO_ROOT, "results")
    from config import SCENARIOS
    from h2b_ceiling_random_split import build_dataset
    from h2b_variational_classifier import stratified_subsample
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    configs = []
    for sc in SCENARIOS:
        for re in (800, 1200, 1600):
            dp = os.path.join(results_dir, f"dns_{sc}_Re{re}_N96.npz")
            pp = os.path.join(
                results_dir, f"patches_{sc}_Re{re}_N96_dim4.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                configs.append((sc, re, dp, pp))
    if not configs:
        pytest.skip("artefacts DNS/patches N96 dim4 absents")

    X_site, _, Y_snap, S_snap, tags = build_dataset(configs, 4, 30)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X_site))
    n_tr_sn = max(1, int(0.7 * len(X_site)))
    tr_sn, va_sn = perm[:n_tr_sn], perm[n_tr_sn:]

    def stack(pool, idxs):
        return np.concatenate([pool[i] for i in idxs], axis=0)
    Xtr_full = stack(X_site, tr_sn); Ytr_full = stack(Y_snap, tr_sn)
    Xva_full = stack(X_site, va_sn); Yva_full = stack(Y_snap, va_sn)

    tr_sel = stratified_subsample(Xtr_full, Ytr_full, n_train, rng)
    va_sel = stratified_subsample(Xva_full, Yva_full, n_val, rng)
    Xtr = Xtr_full[tr_sel]; Ytr = Ytr_full[tr_sel]
    Xva = Xva_full[va_sel]; Yva = Yva_full[va_sel]

    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=4, random_state=0).fit(scaler.transform(Xtr))
    Ptr = pca.transform(scaler.transform(Xtr))
    Pva = pca.transform(scaler.transform(Xva))
    return Ptr, Ytr, Pva, Yva


@pytest.mark.slow
def test_run_vqc_ignores_its_own_seed_argument():
    """Question 3 de VIGIL.md : `run_vqc` consomme-t-il ce que sa
    signature annonce ? Deux appels REELS (pas une reimplementation),
    arguments RIGOUREUSEMENT identiques, `seed=0` les deux fois : le F1,
    l'AUC et les probabilites individuelles different quand meme.
    `run_qke`, la fonction soeur, honore le sien
    (`SVC(random_state=seed)`). Mesure a 778255d : F1 0,653 puis 0,639,
    ecart max sur une probabilite individuelle 0,487 (la moitie de
    l'intervalle [0,1]) -- pas du bruit de mesure, une graine morte."""
    import h2b_variational_classifier as hvc

    Ptr, Ytr, Pva, Yva = _real_dataset_for_vqc()

    r1 = hvc.run_vqc(Ptr, Ytr, Pva, Yva, d_q=4, reps_fm=2, reps_ansatz=2,
                      maxiter=40, seed=0)
    r2 = hvc.run_vqc(Ptr, Ytr, Pva, Yva, d_q=4, reps_fm=2, reps_ansatz=2,
                      maxiter=40, seed=0)

    max_diff = float(np.max(np.abs(r1["p_va"] - r2["p_va"])))
    assert max_diff > 0.05, (
        "run_vqc(seed=0) appele deux fois de suite rend maintenant des "
        f"predictions quasi identiques (ecart max {max_diff:.4f}) -- si "
        "`seed` est desormais consomme (ansatz initial_point graine, "
        "algorithm_globals.random_seed, ...), le defaut semble corrige : "
        "REMESURER et deplacer cette note vers docs/RESULTS.md avant de "
        "considerer ce test comme un faux negatif")
    assert r1["f1"] != r2["f1"] or max_diff > 0.05, (
        "les deux appels rendent maintenant le meme F1 -- remesurer")
