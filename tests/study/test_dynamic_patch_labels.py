"""La verite terrain DYNAMIQUE mesure-t-elle ce qu'elle annonce ?

Protocole v3 §1.2, tache 6. Le label `d_i` est la distance, APRES evolution
de `delta_t`, entre le champ de reference et le champ ou le patch i seul a
ete remplace par sa moyenne.

Ce fichier verifie trois choses, dans cet ordre d'importance :

1. **Que le calcul est juste** — par une identite analytique, `d0 = e/dim`,
   qui lie le nouveau label a l'ancien sans passer par le solveur.
2. **Que la sequence de pas est bien gelee** — sans quoi `d_i` melangerait
   une difference de physique et une difference de pas de temps.
3. **Que le label n'est pas une redite du label statique** — et c'est le
   test qui a change la lecture du protocole : a `delta_t = 0,1`, la valeur
   que le protocole impose, il L'EST.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [os.path.join(_RACINE, "src")] + [
        os.path.join(_RACINE, "study", _d) for _d in ("pipeline", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_patch_labels import (                           # noqa: E402
    coarsen_one_patch, dynamic_patch_errors, evolue, sequence_de_pas,
    spearman, analyse_snapshot, DELTA_T, MAX_SUBSTEPS)
from hard_patch_labels import patch_l2_errors                # noqa: E402


def _champ(N=32, graine=0):
    rng = np.random.default_rng(graine)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return (np.sin(Y) + 0.3 * rng.standard_normal((N, N)),
            np.cos(X) + 0.3 * rng.standard_normal((N, N)),
            np.tanh(np.sin(Y)),
            np.sin(2 * X))


# ==================================================================
#  1. le calcul est-il juste ? — une identite, pas une tolerance
# ==================================================================
@pytest.mark.parametrize("dim", [2, 4, 8])
def test_la_perturbation_initiale_vaut_exactement_le_label_statique_sur_dim(dim):
    """`d0 = e / dim`, EXACTEMENT — et voici la demonstration.

    `e_i` est la RMS de l'ecart a la moyenne, prise sur le patch seul :
        e_i = sqrt( mean_patch(diff^2) ) / rms
    `d0_i` est la meme RMS prise sur le DOMAINE, ou l'ecart est nul partout
    ailleurs :
        d0_i = sqrt( mean_domaine(diff^2) ) / rms
             = sqrt( (p^2 / N^2) * mean_patch(diff^2) ) / rms
             = (p / N) * e_i  =  e_i / dim

    Cette identite ne passe par aucune integration en temps : elle teste le
    grossissement et la normalisation, isolement du solveur. Si elle tombe,
    c'est que l'un des deux est faux — et aucune tolerance ne le cacherait.
    """
    N = 32
    champs = _champ(N)
    d, d0, _ = dynamic_patch_errors(*champs, dim, Re=400, delta_t=1e-6)
    e = patch_l2_errors(*champs, dim)
    np.testing.assert_allclose(d0, e / dim, rtol=1e-12, atol=1e-14)


def test_lidentite_est_FAUSSE_si_le_patch_nest_pas_remplace_par_sa_moyenne():
    """Le garde du garde : l'identite ci-dessus doit pouvoir tomber.

    On remplace le patch par sa MEDIANE au lieu de sa moyenne — un
    grossissement plausible, et faux. `d0` cesse alors de valoir `e/dim`.
    """
    N, dim = 32, 4
    champs = _champ(N)
    p = N // dim
    e = patch_l2_errors(*champs, dim)

    faux = np.zeros((dim, dim))
    rms = float(np.sqrt(np.mean(sum(c ** 2 for c in champs))))
    for pi in range(dim):
        for pj in range(dim):
            diff = 0.0
            for c in champs:
                v = c.copy()
                bloc = c[pi * p:(pi + 1) * p, pj * p:(pj + 1) * p]
                v[pi * p:(pi + 1) * p, pj * p:(pj + 1) * p] = np.median(bloc)
                diff = diff + (v - c) ** 2
            faux[pi, pj] = np.sqrt(np.mean(diff)) / rms

    assert not np.allclose(faux, e / dim, rtol=1e-6), (
        "mediane et moyenne donnent le meme d0 : le champ d'essai est "
        "symetrique, ce test ne separe rien")


def test_le_patch_grossi_ne_touche_que_son_patch():
    """Une variante ne doit differer de la reference QUE dans son patch."""
    N, dim = 32, 4
    p = N // dim
    champ = _champ(N)[0]
    var = coarsen_one_patch(champ, 1, 2, p)
    dehors = np.ones((N, N), dtype=bool)
    dehors[1 * p:2 * p, 2 * p:3 * p] = False
    np.testing.assert_array_equal(var[dehors], champ[dehors])
    assert not np.allclose(var[~dehors], champ[~dehors]), \
        "le patch vise n'a pas bouge : le grossissement est l'identite"
    assert np.allclose(var[~dehors], champ[~dehors].mean()), \
        "le patch n'a pas ete remplace par sa moyenne"


def test_un_patch_dune_cellule_est_refuse():
    """A p=1 le grossissement est l'identite : tout label est nul et
    l'artefact aurait la bonne forme sans vouloir rien dire. Meme famille
    que `test_label_degenere_quand_le_patch_est_trop_petit.py`."""
    with pytest.raises(ValueError, match="dim <="):
        dynamic_patch_errors(*_champ(8), 8, Re=400, delta_t=1e-6)


def test_une_dimension_non_divisante_est_refusee():
    with pytest.raises(ValueError, match="divisible"):
        dynamic_patch_errors(*_champ(32), 5, Re=400, delta_t=1e-6)


# ==================================================================
#  2. la sequence de pas est-elle gelee ?
# ==================================================================
def test_la_sequence_de_pas_tombe_exactement_sur_lhorizon():
    """Reference et variantes doivent arriver au MEME instant."""
    pas, _ = sequence_de_pas(32, 400, _champ(32), delta_t=0.05)
    assert pas.sum() == pytest.approx(0.05, rel=1e-12)
    assert (pas > 0).all()


def test_rejouer_la_meme_sequence_rend_le_meme_champ():
    """`evolue` doit etre deterministe a sequence donnee."""
    champs = _champ(32)
    pas, _ = sequence_de_pas(32, 400, champs, delta_t=0.02)
    a = evolue(32, 400, champs, pas)
    b = evolue(32, 400, champs, pas)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_la_sequence_de_la_reference_nest_PAS_celle_qu_adapterait_la_variante():
    """Le champ qui SEPARE — la raison d'etre du gel.

    `dns_sweep.py` adapte le pas a chaque iteration. Une variante, dont le
    champ differe, adapterait donc une sequence DIFFERENTE. Si on la laissait
    faire, `d_i` compterait cet ecart de pas comme de la physique.

    Ce test mesure que les deux sequences different reellement : sans cela,
    geler la sequence serait une precaution sans objet et personne ne
    s'apercevrait de sa disparition.
    """
    N, dim = 32, 4
    p = N // dim
    champs = list(_champ(N))
    # `adapt_dt` lit des maxima GLOBAUX : grossir un patch quelconque ne les
    # deplace pas au premier pas. Une premiere version de ce test grossissait
    # un patch banal et concluait que le gel ne servait a rien — faux, et la
    # mesure sur les vrais champs le dit (100 % des patches divergent sur 3
    # scenarios sur 4, l'evolution deplacant les maxima des le premier pas).
    # Ici on force le cas au pas ZERO, pour un test rapide et deterministe :
    # l'extremum global est place DANS le patch qu'on grossit.
    champs[0] = champs[0].copy()
    champs[0][:p, :p] = 40.0 * np.linspace(-1, 1, p)[:, None]
    champs = tuple(champs)

    pas_ref, _ = sequence_de_pas(N, 400, champs, delta_t=0.05)
    var = tuple(coarsen_one_patch(c, 0, 0, p) for c in champs)
    pas_var, _ = sequence_de_pas(N, 400, var, delta_t=0.05)

    memes_valeurs = (len(pas_ref) == len(pas_var)
                     and np.allclose(pas_ref, pas_var, rtol=1e-12))
    assert not memes_valeurs, (
        "reference et variante adaptent la MEME sequence de pas alors que "
        "l'extremum global est dans le patch grossi : le gel de la sequence "
        "ne protege plus de rien, verifier `adapt_dt`")


@pytest.mark.slow
def test_sur_les_vrais_champs_le_gel_change_la_sequence_sur_presque_tout_patch():
    """Le gel est-il une precaution theorique ? Non — mesure le 21 aout 2026,
    N=96, Re=400, delta_t=0,05, fraction des patches dont la variante
    adapterait une sequence DIFFERENTE de la reference :

        harris_tearing       0 / 16  et   0 / 64     (0 %)
        kelvin_helmholtz    16 / 16  et  64 / 64   (100 %)
        mhd_rotor           16 / 16  et  64 / 64   (100 %)
        orszag_tang         16 / 16  et  64 / 64   (100 %)

    Sur trois scenarios sur quatre, SANS le gel, `d_i` compterait un ecart de
    pas de temps comme de la physique sur CHAQUE patch.
    """
    dns = os.path.join(_RACINE, "results", "dns_orszag_tang_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")
    d = np.load(dns)
    si = len(d["t"]) // 2
    champs = tuple(d[k][si].astype(float) for k in ("vx", "vy", "Bx", "By"))
    ref, _ = sequence_de_pas(96, 400, champs, delta_t=0.05)

    divergents = 0
    for pi in range(4):
        for pj in range(4):
            var = tuple(coarsen_one_patch(c, pi, pj, 24) for c in champs)
            v, _ = sequence_de_pas(96, 400, var, delta_t=0.05)
            if len(v) != len(ref) or not np.allclose(v, ref, rtol=1e-12):
                divergents += 1
    assert divergents >= 12, (
        f"{divergents}/16 patches divergent en sequence de pas : le gel a "
        "cesse d'etre necessaire, ce qui change la justification du module")


def test_un_horizon_absurde_crie_au_lieu_de_tourner():
    """Un balayage qui ne finira jamais doit lever, pas ramer."""
    with pytest.raises(RuntimeError, match="sous-pas"):
        sequence_de_pas(32, 400, _champ(32), delta_t=1e6)


# ==================================================================
#  3. le label dit-il autre chose que le label statique ?
# ==================================================================
#: Mesure du 21 aout 2026, N=96, Re=400, dim=4, 2 instantanes par scenario,
#: moyenne sur les 4 scenarios canoniques. C'est le nombre qui condamne
#: l'horizon du protocole.
_RHO_ATTENDU = {0.1: 0.99, 2.0: 0.81}


def test_spearman_rend_un_pour_une_transformation_monotone():
    """Le calcul de rho, verifie sur une reponse connue avant d'en tirer
    une conclusion."""
    x = np.array([3.0, 1.0, 4.0, 1.5, 5.0, 9.0])
    assert spearman(x, 2 * x + 1) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones_like(x)))


@pytest.mark.slow
def test_a_lhorizon_du_protocole_le_label_dynamique_est_une_redite_du_statique():
    """LE resultat de la tache 6, et il contredit le protocole.

    Le protocole §1.2 fixe `delta_t = one hybrid step (0.1)` et demande comme
    seul controle « Spearman(d_i, e_i) > 0 ». Ce controle est verifie — et il
    ne suffit pas : a cet horizon, rho vaut **0,99**. Le label dynamique est
    une renumerotation monotone du label statique, il ne repond donc pas au
    probleme de specification de tache (H5) pour lequel il a ete demande.

    Sur quelle entree ce test echoue : si rho tombait sous 0,95 a
    `delta_t = 0,1`, le label dynamique deviendrait informatif a l'horizon du
    protocole et cette lecture serait a reecrire.
    """
    dns = os.path.join(_RACINE, "results", "dns_harris_tearing_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")
    n = len(np.load(dns)["t"])
    r = analyse_snapshot(dns, n // 2, 4, delta_t=DELTA_T)
    rho = spearman(r["d_errors"], r["l2_errors"])

    assert rho > 0.0, (
        f"rho = {rho:+.4f} : le controle du protocole lui-meme echoue")
    assert rho > 0.95, (
        f"rho = {rho:+.4f} a delta_t = {DELTA_T} : le label dynamique s'est "
        "decolle du statique a l'horizon du protocole — c'est une bonne "
        "nouvelle, et la lecture publiee est a reecrire")


@pytest.mark.slow
def test_le_label_se_decolle_quand_on_allonge_lhorizon():
    """Le champ qui SEPARE : sans lui, le test ci-dessus laisserait croire
    que `d` est structurellement condamne a redire `e`.

    Mesure : rho passe de 0,99 (delta_t=0,1) a 0,81 (delta_t=2,0). Ce n'est
    pas le label qui est mauvais, c'est l'horizon qui est trop court.
    """
    dns = os.path.join(_RACINE, "results", "dns_orszag_tang_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")
    n = len(np.load(dns)["t"])
    court = spearman(*[analyse_snapshot(dns, n // 2, 4, delta_t=0.1)[k]
                       for k in ("d_errors", "l2_errors")])
    long_ = spearman(*[analyse_snapshot(dns, n // 2, 4, delta_t=2.0)[k]
                       for k in ("d_errors", "l2_errors")])
    assert court > long_ + 0.05, (
        f"rho court {court:+.4f} contre long {long_:+.4f} : allonger "
        "l'horizon ne decolle plus le label, la conclusion change")


# ==================================================================
#  4. l'artefact et sa tracabilite
# ==================================================================
def test_le_script_refuse_un_artefact_absent_au_lieu_de_rendre_du_vide():
    r = subprocess.run(
        [sys.executable, os.path.join(_RACINE, "study", "pipeline",
                                      "dynamic_patch_labels.py"),
         "--scenario", "scenario_qui_nexiste_pas", "--dry-run"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "artefact DNS absent" in (r.stdout + r.stderr)


def test_lartefact_porte_sa_provenance(tmp_path):
    """Regle du depot : hash du commit et arguments CLI complets dans le npz."""
    dns = os.path.join(_RACINE, "results", "dns_harris_tearing_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")
    out = str(tmp_path / "d.npz")
    r = subprocess.run(
        [sys.executable, os.path.join(_RACINE, "study", "pipeline",
                                      "dynamic_patch_labels.py"),
         "--snaps", "1", "--dim", "4", "--N", "96", "--delta-t", "0.01",
         "--out", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    z = np.load(out, allow_pickle=True)
    for cle in ("d_errors", "d0_errors", "amplification", "l2_errors",
                "is_hard_dynamic", "d_threshold", "rho_d_vs_e",
                "git_hash", "argv", "label_kind", "delta_t"):
        assert cle in z.files, f"cle absente de l'artefact : {cle}"
    assert str(z["label_kind"]) == "dynamic"
    assert str(z["git_hash"]) != ""
    assert "--delta-t" in str(z["argv"])


def test_deux_horizons_ecrivent_DEUX_fichiers(tmp_path, monkeypatch):
    """`delta_t` doit etre dans le nom du fichier.

    C'est le parametre qui decide si le label dit quelque chose : a 0,1 il
    redit le label statique (rho = 0,995), a 2,0 il s'en decolle sur le seul
    scenario turbulent. Deux horizons partageant un nom ecraseraient la
    mesure precedente en silence — et le fichier survivant aurait la bonne
    forme, les bonnes valeurs, et designerait autre chose que ce qu'on croit.

    Ce test appelle le VRAI `main` deux fois et compte les fichiers ecrits.
    Une premiere version reconstruisait le nom dans le test lui-meme : elle
    ne pouvait pas echouer, puisqu'elle ne lisait pas le code teste.
    """
    dns = os.path.join(_RACINE, "results", "dns_harris_tearing_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")

    import dynamic_patch_labels as dpl
    monkeypatch.setattr(dpl, "RESULTS_DIR", str(tmp_path))
    # le DNS est lu depuis `RESULTS_DIR` : on l'y expose par un lien
    os.symlink(dns, os.path.join(str(tmp_path), os.path.basename(dns)))

    for dt in ("0.01", "0.02"):
        monkeypatch.setattr(sys, "argv", [
            "dynamic_patch_labels", "--snaps", "1", "--dim", "4",
            "--N", "96", "--delta-t", dt])
        dpl.main()

    ecrits = sorted(f for f in os.listdir(str(tmp_path))
                    if f.startswith("d_patches_"))
    assert len(ecrits) == 2, (
        f"{len(ecrits)} fichier(s) ecrit(s) pour deux horizons : {ecrits}. "
        "Le second a ecrase le premier — `delta_t` n'est plus dans le nom.")
    assert all("dt0.0" in f for f in ecrits),         f"l'horizon n'apparait pas dans les noms : {ecrits}"


def test_l2_errors_de_lartefact_est_bien_le_label_STATIQUE(tmp_path):
    """La deviation assumee au protocole, epinglee.

    Le protocole demande un artefact « drop-in » ou le label dynamique
    prendrait la place de `l2_errors`. Ce serait un artefact a la bonne forme
    dont une cle designe autre chose que son nom — la classe de defaut que
    `CODE_REVIEW.md` retient comme la seule qui compte. Ici `l2_errors`
    contient le label STATIQUE, recalcule sur le meme instantane, et le
    label dynamique s'appelle `d_errors`.
    """
    dns = os.path.join(_RACINE, "results", "dns_harris_tearing_Re400_N96.npz")
    if not os.path.exists(dns):
        pytest.skip("artefact DNS N=96 absent")
    out = str(tmp_path / "d.npz")
    subprocess.run(
        [sys.executable, os.path.join(_RACINE, "study", "pipeline",
                                      "dynamic_patch_labels.py"),
         "--snaps", "1", "--dim", "4", "--N", "96", "--delta-t", "0.01",
         "--out", out], capture_output=True, text=True, check=True)

    z = np.load(out, allow_pickle=True)
    si = int(z["snap_index"][0])
    d = np.load(dns)
    attendu = patch_l2_errors(*[d[k][si].astype(float)
                                for k in ("vx", "vy", "Bx", "By")], 4)
    np.testing.assert_allclose(z["l2_errors"][0], attendu, rtol=1e-12)
    assert not np.allclose(z["d_errors"][0], attendu), (
        "d_errors et l2_errors coincident : soit le solveur n'a pas tourne, "
        "soit les deux cles portent la meme chose")
