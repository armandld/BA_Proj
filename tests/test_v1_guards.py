"""
V1 guards — the checks that keep a V1 result from being read as something
it is not.

`src/` is frozen, so nothing here changes V1's behaviour. What these tests
do is make three of its silent behaviours *loud*, so that no present or
future V1 test can mistake them for a normal outcome:

  A. The empty Hamiltonian. `cost_hamiltonian.py` prunes every coefficient
     below COEFF_MIN = 1e-6. When that empties the term list it now raises
     `NullHamiltonianError` instead of injecting a ("Z", [0], 1e-3)
     placeholder: the patch defines no optimisation problem, and saying so
     is the only way the caller can tell that apart from a weak Hamiltonian.
     `refinement.py` catches it, keeps the classical decision for that patch
     and records it in `null_hamiltonian_patches()`.

  B. The pruning threshold. The Gaussian uncertainty window drives C_edges
     to ~1e-42, which is 36 orders of magnitude below the 1e-6 cut, so the
     ZZ terms are never constructed. "Ablating ZZ changes no decision" is
     therefore not a statement about the optimiser: there is nothing to
     ablate. This is pinned here so the chain window -> pruning -> empty
     ZZ family is checked mechanically rather than remembered.

  C. The swallowed exception. `execute.py:184` sets the sampler's shot count
     for the MPS final readout inside `try/except Exception: pass`. If that
     assignment ever stops working, the readout silently runs at the wrong
     shot count and every marginal downstream is quietly noisier. The test
     below performs the same assignment on the same primitive objects, so
     the failure surfaces here instead of hiding there.

Run with:
    python -m pytest tests/test_v1_guards.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qiskit.quantum_info import SparsePauliOp

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from VQA.cost_hamiltonian import (
    COEFF_MIN, NullHamiltonianError, create_period_hamiltonian,
)

DIM = 2                      # deployed size: 2*DIM^2 = 8 qubits
PRUNE_THRESHOLD = COEFF_MIN  # 1e-6


def _flat_params(value, dim=DIM):
    """hamilt_params with every coefficient set to `value`."""
    a = np.full((dim, dim), float(value))
    return {
        'H_edges': (a.copy(), a.copy()),
        'C_edges': (a.copy(), a.copy()),
        'K_plaquettes': a.copy(),
    }


def _diagonal_energies(op):
    """Energies of every computational basis state (the operator is diagonal)."""
    return np.real(np.diag(op.to_matrix()))


# ═══════════════════════════════════════════════════════════════════════
#  A. AN EMPTY HAMILTONIAN MUST RAISE, NOT BE FAKED
# ═══════════════════════════════════════════════════════════════════════

class TestNullHamiltonian:

    def test_all_coefficients_pruned_raises(self):
        """Every coefficient below COEFF_MIN -> NullHamiltonianError."""
        tiny = PRUNE_THRESHOLD / 1000.0          # 1e-9
        with pytest.raises(NullHamiltonianError) as excinfo:
            create_period_hamiltonian(_flat_params(tiny), DIM)

        err = excinfo.value
        assert err.num_qubits == 2 * DIM * DIM
        assert err.threshold == COEFF_MIN
        assert str(COEFF_MIN) in str(err) or f"{COEFF_MIN:g}" in str(err)

    def test_nothing_is_returned_that_could_pass_for_a_hamiltonian(self):
        """The whole point: no object comes back at all.

        The previous behaviour returned ("Z", [0], 1e-3) — a term 1e6 times
        larger than the signal it replaced, whose ground state excites qubit
        0, and which slipped past the all-zero shortcut in `execute.py`
        (atol 1e-8). Downstream had no way to tell it from a real operator.
        """
        with pytest.raises(NullHamiltonianError):
            create_period_hamiltonian(
                _flat_params(PRUNE_THRESHOLD / 1000.0), DIM)

    def test_a_coefficient_above_the_cut_builds_a_hamiltonian(self):
        """The boundary is the pruning cut and nothing else."""
        just_above = PRUNE_THRESHOLD * 2.0
        op = create_period_hamiltonian(_flat_params(just_above), DIM)
        assert len(op.to_list()) > 0
        assert op.num_qubits == 2 * DIM * DIM

        just_below = PRUNE_THRESHOLD / 2.0
        with pytest.raises(NullHamiltonianError):
            create_period_hamiltonian(_flat_params(just_below), DIM)

    def test_a_real_hamiltonian_is_built_normally(self):
        op = create_period_hamiltonian(_flat_params(0.5), DIM)
        assert len(op.to_list()) > 1
        energies = _diagonal_energies(op)
        assert energies.min() < energies.max()

    def test_refinement_exposes_the_null_patch_counter(self):
        """`refinement.py` records the event instead of hiding it."""
        from Simulation.refinement import (
            null_hamiltonian_patches, reset_null_hamiltonian_patches,
        )
        reset_null_hamiltonian_patches()
        assert null_hamiltonian_patches() == []


# ═══════════════════════════════════════════════════════════════════════
#  B. WINDOWED ZZ NEVER REACHES THE OPERATOR
# ═══════════════════════════════════════════════════════════════════════

class TestPruningThreshold:

    def test_windowed_zz_is_pruned_before_the_solver(self):
        """C_edges is computed, is nonzero, and is dropped by the 1e-6 cut.

        The three assertions are ordered so that the failure tells you which
        link broke: the coupling exists, it is below the cut, and no ZZ term
        appears in the operator.
        """
        N = DIM
        fields = {
            'vx': np.array([[2.0, 2.0], [0.0, 0.0]], dtype=float),
            'vy': np.zeros((N, N)),
            'Bx': np.zeros((N, N)),
            'By': np.zeros((N, N)),
            'Jz': np.zeros((N, N)),
        }
        hm = PhysicalMapper(cs=1.0, eta_mhd=0.01)
        sim = MHDSolver(PeriodicGrid(N))
        score = hm.physical_score(fields)
        hp = hm.compute_coefficients(sim, score, fields, 0.0)

        C_h, C_v = hp['C_edges']
        c_max = max(np.max(np.abs(C_h)), np.max(np.abs(C_v)))

        assert c_max > 0.0, "the gradient coupling must be computed, not skipped"
        assert c_max < PRUNE_THRESHOLD, (
            f"windowed coupling {c_max:.3e} is expected far below the "
            f"{PRUNE_THRESHOLD:g} pruning cut"
        )

        op = create_period_hamiltonian(hp, DIM)
        zz_terms = [t for t in op.to_list() if t[0].count("Z") == 2]
        assert zz_terms == [], (
            f"no ZZ term can reach the solver at this size, found {zz_terms}"
        )

    def test_a_coupling_above_the_cut_does_reach_the_operator(self):
        """Control: the ZZ family is not structurally absent."""
        hp = _flat_params(0.0)
        hp['C_edges'][0][:] = 0.5
        op = create_period_hamiltonian(hp, DIM)
        zz_terms = [t for t in op.to_list() if t[0].count("Z") == 2]
        assert len(zz_terms) == DIM * DIM, (
            "with a coupling above the cut, one ZZ term per site is expected"
        )


# ═══════════════════════════════════════════════════════════════════════
#  C. THE SWALLOWED EXCEPTION
# ═══════════════════════════════════════════════════════════════════════

class TestSamplerShotOption:
    """`execute.py:182-185` swallows any failure of

        sampler.options.default_shots = mps_shots

    If it fails, the MPS final readout runs at the previous shot count and
    nothing says so. The same assignment is exercised here, un-caught, on
    both primitive construction paths.
    """

    def test_legacy_path_accepts_the_shot_option(self):
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(mode=AerSimulator(method='matrix_product_state'))
        sampler.options.default_shots = 4096
        assert sampler.options.default_shots == 4096

        sampler.options.default_shots = 8192          # what execute() does
        assert sampler.options.default_shots == 8192, (
            "the assignment execute.py performs inside try/except must take "
            "effect, otherwise the MPS readout is silently under-sampled"
        )

    def test_runtime_path_accepts_the_shot_option(self):
        from VQA.runtime import VQARuntime

        rt = VQARuntime(backend_name="matrix_product_state", mode="local",
                        shots=4096, opt_level=1)
        assert rt.sampler.options.default_shots == 4096

        rt.sampler.options.default_shots = 8192
        assert rt.sampler.options.default_shots == 8192


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ═══════════════════════════════════════════════════════════════════════
#  D. THE REFINEMENT HANDLER, END TO END
# ═══════════════════════════════════════════════════════════════════════

class TestRefinementHandlesNullHamiltonian:
    """`refinement.py` must survive a patch with no Hamiltonian, keep its
    classical decision, and record the event.

    Before the change, such a patch received the injected placeholder and
    ran a full COBYLA optimisation against it; nothing distinguished it from
    a patch that had a real Hamiltonian.
    """

    def test_null_patch_is_recorded_and_the_scan_continues(self):
        from types import SimpleNamespace

        from Simulation import refinement as R
        from Simulation.PhysToAngle import AngleMapper

        N = 16
        grid = PeriodicGrid(N)
        sim = MHDSolver(grid, dt=1e-3, Re=400, Rm=400)
        # Champ parfaitement uniforme : aucun coefficient ne passera le seuil
        for name in ('vx', 'vy', 'Bx', 'By'):
            setattr(sim, name, np.full((N, N), 0.5))

        args = SimpleNamespace(
            reps=2, mode="simulator", backend="state_vector",
            shots=1024, method="COBYLA", opt_level=1,
            AdvAnomaliesEnable=False, K_opt=20, eps=1e-2,
            eta=0.001, Bz_guide=0.1, c_s=1.0, Re=400, Rm=400,
        )

        R.reset_null_hamiltonian_patches()
        result = R.run_adaptive_vqa(
            sim, AngleMapper(), PhysicalMapper(cs=1.0, eta_mhd=0.01,
                                               dx=grid.dx),
            args, None,
            beta=1.0, threshold_amr=0.3, target_dim=DIM,
            max_depth=1, min_size=4, verbose=False,
        )

        assert result is not None, "the scan must complete, not propagate"
        recorded = R.null_hamiltonian_patches()
        assert recorded, (
            "a uniform field defines no Hamiltonian anywhere, so at least "
            "one patch must be recorded"
        )
        assert all('bounds' in r and 'depth' in r for r in recorded)


# ═══════════════════════════════════════════════════════════════════════
#  E. THE CLASSICAL SCORE HAS NO ABSOLUTE ZERO
# ═══════════════════════════════════════════════════════════════════════

class TestClassicalScoreIsRelative:
    """`AngleMapper.classical_score` normalises each of its four indicators
    by its own domain-wide maximum, so the score measures RELATIVE structure
    and never how much structure there is.

    A field that is uniform plus 1e-12 of noise -- as calm as a field can be
    without being exactly constant -- scores a median of 0.237 and a maximum
    of 0.657, identically to the same field with 1e-6 of noise. Meanwhile a
    genuinely turbulent field (noise 1.0) scores LOWER, 0.574 at the maximum.

    Two consequences carried by the study:
      * a quiet uniform field is refined everywhere, because 0.55 > the
        threshold 0.3 (see tests/test_qaoa_decisions.py, check
        `quiet_no_refine`);
      * the classical baseline is not comparable across scenarios, which is
        why its LOSO F1 swings from 0.155 to 1.000 depending on the fold.

    Same disease as the per-scenario percentile label: a rank presented as a
    measurement.
    """

    @staticmethod
    def _score(amp, n=32, seed=0):
        from Simulation.PhysToAngle import AngleMapper
        rng = np.random.default_rng(seed)
        f = {k: np.full((n, n), 1.0) + amp * rng.normal(size=(n, n))
             for k in ("vx", "vy", "Bx", "By")}
        f["Jz"] = np.zeros((n, n))
        f["dx"] = 2 * np.pi / n
        return AngleMapper.classical_score(f)

    def test_a_numerically_calm_field_scores_high(self):
        s = self._score(1e-12)
        assert np.median(s) > 0.15, (
            f"a field at round-off scores {np.median(s):.4f} at the median; "
            "if this ever drops the score has gained an absolute zero")
        assert s.max() > 0.5

    def test_the_score_is_blind_to_amplitude(self):
        tiny, small = self._score(1e-12), self._score(1e-6)
        assert abs(np.median(tiny) - np.median(small)) < 1e-3, (
            f"six orders of magnitude of amplitude move the median score by "
            f"{abs(np.median(tiny) - np.median(small)):.2e} only -- it is "
            "normalised away")

    def test_more_turbulence_does_not_raise_the_score(self):
        calm, turbulent = self._score(1e-12), self._score(1.0)
        assert turbulent.max() < calm.max(), (
            f"turbulent max {turbulent.max():.4f} vs calm {calm.max():.4f}: "
            "the score is not monotone in the amount of structure")
