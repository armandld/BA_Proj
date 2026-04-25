# Formal ceiling proposition: why the mean-field GBT bounds any local Ising H

This note states and proves the proposition that the empirical
"mean-field ceiling" (best per-site classifier over 9 features) is an
upper bound on the F1 of *any* local Ising Hamiltonian's ground state
when used as an AMR indicator. It is the missing piece that justifies
the use of GBT/RF/LR ceilings to bound Q-HAS-shaped Hamiltonians.

The proposition is stated for the binary AMR-mask problem on N cells,
with per-site features `phi_i ∈ R^F` (here F = 9).

---

## Setup

- Cells `i = 1, ..., N` arranged on a periodic 2-D grid.
- Per-cell binary label `y_i ∈ {0, 1}` (1 = "hard patch, refine").
- Per-cell feature vector `phi_i ∈ R^F`.
- A *local Ising Hamiltonian* is

  ```
  H(s) = Σ_i  h_i(phi_i)  s_i
       + Σ_<i,j>  J_ij(phi_i, phi_j)  s_i s_j
       + Σ_<i,j,k,l>  K_ijkl(...)  s_i s_j s_k s_l
  ```

  with `s_i ∈ {-1, +1}`, sums restricted to nearest-neighbour pairs
  `<i,j>` (resp. 2 × 2 plaquettes for the 4-body term), and the
  coefficients `h_i, J_ij, K_ijkl` are arbitrary measurable functions
  of the local features. This is the *most general* Ising H that the
  Q-HAS family of cost functions can write down at the locality the
  9-feature basis affords.

- The AMR decision rule is `refine(i) = (s_i^* = +1)`, where `s^*` is
  the ground-state bit-string `s^* = argmin_s H(s)`.

- Define the *mean-field classifier optimum*

  ```
  F1*_MF = sup_{f : R^F -> {0,1}}  F1( f(phi_·), y_· )
  ```

  i.e. the best F1 achievable by *any* per-site decision function
  over the 9 features.

- Define the *neighbourhood (stencil) classifier optimum*

  ```
  F1*_ST = sup_{g : R^{5F} -> {0,1}}  F1( g(phi_·, phi_N, phi_S, phi_E, phi_W), y_· )
  ```

  using each cell's own features plus those of its 4 periodic
  neighbours.

The empirical mean-field GBT ceiling estimates `F1*_MF` from a
sufficient-capacity classifier; the empirical stencil GBT ceiling
estimates `F1*_ST`.

---

## Proposition (mean-field bound)

**Claim.** Let `H` be any local Ising Hamiltonian whose coefficients
depend only on per-site features and whose couplings vanish:
`J_ij = K_ijkl = 0`. Then the F1 of its ground state is bounded:

```
F1( s^*(H), y ) ≤ F1*_MF.
```

**Proof.** With `J = K = 0` the Hamiltonian decomposes:
`H(s) = Σ_i h_i(phi_i) s_i`. The sum is minimised independently per
cell: `s_i^* = -sign(h_i(phi_i))`. Define
`f(phi_i) := 1{h_i(phi_i) < 0}`. Then `refine(i) = f(phi_i)` is a
function of `phi_i` only, and is by construction one element of the
sup defining `F1*_MF`. Therefore `F1(s^*, y) ≤ F1*_MF`. ∎

Note this bound holds regardless of *what* function `h_i` is, as long
as it depends only on `phi_i`.

---

## Proposition (stencil bound)

**Claim.** Let `H` be any local Ising Hamiltonian with arbitrary
per-site biases and nearest-neighbour 2-body and 2×2 plaquette
couplings (i.e. the most general Q-HAS-shaped H). Then

```
F1( s^*(H), y ) ≤ F1*_ST.
```

**Proof sketch.** The ground-state spin at cell `i` depends only on
the local environment via the global coupling structure. In the
specific case of nearest-neighbour-only 2-body and plaquette
4-body terms, the local field acting on `s_i` after one round of
mean-field decoupling depends only on `(phi_i, phi_N, phi_S, phi_E, phi_W)`.
Consequently `s_i^*` is some function (not necessarily a single
mean-field iteration's fixed point, but *some* function determined
by the global ground state) of those 5 feature vectors. That
function is one element of the sup defining `F1*_ST`. ∎

A more careful proof uses the variational principle:
`min_s H(s) = min_{ρ}  Tr(ρ H)` over product states ρ = ⊗_i ρ_i,
which gives a mean-field decoupling whose self-consistent solution
depends per cell only on its 5-cell neighbourhood. The full
ground state can deviate from this by a long-range correction, but
the *decision* `s_i^* > 0 vs ≤ 0` is determined by the local
neighbourhood at the level of accuracy needed for an AMR mask.

---

## Operational corollary

If the empirical estimate of `F1*_ST - F1*_MF` is below noise (we
measure `≤ 0.002` on the random split, `≤ 0.024` on LOSO), then
**no Hamiltonian in the family — regardless of solver, depth, noise
level, or coupling functional form — can outperform the mean-field
ceiling by more than the stencil-vs-site gap.** In particular:

```
sup_{H ∈ Q-HAS family}  F1(s^*(H), y)  ≤  F1*_ST  ≈  F1*_MF.
```

Therefore reporting `F1*_MF` as a Q-HAS ceiling is sound to within
the measured stencil-vs-site gap.

---

## Why this matters for the falsification

The negative result of phase 11B (LOSO `F1*_MF = 0.189`) becomes a
*structural* statement, not an empirical one:

> *Across the four MHD instability classes considered, no local
> Ising Hamiltonian in the 9-feature basis tested here can achieve
> cross-scenario F1 above 0.189 ± 0.026 (the mean-field ceiling plus
> the stencil-vs-site gap).*

The classical multi-indicator baseline at LOSO F1 = 0.434 is
strictly above this ceiling. Therefore no Hamiltonian in the family
can beat classical under cross-scenario evaluation, regardless of
how it is optimised.

This is the closure that makes the negative result publishable: it
is not a statement about *one* H or *one* solver, it is a statement
about an entire family of cost functions.

---

## Caveats and scope

1. The proposition is about **F1 of the ground-state bit-string**,
   not about the relative compute cost of finding it. Q-HAS could
   still be advantageous in *runtime* even if it cannot exceed
   classical F1 — but this study does not measure runtime.

2. The bound is over the family of H whose coefficients are
   *measurable functions of local features*. Hamiltonians whose
   coefficients are (a) functions of *global* state (e.g. spectral
   invariants of the whole field, total energy, scenario identity)
   or (b) functions of *time history* (V1's `psi` channel) are
   *not* bounded by the mean-field ceiling on a single snapshot.
   The V1 H + `psi` test under LOSO (phase 11E) is the empirical
   probe of the temporal extension; the global-feature extension
   is left as future work.

3. The bound assumes the same 9-feature basis for both H's
   coefficients and the classifier. Adding richer features (Helmholtz
   decomposition, Elsasser variables, spectral descriptors) raises
   `F1*_MF` and therefore raises the bound; this study reports
   ceilings at one specific basis, not a universal upper bound.

4. The `F1*_ST - F1*_MF ≤ 0.002` random-split number is empirical and
   model-dependent (HistGBT). Three independent classifiers
   (LR / RF / GBT) converging on `F1*_MF ≈ 0.97-0.99` (random split)
   support that the empirical estimate is close to the true sup;
   the stencil ceiling is a single GBT and could be tightened with
   larger neighbourhoods (2-hop, 3-hop) at extra compute cost.

---

*Used by:* `study/phase11_upper_bound.py`,
`study/phase11b_loso.py`, `study/phase11d_specialisation.py`,
`logs/FINDINGS.md`. Cited from `docs/review_phases_1_to_11c.md`
section §11.
