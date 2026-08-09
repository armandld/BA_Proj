### Primary endpoint (pre-registered): paired `combined`

| fold | scenario | Q-HAS combined | classical combined | delta (Q-HAS-cl) | better |
|---|---|---|---|---|---|
| ot | orszag_tang | 0.3328 | 0.4386 | -0.1058 | Q-HAS |
| kh | kelvin_helmholtz | 0.2443 | 0.1800 | +0.0643 | classical |
| rotor | mhd_rotor | 0.2273 | 0.9294 | -0.7021 | Q-HAS |
| tearing | harris_tearing | 0.2330 | 0.1817 | +0.0513 | classical |

- **validity unaudited**: run `t19_arm_divergence_audit.py` — an aborted arm is indistinguishable from a completed one in the stored output
- folds usable: 4/4 — Q-HAS better on 2, classical better on 2 (pre-registered rule: >= 3/4)
- TOST margin (5% of mean classical combined) = 0.0216; diff = -0.1731; p_TOST = 0.7685 => equivalence NOT established
- paired t p = 0.4084; Holm-adjusted = 0.8168 (difference), 0.8168 (equivalence)
- exact sign test p = 1.0000 (minimum attainable at n=4: 0.1250)

### Secondary (post-hoc, defect D4): equal-budget comparison

| fold | Q-HAS patch | Q-HAS phys | matched thr | matched patch | matched phys | Q-HAS/frontier | dominated? |
|---|---|---|---|---|---|---|---|
| ot | 0.6797 | 0.1940 | 0.1906 | 0.6412 | 0.0827 | 2.57x | yes |
| kh | 0.8376 | 0.0070 | 0.1906 | 0.7943 | 0.0017 | 4.41x | yes |
| rotor | 0.3761 | 0.1678 | 0.0969 | 0.3562 | 0.0536 | 3.62x | yes |
| tearing | 0.7692 | 0.0185 | 0.4250 | 0.6250 | 0.0044 | 4.38x | yes |

- Q-HAS strictly Pareto-dominated on 4/4 budget-matched folds
