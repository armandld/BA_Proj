### Primary endpoint (pre-registered): paired `combined`

| fold | scenario | Q-HAS combined | classical combined | delta (Q-HAS-cl) | better |
|---|---|---|---|---|---|
| kh | kelvin_helmholtz | 0.2443 | 0.1800 | +0.0643 | classical |
| ot | orszag_tang | 0.3328 | 0.4386 | -0.1058 | Q-HAS |
| tearing | harris_tearing | 0.2330 | 0.1817 | +0.0513 | classical |

- **excluded as failed** (pre-registration §5, an arm did not complete its trajectory): rotor
- folds usable: 3/3 — Q-HAS better on 1, classical better on 2 (diagnostic rule: >= 3/3)
- TOST margin (5% of mean classical combined) = 0.0133; diff = +0.0033; p_TOST = 0.4353 => equivalence NOT established
- paired t p = 0.9579; Holm-adjusted = 0.9579 (difference), 0.8707 (equivalence)
- exact sign test p = 1.0000 (minimum attainable at n=3: 0.2500)

### Secondary (post-hoc, defect D4): equal-budget comparison

| fold | Q-HAS patch | Q-HAS phys | matched thr | matched patch | matched phys | Q-HAS/frontier | dominated? |
|---|---|---|---|---|---|---|---|
| kh | 0.8704 | 0.0032 | 0.1906 | 0.7943 | 0.0017 | 2.10x | yes |
| ot | 0.7558 | 0.1073 | 0.1906 | 0.6412 | 0.0827 | 1.79x | yes |
| tearing | 0.9102 | 0.0080 | 0.4250 | 0.6250 | 0.0044 | 1.98x | yes |

- Q-HAS strictly Pareto-dominated on 3/3 budget-matched folds
