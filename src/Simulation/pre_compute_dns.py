from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver

def _init_dns_scenario(sim, scenario):
    """Dispatch IC pour le DNS de pré-calcul."""
    init_map = {
        'orszag_tang':       sim.init_orszag_tang,
        'kelvin_helmholtz':  sim.init_kelvin_helmholtz,
        'magnetic_twist':    sim.init_magnetic_twist,
        'noisy_uniform':     sim.init_noisy_uniform,
        'harris_tearing':    sim.init_harris_tearing,
        'double_tearing':    sim.init_double_tearing,
        'lamb_oseen_vortex': sim.init_lamb_oseen_vortex,
        'island_coalescence': sim.init_island_coalescence,
        'mhd_rotor':         sim.init_mhd_rotor,
    }
    if scenario not in init_map:
        raise ValueError(
            f"scenario inconnu : {scenario!r}. Attendu l'un de "
            + ", ".join(sorted(init_map))
        )
    init_map[scenario]()


def precompute_dns(phase_config):
    """Compute the DNS trajectory once and return a lightweight trace.

    Time convention:

      ``dns_trace[k]['dt']`` is the duration of step ``k``;
      ``dns_trace[k]['fluxes']`` is the state after step ``k`` when that
      snapshot is retained.

    A single convention lets every scoring path compare a solver immediately
    after step ``k`` with ``dns_trace[k]``.

    Memory optimization: we store the full field snapshots (fluxes) only at
    hybrid-update boundaries and at the final step.  The per-step dt is
    always stored (it's a single scalar, negligible memory).  This reduces
    memory usage by ~60x compared to storing fluxes at every micro-step.

    For N=256, T_MAX=3.0, ~1000 steps:
      Before: ~1000 * 5 arrays * 256^2 * 8 bytes ~ 2.5 GB
      After:  ~1000 scalars + ~60 snapshots       ~ 150 MB
    """
    N = phase_config["N"]
    DT = phase_config["DT"]
    HYBRID_DT = phase_config.get("HYBRID_DT", 0.05)

    grid = PeriodicGrid(resolution_N=N)
    sim_dns = MHDSolver(grid, dt=DT, Re=phase_config["Re"], Rm=phase_config["Rm"])
    scenario = phase_config.get("scenario", "orszag_tang")
    _init_dns_scenario(sim_dns, scenario)
    phys_seed = int(phase_config.get("phys_seed", 0))
    noise_amplitude = float(phase_config.get("physics_noise_amplitude", 0.1))
    sim_dns.apply_physics_perturbation(phys_seed, noise_amplitude)

    t_current = 0.0
    T_MAX = phase_config["T_MAX"]
    T_START = phase_config.get("T_START", 0.0)

    dns_trace = {}
    hot_start_state = None
    step = 0
    next_snapshot_time = T_START - HYBRID_DT  # First snapshot at the hot-start time

    while t_current < T_MAX:
        dt = sim_dns.adapt_dt(cfl_target=0.4)
        dt = min(dt, T_MAX - t_current)
        # Le clamp doit etre REECRIT dans le solveur. `adapt_dt` fixe
        # `sim_dns.dt` et le rend ; le `min` ci-dessus ne creait qu'une
        # variable locale, si bien que `step_full` integrait avec le dt NON
        # borne pendant que `t_current` avancait du dt borne.
        #
        # Mesure sur orszag_tang N=32, T_MAX=0.05 : au dernier pas le
        # solveur integrait 0.037997804 alors que la comptabilite avancait
        # de 0.010730092. La trajectoire de REFERENCE finissait donc a
        # t ~ 0.077 tandis que la trace annoncait 0.050 — et le pipeline,
        # qui avance ses deux bras avec `dns_trace[step]['dt']`, les
        # comparait a une reference prise 3.5 fois plus loin dans le temps.
        sim_dns.dt = dt

        # Capture Hot-Start state
        if t_current >= T_START and hot_start_state is None:
            print(f"Hot-Start captured at t={t_current:.4f}")
            hot_start_state = {
                'vx': sim_dns.vx.copy(), 'vy': sim_dns.vy.copy(),
                'Bx': sim_dns.Bx.copy(), 'By': sim_dns.By.copy(),
                't_current': t_current,
                'step': step,
                'phys_seed': phys_seed,
                'physics_noise_amplitude': noise_amplitude,
            }

        entry = {'dt': dt}

        # Détection des frontières avec epsilon
        is_hybrid_boundary = (t_current >= next_snapshot_time - 1e-9 and t_current >= T_START - HYBRID_DT - 1e-9)
        is_last_step = (t_current + dt >= T_MAX - 1e-9)

        dns_trace[step] = entry
        sim_dns.step_full(record_stats=False)
        t_current += dt
        if is_hybrid_boundary or is_last_step:
            dns_trace[step]['fluxes'] = sim_dns.get_fluxes()
            if is_hybrid_boundary:
                next_snapshot_time += HYBRID_DT
        step += 1

        # Safety: abort if DNS diverges (garbage data would poison all trials)
        if sim_dns.is_diverged():
            raise RuntimeError(
                f"DNS diverged during precomputation at step {step-1} "
                f"(t={t_current:.4f}). Lower DT or reduce Re/Rm."
            )

    n_snapshots = sum(1 for v in dns_trace.values() if 'fluxes' in v)
    print(f"DNS pre-computed: {step} steps, {n_snapshots} flux snapshots stored.")
    return dns_trace, hot_start_state
