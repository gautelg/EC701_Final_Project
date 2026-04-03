"""
tests/test_closed_loop_pd.py

Basilisk closed-loop smoke test: PD controller driving the chaser to the
origin of the Hill frame (rendezvous point at target's location).

What this test verifies
-----------------------
1. The full pipeline runs without errors:
       Basilisk → BskInterface.read_state() → ControllerAdapter.step()
       → BskInterface.write_command() → Basilisk (next step)
2. No NaN values appear in position, velocity, or force.
3. The chaser converges: Hill-frame relative-position norm is substantially
   smaller at the end than at the start (factor of 10 over 2 orbits).

Initial conditions
------------------
Chaser starts 100 m behind the target on V-bar (same as zero-control test).
Desired relative position: origin [0, 0, 0] (rendezvous).

PD gains (for 500 kg spacecraft on ISS orbit, n ≈ 0.00113 rad/s)
-----------------------------------------------------------------
  Kp = 0.05 N/m   →   ωn = sqrt(Kp/m) ≈ 0.010 rad/s  (~9 × n)
  Kd = 14.0 N·s/m →   critically damped (Kd = 2·m·ωn)

Expected settling within ~400 s; 2 orbits (~11 400 s) gives large margin.

How to run (from repo root)
---------------------------
    python src/sim/tests/test_closed_loop_pd.py
Or:
    python -m pytest src/sim/tests/test_closed_loop_pd.py -v
"""

import os
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.dirname(_TESTS_DIR)
_SRC_DIR   = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Basilisk.utilities import macros, vizSupport

from core.bsk_sim         import BskSim
from core.bsk_environment import BskEnvironment
from core.bsk_spacecraft  import BskSpacecraft
from core.bsk_interface   import BskInterface
from adapter.controller_adapter import ControllerAdapter
from control.pd_controller import PDController


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_KP          = 0.05    # N/m
_KD          = 14.0    # N·s/m
_MAX_FORCE   = 50.0    # N  (per axis; ~0.1 m/s² for 500 kg spacecraft)
_MAX_TORQUE  = 10.0    # N·m
_N_ORBITS    = 2       # number of orbits to simulate
_T_ORBIT     = 5700.0  # s  (approximately one ISS-altitude orbit)
_CONVERGENCE_FACTOR = 10.0  # final distance must be < initial / this factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config():
    config_path = os.path.join(_SIM_DIR, "sim_config.yaml")
    with open(config_path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_closed_loop():
    config  = _load_config()
    sim_cfg = config["simulation"]

    dynamics_dt   = sim_cfg["dynamics_dt"]
    controller_dt = sim_cfg["controller_dt"]
    t_end         = _N_ORBITS * _T_ORBIT

    # ------------------------------------------------------------------
    # Build simulation objects (same pattern as test_two_sc.py)
    # ------------------------------------------------------------------
    bsk = BskSim(dynamics_dt, controller_dt)
    bsk.setup()

    env = BskEnvironment(bsk, config)
    env.setup()

    sc = BskSpacecraft(bsk, config)
    sc.setup(env)

    # ------------------------------------------------------------------
    # Build controller pipeline
    # ------------------------------------------------------------------
    pd_ctrl = PDController(Kp=_KP, Kd=_KD, desired_rel_pos=[0.0, 0.0, 0.0])
    adapter = ControllerAdapter(pd_ctrl, {"max_force": _MAX_FORCE, "max_torque": _MAX_TORQUE})
    iface   = BskInterface(sc, env)

    # ------------------------------------------------------------------
    # Vizard output
    # ------------------------------------------------------------------
    output_dir = os.path.join(_SRC_DIR, "sim", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path  = os.path.join(output_dir, "closed_loop_pd.bin")

    vizSupport.enableUnityVisualization(
        bsk.scSim,
        bsk.taskName,
        [sc.scTarget, sc.scChaser],
        saveFile=save_path,
    )

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    print("Initialising Basilisk simulation...")
    bsk.initialize()

    # ------------------------------------------------------------------
    # Closed-loop execution
    # ------------------------------------------------------------------
    t_current   = 0.0
    n_steps     = int(round(t_end / controller_dt))

    # Pre-allocate history arrays
    times     = np.zeros(n_steps + 1)
    rel_pos_h = np.zeros((n_steps + 1, 3))
    rel_vel_h = np.zeros((n_steps + 1, 3))
    forces    = np.zeros((n_steps + 1, 3))
    valids    = np.ones(n_steps + 1, dtype=bool)

    # Record initial state (before any propagation)
    state0          = iface.read_state()
    times[0]        = state0.time
    rel_pos_h[0]    = state0.rel_pos
    rel_vel_h[0]    = state0.rel_vel

    print(f"Running closed-loop for {n_steps} controller steps "
          f"({t_end:.0f} s / {_N_ORBITS} orbits)...")
    print(f"  Initial relative distance: {np.linalg.norm(state0.rel_pos):.1f} m")

    for k in range(n_steps):
        state = iface.read_state()
        cmd   = adapter.step(state)
        iface.write_command(cmd)

        t_current += controller_dt
        bsk.run(t_current)

        times[k + 1]     = state.time
        rel_pos_h[k + 1] = state.rel_pos
        rel_vel_h[k + 1] = state.rel_vel
        forces[k + 1]    = cmd.force
        valids[k + 1]    = cmd.valid

    final_state = iface.read_state()
    print(f"  Final relative distance:   {np.linalg.norm(final_state.rel_pos):.2f} m")
    print("Simulation complete.")

    return times, rel_pos_h, rel_vel_h, forces, valids, output_dir


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_results(times, rel_pos_h, forces, output_dir):
    t_min = times / 60.0

    # Relative position components
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Radial x [m]", "Along-track y [m]", "Cross-track z [m]"]
    for k, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(t_min, rel_pos_h[:, k])
        ax.set_ylabel(lbl)
        ax.grid(True)
    axes[-1].set_xlabel("Time [min]")
    axes[0].set_title(f"PD closed-loop — relative position (Kp={_KP}, Kd={_KD})")
    plt.tight_layout()
    path = os.path.join(output_dir, "closed_loop_pd_rel_pos.png")
    plt.savefig(path, dpi=150)
    print(f"Relative-position plot: {path}")

    # Distance to target
    dist = np.linalg.norm(rel_pos_h, axis=1)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_min, dist)
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("Distance to target [m]")
    ax2.set_title("PD closed-loop — chaser distance to target")
    ax2.grid(True)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "closed_loop_pd_distance.png")
    plt.savefig(path2, dpi=150)
    print(f"Distance plot:          {path2}")

    # Commanded force magnitude
    force_mag = np.linalg.norm(forces, axis=1)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t_min, force_mag)
    ax3.set_xlabel("Time [min]")
    ax3.set_ylabel("Force magnitude [N]")
    ax3.set_title("PD closed-loop — commanded force magnitude")
    ax3.grid(True)
    plt.tight_layout()
    path3 = os.path.join(output_dir, "closed_loop_pd_force.png")
    plt.savefig(path3, dpi=150)
    print(f"Force plot:             {path3}")

    plt.show()


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

def test_closed_loop_pd_convergence():
    times, rel_pos_h, rel_vel_h, forces, valids, _ = run_closed_loop()

    # No NaN anywhere
    assert not np.any(np.isnan(rel_pos_h)), "NaN in relative position"
    assert not np.any(np.isnan(rel_vel_h)), "NaN in relative velocity"
    assert not np.any(np.isnan(forces)),    "NaN in commanded forces"

    # Chaser converges: final distance < initial / CONVERGENCE_FACTOR
    dist_initial = np.linalg.norm(rel_pos_h[0])
    dist_final   = np.linalg.norm(rel_pos_h[-1])
    assert dist_final < dist_initial / _CONVERGENCE_FACTOR, (
        f"Expected convergence by factor {_CONVERGENCE_FACTOR}; "
        f"initial={dist_initial:.1f} m, final={dist_final:.2f} m"
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    times, rel_pos_h, rel_vel_h, forces, valids, output_dir = run_closed_loop()
    _plot_results(times, rel_pos_h, forces, output_dir)
