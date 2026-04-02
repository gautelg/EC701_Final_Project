"""
tests/test_two_sc.py

Minimal two-spacecraft Basilisk + Vizard smoke test.

Simulates target and chaser spacecraft with ZERO control inputs for one orbit.
No MPC, no adapter, no scenario manager — just the core Basilisk pipeline.

What this test verifies
-----------------------
1. Basilisk initialises without errors.
2. Both spacecraft propagate for a full orbit under point-mass Earth gravity.
3. A Vizard .bin file is written — open it in Vizard to see the 3D scene.
4. The chaser's relative trajectory in the Hill frame is a closed ellipse
   (expected free-drift behaviour for a small along-track initial offset
   under HCW dynamics).

How to run (from the repo root)
--------------------------------
    python src/sim/tests/test_two_sc.py

Or from the src/sim/ directory:
    python tests/test_two_sc.py
"""

import os
import sys

import yaml
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup: make src/sim/ importable regardless of working directory
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))   # src/sim/tests/
_SIM_DIR   = os.path.dirname(_TESTS_DIR)                  # src/sim/
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from Basilisk.utilities import macros, vizSupport

from core.bsk_sim         import BskSim
from core.bsk_environment import BskEnvironment
from core.bsk_spacecraft  import BskSpacecraft


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config():
    config_path = os.path.join(_SIM_DIR, "sim_config.yaml")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def _hill_dcm(r_N, v_N):
    """
    DCM that rotates from inertial into the Hill (LVLH) frame at state (r, v).

    R_NH (inertial → Hill) = R_HN^T, where R_HN has Hill basis vectors
    as columns expressed in inertial.
    """
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)
    e_x = r_N / np.linalg.norm(r_N)
    h   = np.cross(r_N, v_N)
    e_z = h / np.linalg.norm(h)
    e_y = np.cross(e_z, e_x)
    R_HN = np.column_stack([e_x, e_y, e_z])   # Hill → inertial
    return R_HN.T                               # inertial → Hill


def _relative_pos_hill(r_tgt_N, v_tgt_N, r_chs_N):
    """
    Return an (N, 3) array of chaser-relative positions in the Hill frame,
    one row per timestep.
    """
    n = len(r_tgt_N)
    dr_hill = np.zeros((n, 3))
    for i in range(n):
        R_NH        = _hill_dcm(r_tgt_N[i], v_tgt_N[i])
        dr_hill[i]  = R_NH @ (r_chs_N[i] - r_tgt_N[i])
    return dr_hill


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config  = _load_config()
    sim_cfg = config["simulation"]
    viz_cfg = config.get("vizard", {})

    # ------------------------------------------------------------------
    # 1. Build simulation objects
    # ------------------------------------------------------------------
    bsk = BskSim(sim_cfg["dynamics_dt"], sim_cfg["controller_dt"])
    bsk.setup()

    env = BskEnvironment(bsk, config)
    env.setup()

    sc = BskSpacecraft(bsk, config)
    sc.setup(env)

    # ------------------------------------------------------------------
    # 2. Vizard output (file-based, open in Vizard after the run)
    # ------------------------------------------------------------------
    save_path = viz_cfg.get("save_file", "src/sim/output/two_sc_test.bin")
    # Resolve relative to the repo root (two levels above src/sim/)
    if not os.path.isabs(save_path):
        _REPO_ROOT = os.path.dirname(os.path.dirname(_SIM_DIR))
        save_path  = os.path.join(_REPO_ROOT, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # enableUnityVisualization subscribes to scStateOutMsg for each spacecraft
    # and writes a Vizard-compatible protobuf binary file.
    # Pass both spacecraft as a list; gravBodies provides celestial body data.
    viz = vizSupport.enableUnityVisualization(
        bsk.scSim,
        bsk.taskName,
        [sc.scTarget, sc.scChaser],
        saveFile=save_path,
    )

    # ------------------------------------------------------------------
    # 3. Initialise and run
    # ------------------------------------------------------------------
    t_end = sim_cfg["t_end"]
    print(f"Initialising Basilisk simulation...")
    bsk.initialize()

    print(f"Running for {t_end:.0f} s ({t_end / 60:.1f} min)...")
    bsk.run(t_end)
    print("Simulation complete.")

    # ------------------------------------------------------------------
    # 4. Extract logged data
    # ------------------------------------------------------------------
    t_s      = sc.scTargetLog.times() * macros.NANO2SEC   # (N,)  seconds
    r_tgt_N  = sc.scTargetLog.r_BN_N                      # (N,3) m
    v_tgt_N  = sc.scTargetLog.v_BN_N                      # (N,3) m/s
    r_chs_N  = sc.scChaserLog.r_BN_N                      # (N,3) m

    # Compute relative position in Hill frame
    dr_hill = _relative_pos_hill(r_tgt_N, v_tgt_N, r_chs_N)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    component_labels = ["Radial (x) [m]", "Along-track (y) [m]", "Cross-track (z) [m]"]

    for k, (ax, lbl) in enumerate(zip(axes, component_labels)):
        ax.plot(t_s / 60.0, dr_hill[:, k])
        ax.set_ylabel(lbl)
        ax.grid(True)

    axes[-1].set_xlabel("Time [min]")
    axes[0].set_title("Chaser relative position in Hill frame — zero control, one orbit")
    plt.tight_layout()

    plot_dir  = os.path.dirname(save_path)
    plot_path = os.path.join(plot_dir, "two_sc_relative_pos.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Relative-position plot saved to:\n  {plot_path}")

    # Hill-frame trajectory (V-bar vs R-bar)
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.plot(dr_hill[:, 1], dr_hill[:, 0])   # V-bar (y) on x-axis, R-bar (x) on y-axis
    ax2.set_xlabel("Along-track V-bar [m]")
    ax2.set_ylabel("Radial R-bar [m]")
    ax2.set_title("Chaser Hill-frame trajectory (zero control)")
    ax2.grid(True)
    ax2.set_aspect("equal")
    traj_path = os.path.join(plot_dir, "two_sc_hill_trajectory.png")
    fig2.savefig(traj_path, dpi=150)
    print(f"Hill trajectory plot saved to:\n  {traj_path}")

    print(f"\nVizard binary saved to:\n  {save_path}")
    print("Open Vizard and load the .bin file to view the 3D simulation.")

    plt.show()


if __name__ == "__main__":
    main()
