"""
sim_runner.py

Main entry point for the Case 1 closed-loop mission.
"""

import os
import sys

_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SIM_DIR)

# When this file is launched as "python src/sim/sim_runner.py", Python puts
# src/sim on sys.path before the standard library. That makes src/sim/logging
# shadow stdlib logging during third-party imports such as matplotlib/Pillow.
if _SIM_DIR in sys.path:
    sys.path.remove(_SIM_DIR)

import matplotlib.pyplot as plt
import numpy as np
import yaml

for _path in [_SRC_DIR, _SIM_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from Basilisk.utilities import vizSupport

from adapter.controller_adapter import ControllerAdapter
from core.bsk_environment import BskEnvironment
from core.bsk_interface import BskInterface
from core.bsk_sim import BskSim
from core.bsk_spacecraft import BskSpacecraft
from control.Case1.case1_mission_controller import Case1MissionController


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _load_config(scenario_path=None):
    base_path = os.path.join(_SIM_DIR, "sim_config.yaml")
    config = _read_yaml(base_path)

    if scenario_path is None:
        scenario_path = os.path.join(_SIM_DIR, "scenarios", "case1.yaml")

    if scenario_path:
        if not os.path.isabs(scenario_path):
            scenario_path = os.path.join(_SIM_DIR, scenario_path)
        config = _deep_merge(config, _read_yaml(scenario_path))

    return config


def _case1_controller_config(config, env):
    case1_cfg = dict(config.get("case1", {}))
    sim_cfg = config["simulation"]
    sc_cfg = config["spacecraft"]
    orbit_cfg = config["orbit"]

    case1_cfg["mass"] = sc_cfg["mass"]
    case1_cfg["controller_dt"] = sim_cfg["controller_dt"]
    case1_cfg.setdefault(
        "mean_motion",
        np.sqrt(env.earth.mu / float(orbit_cfg["a"]) ** 3),
    )
    return case1_cfg


def run_case1_mission(scenario_path=None):
    config = _load_config(scenario_path)
    sim_cfg = config["simulation"]
    case1_cfg = config.get("case1", {})

    bsk = BskSim(sim_cfg["dynamics_dt"], sim_cfg["controller_dt"])
    bsk.setup()

    env = BskEnvironment(bsk, config)
    env.setup()

    sc = BskSpacecraft(bsk, config)
    sc.setup(env)

    controller = Case1MissionController(_case1_controller_config(config, env))
    adapter = ControllerAdapter(
        controller,
        {
            "max_force": case1_cfg.get("max_force", 50.0),
            "max_torque": case1_cfg.get("max_torque", 10.0),
        },
    )
    iface = BskInterface(sc, env)

    save_path = config.get("vizard", {}).get("save_file", "src/sim/output/case1_mission.bin")
    if not os.path.isabs(save_path):
        repo_root = os.path.dirname(_SRC_DIR)
        save_path = os.path.join(repo_root, save_path)
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    vizSupport.enableUnityVisualization(
        bsk.scSim,
        bsk.taskName,
        [sc.scTarget, sc.scChaser],
        saveFile=save_path,
    )

    bsk.initialize()

    t_end = float(case1_cfg.get("t_end", sim_cfg["t_end"]))
    controller_dt = float(sim_cfg["controller_dt"])
    n_steps = int(round(t_end / controller_dt))

    history = {
        "time": [],
        "rel_pos": [],
        "rel_vel": [],
        "quaternion": [],
        "omega": [],
        "attitude_error": [],
        "force": [],
        "torque": [],
        "valid": [],
        "mode": [],
        "waypoint_index": [],
    }

    t_current = 0.0
    initial_state = iface.read_state()
    print("Initialised Case 1 mission.")
    print(f"  Initial relative position: {initial_state.rel_pos}")
    print(f"  Controller steps: {n_steps} at dt={controller_dt:.3f} s")

    for k in range(n_steps):
        state = iface.read_state()
        cmd = adapter.step(state)
        iface.write_command(cmd)
        status = controller.status()

        history["time"].append(state.time)
        history["rel_pos"].append(state.rel_pos.copy())
        history["rel_vel"].append(state.rel_vel.copy())
        history["quaternion"].append(state.quaternion.copy())
        history["omega"].append(state.omega.copy())
        history["attitude_error"].append(controller.last_q_err.copy())
        history["force"].append(cmd.force.copy())
        history["torque"].append(cmd.torque.copy())
        history["valid"].append(cmd.valid)
        history["mode"].append(status.mode)
        history["waypoint_index"].append(status.waypoint_index)

        if k % 25 == 0:
            dist = np.linalg.norm(state.rel_pos)
            wp = status.current_waypoint
            wp_err = np.nan if wp is None else np.linalg.norm(state.rel_pos - wp)
            print(
                f"  step={k:04d} t={state.time:8.1f}s "
                f"mode={status.mode:9s} wp={status.waypoint_index} "
                f"|r|={dist:7.2f}m wp_err={wp_err:7.2f}m"
            )

        t_current += controller_dt
        bsk.run(t_current)

        if controller.done:
            print(f"Mission complete at t={t_current:.1f} s.")
            break

    final_state = iface.read_state()
    print(f"Final relative position: {final_state.rel_pos}")
    print(f"Final relative velocity: {final_state.rel_vel}")

    results = _finalize_history(history)
    results["vizard_path"] = save_path
    results["output_dir"] = output_dir

    _save_outputs(results, case1_cfg)
    print(f"Vizard binary: {save_path}")
    return results


def _finalize_history(history):
    results = {}
    for key, values in history.items():
        if key in {"mode"}:
            results[key] = np.asarray(values, dtype=object)
        else:
            results[key] = np.asarray(values)
    return results


def _save_outputs(results, case1_cfg):
    output_dir = results["output_dir"]
    npz_path = os.path.join(output_dir, "case1_mission_history.npz")
    np.savez(
        npz_path,
        time=results["time"],
        rel_pos=results["rel_pos"],
        rel_vel=results["rel_vel"],
        quaternion=results["quaternion"],
        omega=results["omega"],
        attitude_error=results["attitude_error"],
        force=results["force"],
        torque=results["torque"],
        valid=results["valid"],
        mode=results["mode"],
        waypoint_index=results["waypoint_index"],
    )

    if results["time"].size == 0:
        return

    t_min = results["time"] / 60.0
    rel_pos = results["rel_pos"]
    omega = results["omega"]
    attitude_error = results["attitude_error"]
    force = results["force"]
    torque = results["torque"]
    waypoints = np.asarray(case1_cfg.get("waypoints", []), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rel_pos[:, 1], rel_pos[:, 0], label="trajectory")
    ax.scatter(rel_pos[0, 1], rel_pos[0, 0], marker="o", label="start")
    ax.scatter(0.0, 0.0, marker="x", s=80, label="target")
    if waypoints.size:
        ax.scatter(waypoints[:, 1], waypoints[:, 0], marker="s", label="waypoints")
    ax.set_xlabel("Along-track y [m]")
    ax.set_ylabel("Radial x [m]")
    ax.set_title("Case 1 mission trajectory in Hill frame")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "case1_mission_trajectory.png"), dpi=150)
    plt.close(fig)

    fig2, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    axes[0].plot(t_min, rel_pos)
    axes[0].set_ylabel("Relative position [m]")
    axes[0].legend(["x", "y", "z"])
    axes[0].grid(True)

    axes[1].plot(t_min, attitude_error)
    axes[1].set_ylabel("Quaternion error")
    axes[1].legend(["q0", "q1", "q2", "q3"])
    axes[1].grid(True)

    axes[2].plot(t_min, omega)
    axes[2].set_ylabel("Body rate [rad/s]")
    axes[2].legend(["wx", "wy", "wz"])
    axes[2].grid(True)

    axes[3].plot(t_min, torque)
    axes[3].set_ylabel("Torque [N*m]")
    axes[3].legend(["tx", "ty", "tz"])
    axes[3].grid(True)

    axes[4].plot(t_min, np.linalg.norm(force, axis=1), label="|force|")
    axes[4].plot(t_min, np.linalg.norm(torque, axis=1), label="|torque|")
    axes[4].set_xlabel("Time [min]")
    axes[4].set_ylabel("Command norm")
    axes[4].legend()
    axes[4].grid(True)

    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "case1_mission_timeseries.png"), dpi=150)
    plt.close(fig2)

    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes3[0].plot(t_min, np.linalg.norm(attitude_error[:, 1:], axis=1))
    axes3[0].set_ylabel("|q_err vec|")
    axes3[0].grid(True)
    axes3[1].plot(t_min, np.linalg.norm(omega, axis=1))
    axes3[1].set_ylabel("|omega| [rad/s]")
    axes3[1].grid(True)
    axes3[2].plot(t_min, np.linalg.norm(torque, axis=1))
    axes3[2].set_xlabel("Time [min]")
    axes3[2].set_ylabel("|torque| [N*m]")
    axes3[2].grid(True)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "case1_mission_attitude.png"), dpi=150)
    plt.close(fig3)

    mode_values = np.array([0 if mode == "TRANSLATE" else 1 for mode in results["mode"]])
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.step(t_min, mode_values, where="post", label="mode")
    ax4.step(t_min, results["waypoint_index"], where="post", label="waypoint")
    ax4.set_xlabel("Time [min]")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["TRANSLATE", "ROTATE"])
    ax4.set_title("Case 1 mission mode transitions")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "case1_mission_modes.png"), dpi=150)
    plt.close(fig4)


def main():
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_case1_mission(scenario_path)


if __name__ == "__main__":
    main()
