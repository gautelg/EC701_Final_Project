"""
octagon_inspection_comparison.py

Runs a shared octagonal inspection scenario with two controllers:
Case 1 waypoint-stop guidance and Case 2 continuous-motion guidance.

The experiment loads src/sim/sim_config.yaml as the base configuration and
overrides only the fields needed for this maneuver.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_EXPERIMENTS_DIR)
_SRC_DIR = os.path.dirname(_SIM_DIR)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib.pyplot as plt
import numpy as np
import yaml


G0_M_S2 = 9.80665


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
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_base_config():
    return _read_yaml(os.path.join(_SIM_DIR, "sim_config.yaml"))


def generate_octagon_waypoints(
    radius_m=20.0,
    num_points=8,
    z_m=0.0,
    start_angle_deg=-90.0,
):
    angles = np.deg2rad(start_angle_deg) + np.linspace(
        0.0, 2.0 * np.pi, num_points, endpoint=False
    )
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    z = np.full(num_points, z_m, dtype=float)
    return np.column_stack([x, y, z])


def generate_tangent_velocities(waypoints, period_s=600.0, clockwise=False):
    waypoints = np.asarray(waypoints, dtype=float)
    radii = np.linalg.norm(waypoints[:, :2], axis=1)
    valid_radii = radii[radii > 1e-9]
    radius_ref = float(np.mean(valid_radii)) if valid_radii.size else 0.0
    speed = 0.0 if radius_ref <= 0.0 else 2.0 * np.pi * radius_ref / float(period_s)

    tangents = np.zeros_like(waypoints)
    for i, wp in enumerate(waypoints):
        radial_xy = wp[:2]
        radius = np.linalg.norm(radial_xy)
        if radius <= 1e-9:
            continue

        if clockwise:
            tangent_xy = np.array([radial_xy[1], -radial_xy[0]], dtype=float) / radius
        else:
            tangent_xy = np.array([-radial_xy[1], radial_xy[0]], dtype=float) / radius

        tangents[i, :2] = speed * tangent_xy

    return tangents


def estimate_propellant_usage(force_history, dt_s, initial_mass_kg, isp_s, g0=G0_M_S2):
    force_history = np.asarray(force_history, dtype=float)
    dt_s = float(dt_s)
    initial_mass_kg = float(initial_mass_kg)
    isp_s = float(isp_s)

    if initial_mass_kg <= 0.0:
        raise ValueError("initial_mass_kg must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    if isp_s <= 0.0:
        raise ValueError("isp_s must be positive")

    mass_kg = initial_mass_kg
    total_impulse = 0.0
    total_delta_v = 0.0

    for force_vec in force_history:
        thrust_n = float(np.linalg.norm(force_vec))
        total_impulse += thrust_n * dt_s

        if thrust_n <= 0.0:
            continue

        delta_v = thrust_n * dt_s / mass_kg
        mass_kg *= np.exp(-delta_v / (isp_s * g0))
        total_delta_v += delta_v

    return {
        "total_impulse_n_s": total_impulse,
        "delta_v_m_s": total_delta_v,
        "propellant_used_kg": initial_mass_kg - mass_kg,
        "final_mass_kg": mass_kg,
    }


def _quat_to_dcm(q_scalar_first):
    q = np.asarray(q_scalar_first, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _line_of_sight_inertial(state):
    if state.rel_pos_inertial is not None:
        los_inertial = -np.asarray(state.rel_pos_inertial, dtype=float)
    elif state.hill_to_inertial_dcm is not None:
        los_inertial = -np.asarray(state.hill_to_inertial_dcm, dtype=float) @ state.rel_pos
    else:
        los_inertial = -np.asarray(state.rel_pos, dtype=float)
    return los_inertial


def _safe_unit(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return None
    return vec / norm


def compute_case1_boresight_metrics(state):
    from control.Case1.Case1_attitude_controller import (
        quat_conj as case1_quat_conj,
        rotate_vector_by_quaternion,
    )

    los_inertial = _safe_unit(_line_of_sight_inertial(state))
    if los_inertial is None:
        return 0.0, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    q = np.asarray(state.quaternion, dtype=float)
    # Case 1 uses an active body->inertial quaternion.
    q_body_to_inertial = np.array([q[3], -q[0], -q[1], -q[2]], dtype=float)
    los_body = rotate_vector_by_quaternion(case1_quat_conj(q_body_to_inertial), los_inertial)
    body_axis_body = np.array([1.0, 0.0, 0.0])
    cos_angle = np.clip(np.dot(los_body, body_axis_body), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle))), los_body, body_axis_body


def compute_case2_boresight_metrics(state):
    from control.Case2.Case2_Controller import _sim_state_to_case2_quaternion, rotate_vector

    los_hill = _safe_unit(-np.asarray(state.rel_pos, dtype=float))
    if los_hill is None:
        return 0.0, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])

    q_case2 = _sim_state_to_case2_quaternion(state)
    los_body = rotate_vector(q_case2, los_hill)
    body_axis_body = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(np.dot(los_body, body_axis_body), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle))), los_body, body_axis_body


def compute_boresight_metrics(controller_name, state):
    if controller_name == "Case1":
        return compute_case1_boresight_metrics(state)
    if controller_name == "Case2":
        return compute_case2_boresight_metrics(state)
    raise ValueError(f"Unknown controller_name {controller_name!r}")


def compute_boresight_overlay_vectors(controller_name, state):
    if controller_name == "Case1":
        from control.Case1.Case1_attitude_controller import rotate_vector_by_quaternion

        los_inertial = _safe_unit(_line_of_sight_inertial(state))
        if los_inertial is None:
            return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

        q = np.asarray(state.quaternion, dtype=float)
        q_body_to_inertial = np.array([q[3], -q[0], -q[1], -q[2]], dtype=float)
        boresight_inertial = rotate_vector_by_quaternion(
            q_body_to_inertial,
            np.array([1.0, 0.0, 0.0]),
        )
        if state.hill_to_inertial_dcm is not None:
            dcm_hn = np.asarray(state.hill_to_inertial_dcm, dtype=float).T
            return _safe_unit(dcm_hn @ boresight_inertial), _safe_unit(dcm_hn @ los_inertial)
        return _safe_unit(boresight_inertial), los_inertial

    if controller_name == "Case2":
        from control.Case2.Case2_Controller import (
            _sim_state_to_case2_quaternion,
            quat_conjugate,
            rotate_vector,
        )

        los_hill = _safe_unit(-np.asarray(state.rel_pos, dtype=float))
        if los_hill is None:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])

        q_case2 = _sim_state_to_case2_quaternion(state)
        boresight_hill = rotate_vector(quat_conjugate(q_case2), np.array([0.0, 0.0, 1.0]))
        return _safe_unit(boresight_hill), los_hill

    raise ValueError(f"Unknown controller_name {controller_name!r}")


def build_experiment_config(args=None):
    waypoints = generate_octagon_waypoints(radius_m=20.0, num_points=8, z_m=0.0)
    tangent_velocities = generate_tangent_velocities(waypoints, period_s=900.0)
    tangent_velocities[0] = np.zeros(3)

    output_root = os.path.join(_SIM_DIR, "output", "octagon_inspection")
    overrides = {
        "simulation": {
            "controller_dt": 2.0,
            "t_end": 7200.0,
        },
        "environment": {
            "gravity": {
                "enable_j2": True,
            },
        },
        "case1": {
            "t_end": 7200.0,
            "waypoints": waypoints.tolist(),
            "horizon": 40,
            "u_max": [0.01, 0.01, 0.01],
            "Q": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
            "R": [0.1, 0.1, 0.1],
            "P": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
            "use_cbf": True,
            "R_koz": 17.0,
            "cbf_k0": 1.0,
            "cbf_k1": 2.0,
            "cbf_rho": 10000.0,
            "cbf_use_slack": False,
            "Kq": [10.0, 10.0, 10.0],
            "Kw": [60.0, 60.0, 60.0],
            "tau_max": [0.5, 0.5, 0.5],
            "max_force": 5.0,
            "max_torque": 0.5,
            "mission": {
                "eps_r": 1.0,
                "eps_v": 0.05,
                "eps_boresight_deg": 5.0,
                "eps_w": 0.025,
                "required_count": 3,
            },
        },
        "case2": {
            "t_end": 7200.0,
            "horizon": 24,
            "u_max_thrust": 5.0,
            "tau_max": 0.5,
            "waypoints": waypoints.tolist(),
            "wp_tangent_vels": tangent_velocities.tolist(),
            "use_cbf": True,
            "R_koz": 17.0,
            "cbf_k0": 1.0,
            "cbf_k1": 2.0,
            "cbf_rho": 10000.0,
            "cbf_use_slack": False,
            "arrival_tol_pre": 2.0,
            "arrival_tol": 2.0,
            "pre_approach_waypoint_index": None,
            "keep_out_radius": 17.0,
            "max_force": 5.2,
            "max_torque": 0.5,
            "weights": {
                "Q_wp_pos": 80000.0,
                "Q_wp_vel": 30000.0,
                "Q_pointing": 2000.0,
                "Q_pointing_rate": 800.0,
                "R_thrust": 250.0,
                "Q_du": 250.0,
            },
            "verbose": False,
        },
        "inspection_experiment": {
            "output_root": output_root,
            "radius_m": 20.0,
            "num_points": 8,
            "inspection_period_s": 900.0,
            "clockwise": False,
        },
    }
    config = _deep_merge(load_base_config(), overrides)

    if args is not None:
        if args.t_end is not None:
            config["simulation"]["t_end"] = float(args.t_end)
            config["case1"]["t_end"] = float(args.t_end)
            config["case2"]["t_end"] = float(args.t_end)
        if args.case2_horizon is not None:
            config["case2"]["horizon"] = int(args.case2_horizon)
        if args.case2_verbose:
            config["case2"]["verbose"] = True

    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the octagon inspection comparison experiment.")
    parser.add_argument(
        "--backend",
        choices=["basilisk", "internal", "both"],
        default="both",
        help="Run the Basilisk sim backend, the controller-only backend, or both.",
    )
    parser.add_argument(
        "--controller",
        choices=["Case1", "Case2", "both"],
        default="both",
        help="Run only one controller or both.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on controller steps for quick debugging.",
    )
    parser.add_argument(
        "--step-log-every",
        type=int,
        default=50,
        help="How often to print progress lines.",
    )
    parser.add_argument(
        "--case2-horizon",
        type=int,
        default=None,
        help="Override the Case2 MPC horizon for debugging speed.",
    )
    parser.add_argument(
        "--case2-verbose",
        action="store_true",
        help="Enable Case2 warm-start/failure messages.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Override the simulation end time in seconds.",
    )
    return parser.parse_args(argv)


def _mean_motion(config):
    mu = 3.986004418e14
    return float(np.sqrt(mu / float(config["orbit"]["a"]) ** 3))


def _mrp_to_quat_scalar_last(sigma):
    from Basilisk.utilities import RigidBodyKinematics as rbk

    q_sf = np.asarray(rbk.MRP2EP(sigma), dtype=float)
    return np.array([q_sf[1], q_sf[2], q_sf[3], q_sf[0]], dtype=float)


def _quat_multiply_sf(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_normalize_sf(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else q


def _initial_internal_state(config):
    from sim.interface.sim_state import SimState

    chaser_cfg = config["chaser"]
    return SimState(
        time=0.0,
        rel_pos=np.asarray(chaser_cfg["offset_hill"], dtype=float),
        rel_vel=np.asarray(chaser_cfg["v_offset_hill"], dtype=float),
        quaternion=_mrp_to_quat_scalar_last(np.asarray(chaser_cfg["attitude_mrp"], dtype=float)),
        omega=np.asarray(chaser_cfg["omega_BN_B"], dtype=float),
        rel_pos_inertial=np.asarray(chaser_cfg["offset_hill"], dtype=float),
        hill_to_inertial_dcm=np.eye(3),
    )


def _propagate_internal_state(state, command, dt, mass, mean_motion, inertia_diag):
    from sim.interface.sim_state import SimState

    pos = np.asarray(state.rel_pos, dtype=float)
    vel = np.asarray(state.rel_vel, dtype=float)
    accel = np.asarray(command.force, dtype=float) / float(mass)

    dpos = vel
    dvel = np.array(
        [
            3.0 * mean_motion**2 * pos[0] + 2.0 * mean_motion * vel[1] + accel[0],
            -2.0 * mean_motion * vel[0] + accel[1],
            -mean_motion**2 * pos[2] + accel[2],
        ],
        dtype=float,
    )
    pos_next = pos + dt * dpos
    vel_next = vel + dt * dvel

    q_sf = np.array(
        [state.quaternion[3], state.quaternion[0], state.quaternion[1], state.quaternion[2]],
        dtype=float,
    )
    omega = np.asarray(state.omega, dtype=float)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
    q_dot = 0.5 * _quat_multiply_sf(q_sf, omega_quat)
    q_next = _quat_normalize_sf(q_sf + dt * q_dot)

    inertia = np.diag(np.asarray(inertia_diag, dtype=float))
    torque = np.asarray(command.torque, dtype=float)
    omega_dot = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
    omega_next = omega + dt * omega_dot

    return SimState(
        time=state.time + dt,
        rel_pos=pos_next,
        rel_vel=vel_next,
        quaternion=np.array([q_next[1], q_next[2], q_next[3], q_next[0]], dtype=float),
        omega=omega_next,
        rel_pos_inertial=pos_next.copy(),
        hill_to_inertial_dcm=np.eye(3),
    )


def _backend_output_dir(config, backend_name):
    root = config["inspection_experiment"]["output_root"]
    folder = "sim_dynamics" if backend_name == "basilisk" else "controller_only_dynamics"
    return os.path.join(root, folder)


def _case1_controller_config(config, env):
    case1_cfg = dict(config.get("case1", {}))
    sim_cfg = config["simulation"]
    sc_cfg = config["spacecraft"]
    orbit_cfg = config["orbit"]

    case1_cfg["mass"] = sc_cfg["mass"]
    case1_cfg["controller_dt"] = sim_cfg["controller_dt"]
    if env is not None:
        case1_cfg.setdefault(
            "mean_motion",
            np.sqrt(env.earth.mu / float(orbit_cfg["a"]) ** 3),
        )
    else:
        case1_cfg.setdefault("mean_motion", _mean_motion(config))
    return case1_cfg


def _case2_controller_config(config, env):
    case2_cfg = dict(config.get("case2", {}))
    sim_cfg = config["simulation"]
    sc_cfg = config["spacecraft"]
    orbit_cfg = config["orbit"]

    case2_cfg["mass"] = sc_cfg["mass"]
    case2_cfg["controller_dt"] = sim_cfg["controller_dt"]
    case2_cfg.setdefault("inertia_diag", sc_cfg["inertia_diag"])
    if env is not None:
        case2_cfg.setdefault(
            "mean_motion",
            np.sqrt(env.earth.mu / float(orbit_cfg["a"]) ** 3),
        )
    else:
        case2_cfg.setdefault("mean_motion", _mean_motion(config))
    return case2_cfg


def _finalize_history(history):
    results = {}
    for key, values in history.items():
        if key in {"mode"}:
            results[key] = np.asarray(values, dtype=object)
        else:
            results[key] = np.asarray(values)
    return results


def _write_summary_text(summary_path, controller_name, results):
    vizard_path = results.get("vizard_path", "N/A")
    lines = [
        f"Controller: {controller_name}",
        f"Backend: {results['backend_name']}",
        f"Completed: {results['completed']}",
        f"Simulation time [s]: {results['sim_time_s']:.1f}",
        f"Controller steps: {results['num_steps']}",
        f"Vizard binary: {vizard_path}",
        f"Waypoints reached: {results['waypoints_reached']}",
        f"Min distance to target [m]: {results['min_distance_m']:.3f}",
        f"Final distance to target [m]: {results['final_distance_m']:.3f}",
        f"Total impulse [N s]: {results['fuel']['total_impulse_n_s']:.3f}",
        f"Estimated delta-v [m/s]: {results['fuel']['delta_v_m_s']:.6f}",
        f"Estimated propellant used [kg]: {results['fuel']['propellant_used_kg']:.6f}",
        f"Estimated final mass [kg]: {results['fuel']['final_mass_kg']:.6f}",
        f"Mean boresight error [deg]: {results['mean_boresight_error_deg']:.3f}",
        f"Max boresight error [deg]: {results['max_boresight_error_deg']:.3f}",
        f"Mean body-rate norm [rad/s]: {results['mean_rate_error_norm']:.6f}",
        f"Mean command validity: {results['valid_fraction']:.6f}",
    ]
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _save_run_outputs(results, config, controller_name):
    output_dir = results["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.join(output_dir, f"octagon_{controller_name.lower()}")
    np.savez(
        f"{prefix}_history.npz",
        time=results["time"],
        rel_pos=results["rel_pos"],
        rel_vel=results["rel_vel"],
        quaternion=results["quaternion"],
        omega=results["omega"],
        force=results["force"],
        torque=results["torque"],
        valid=results["valid"],
        waypoint_index=results["waypoint_index"],
        mode=results["mode"],
        translate_counter=results["translate_counter"],
        rotate_counter=results["rotate_counter"],
        boresight_error_deg=results["boresight_error_deg"],
        controller_boresight_error_deg=results["controller_boresight_error_deg"],
        quaternion_error_norm=results["quaternion_error_norm"],
        rate_error_norm=results["rate_error_norm"],
        los_body=results["los_body"],
        boresight_hill=results["boresight_hill"],
        target_los_hill=results["target_los_hill"],
        desired_body_axis=results["desired_body_axis"],
        distance=results["distance"],
        rel_pos_inertial=results["rel_pos_inertial"],
        hill_to_inertial_dcm=results["hill_to_inertial_dcm"],
    )
    _write_summary_text(f"{prefix}_summary.txt", controller_name, results)

    if results["time"].size == 0:
        return

    waypoints = np.asarray(config[controller_name.lower()]["waypoints"], dtype=float)
    t_min = results["time"] / 60.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(results["rel_pos"][:, 1], results["rel_pos"][:, 0], lw=2, label=controller_name)
    ax.scatter(results["rel_pos"][0, 1], results["rel_pos"][0, 0], s=50, label="start")
    ax.scatter(0.0, 0.0, marker="x", s=80, label="target")
    ax.scatter(waypoints[:, 1], waypoints[:, 0], marker="s", s=45, label="waypoints")
    ax.set_xlabel("Along-track y [m]")
    ax.set_ylabel("Radial x [m]")
    ax.set_title(f"{controller_name} Octagon Inspection Trajectory")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{prefix}_trajectory.png", dpi=150)
    plt.close(fig)

    fig2, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(t_min, results["rel_pos"])
    axes[0].set_ylabel("Rel pos [m]")
    axes[0].legend(["x", "y", "z"])
    axes[0].grid(True)

    axes[1].plot(t_min, results["rel_vel"])
    axes[1].set_ylabel("Rel vel [m/s]")
    axes[1].legend(["vx", "vy", "vz"])
    axes[1].grid(True)

    axes[2].plot(t_min, results["quaternion"])
    axes[2].set_ylabel("Quaternion")
    axes[2].legend(["qx", "qy", "qz", "qw"])
    axes[2].grid(True)

    axes[3].plot(t_min, results["omega"])
    axes[3].set_ylabel("Omega [rad/s]")
    axes[3].legend(["wx", "wy", "wz"])
    axes[3].set_xlabel("Time [min]")
    axes[3].grid(True)

    fig2.tight_layout()
    fig2.savefig(f"{prefix}_timeseries.png", dpi=150)
    plt.close(fig2)

    fig3, axes3 = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes3[0].plot(t_min, results["boresight_error_deg"], label="boresight error")
    axes3[0].set_ylabel("Boresight [deg]")
    axes3[0].grid(True)
    axes3[0].legend()

    axes3[1].plot(t_min, results["los_body"])
    axes3[1].set_ylabel("LOS in body")
    axes3[1].legend(["x", "y", "z"])
    axes3[1].grid(True)
    desired_axis = np.asarray(results["desired_body_axis"], dtype=float)
    if desired_axis.shape == (3,):
        for idx, target_val in enumerate(desired_axis):
            axes3[1].axhline(target_val, linestyle="--", linewidth=1.0, alpha=0.6)

    axes3[2].plot(t_min, np.linalg.norm(results["force"], axis=1), label="|force|")
    axes3[2].set_ylabel("Force [N]")
    axes3[2].grid(True)
    axes3[2].legend()

    axes3[3].step(t_min, results["waypoint_index"], where="post", label="waypoint")
    axes3[3].set_ylabel("Waypoint")
    axes3[3].set_xlabel("Time [min]")
    axes3[3].grid(True)
    axes3[3].legend()

    fig3.tight_layout()
    fig3.savefig(f"{prefix}_metrics.png", dpi=150)
    plt.close(fig3)

    fig4, axes4 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    component_labels = ["x", "y", "z"]
    for idx, label in enumerate(component_labels):
        axes4[idx].plot(
            t_min,
            results["boresight_hill"][:, idx],
            linewidth=2,
            label=f"boresight {label}",
        )
        axes4[idx].plot(
            t_min,
            results["target_los_hill"][:, idx],
            linestyle="--",
            linewidth=2,
            label=f"target LOS {label}",
        )
        axes4[idx].set_ylabel(f"{label}-comp")
        axes4[idx].set_ylim(-1.05, 1.05)
        axes4[idx].grid(True)
        axes4[idx].legend()
    axes4[-1].set_xlabel("Time [min]")
    fig4.suptitle(f"{controller_name} Boresight vs Target LOS in Hill Frame")
    fig4.tight_layout()
    fig4.savefig(f"{prefix}_boresight_overlay.png", dpi=150)
    plt.close(fig4)


def _save_comparison_outputs(all_results, config):
    first_result = next(iter(all_results.values()))
    output_dir = first_result["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    case1 = all_results["Case1"]
    case2 = all_results["Case2"]
    waypoints = np.asarray(config["case1"]["waypoints"], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(11, 14))
    axes[0].plot(case1["rel_pos"][:, 1], case1["rel_pos"][:, 0], lw=2, label="Case1")
    axes[0].plot(case2["rel_pos"][:, 1], case2["rel_pos"][:, 0], lw=2, label="Case2")
    axes[0].scatter(waypoints[:, 1], waypoints[:, 0], marker="s", s=45, label="waypoints")
    axes[0].scatter(0.0, 0.0, marker="x", s=80, label="target")
    axes[0].set_xlabel("Along-track y [m]")
    axes[0].set_ylabel("Radial x [m]")
    axes[0].set_title("Trajectory Comparison")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    t1 = case1["time"] / 60.0
    t2 = case2["time"] / 60.0
    axes[1].plot(t1, case1["distance"], label="Case1")
    axes[1].plot(t2, case2["distance"], label="Case2")
    axes[1].set_ylabel("Range [m]")
    axes[1].set_title("Relative Distance")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t1, case1["boresight_error_deg"], label="Case1 (+x to target)")
    axes[2].plot(t2, case2["boresight_error_deg"], label="Case2 (+z to target)")
    axes[2].set_ylabel("Boresight [deg]")
    axes[2].set_title("Boresight-to-Target Error")
    axes[2].grid(True)
    axes[2].legend()

    fuel_values = [
        case1["fuel"]["propellant_used_kg"],
        case2["fuel"]["propellant_used_kg"],
    ]
    axes[3].bar(["Case1", "Case2"], fuel_values, width=0.5)
    axes[3].set_ylabel("Propellant used [kg]")
    axes[3].set_title("Estimated Translation Fuel Usage")
    axes[3].grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "octagon_comparison.png"), dpi=150)
    plt.close(fig)

    summary_path = os.path.join(output_dir, "octagon_comparison_summary.txt")
    lines = []
    for label in ["Case1", "Case2"]:
        result = all_results[label]
        lines.extend(
            [
                f"[{label}]",
                f"completed={result['completed']}",
                f"sim_time_s={result['sim_time_s']:.1f}",
                f"waypoints_reached={result['waypoints_reached']}",
                f"delta_v_m_s={result['fuel']['delta_v_m_s']:.6f}",
                f"propellant_used_kg={result['fuel']['propellant_used_kg']:.6f}",
                "",
            ]
        )
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _initialize_controller(controller_name, config, env=None):
    from sim.adapter.controller_adapter import ControllerAdapter
    sim_cfg = config["simulation"]

    if controller_name == "Case1":
        from control.Case1.case1_mission_controller import Case1MissionController

        controller_cfg = _case1_controller_config(config, env)
        controller = Case1MissionController(controller_cfg)
        adapter = ControllerAdapter(
            controller,
            {
                "max_force": controller_cfg.get("max_force", 5.0),
                "max_torque": controller_cfg.get("max_torque", 0.5),
            },
        )
        t_end = float(controller_cfg.get("t_end", sim_cfg["t_end"]))
    elif controller_name == "Case2":
        from control.Case2.Case2_Controller import Case2MissionController

        controller_cfg = _case2_controller_config(config, env)
        controller = Case2MissionController(controller_cfg)
        adapter = ControllerAdapter(
            controller,
            {
                "max_force": controller_cfg.get("max_force", 5.0),
                "max_torque": controller_cfg.get("max_torque", 0.5),
            },
        )
        t_end = float(controller_cfg.get("t_end", sim_cfg["t_end"]))
    else:
        raise ValueError(f"Unknown controller_name {controller_name!r}")

    return controller, adapter, controller_cfg, t_end


def _execute_run_loop(
    controller_name,
    backend_name,
    controller,
    adapter,
    config,
    t_end,
    state_reader,
    state_writer,
    step_advance,
    vizard_path,
    output_dir,
    max_steps=None,
    step_log_every=50,
):
    sim_cfg = config["simulation"]
    controller_dt = float(sim_cfg["controller_dt"])
    n_steps = int(round(t_end / controller_dt))
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))

    history = {
        "time": [],
        "rel_pos": [],
        "rel_vel": [],
        "quaternion": [],
        "omega": [],
        "force": [],
        "torque": [],
        "valid": [],
        "waypoint_index": [],
        "mode": [],
        "translate_counter": [],
        "rotate_counter": [],
        "boresight_error_deg": [],
        "controller_boresight_error_deg": [],
        "quaternion_error_norm": [],
        "rate_error_norm": [],
        "los_body": [],
        "boresight_hill": [],
        "target_los_hill": [],
        "distance": [],
        "rel_pos_inertial": [],
        "hill_to_inertial_dcm": [],
    }

    print(f"Running {controller_name} octagon inspection [{backend_name}]...")
    t_current = 0.0
    desired_body_axis = np.array([1.0, 0.0, 0.0]) if controller_name == "Case1" else np.array([0.0, 0.0, 1.0])
    for step in range(n_steps):
        state = state_reader()
        should_log = (step % max(1, step_log_every) == 0)

        if should_log:
            print(
                f"  starting step={step:04d} t={state.time:7.1f}s "
                f"wp={getattr(controller.status(), 'waypoint_index', -1)} "
                f"|r|={np.linalg.norm(state.rel_pos):6.2f}m"
            )

        step_wall_start = time.perf_counter()
        cmd = adapter.step(state)
        step_wall_seconds = time.perf_counter() - step_wall_start
        state_writer(cmd)
        status = controller.status()
        boresight_error_deg, los_body, desired_body_axis = compute_boresight_metrics(
            controller_name,
            state,
        )
        boresight_hill, target_los_hill = compute_boresight_overlay_vectors(
            controller_name,
            state,
        )

        history["time"].append(state.time)
        history["rel_pos"].append(state.rel_pos.copy())
        history["rel_vel"].append(state.rel_vel.copy())
        history["quaternion"].append(state.quaternion.copy())
        history["omega"].append(state.omega.copy())
        history["force"].append(cmd.force.copy())
        history["torque"].append(cmd.torque.copy())
        history["valid"].append(cmd.valid)
        history["waypoint_index"].append(getattr(status, "waypoint_index", -1))
        history["mode"].append(getattr(status, "mode", "TRACK"))
        history["translate_counter"].append(getattr(status, "translate_counter", -1))
        history["rotate_counter"].append(getattr(status, "rotate_counter", -1))
        history["boresight_error_deg"].append(boresight_error_deg)
        history["controller_boresight_error_deg"].append(
            getattr(status, "boresight_error_deg", getattr(status, "pointing_error_deg", np.nan))
        )
        history["quaternion_error_norm"].append(getattr(status, "quaternion_error_norm", np.nan))
        history["rate_error_norm"].append(
            getattr(status, "rate_error_norm", float(np.linalg.norm(state.omega)))
        )
        history["los_body"].append(los_body.copy())
        history["boresight_hill"].append(boresight_hill.copy())
        history["target_los_hill"].append(target_los_hill.copy())
        history["distance"].append(np.linalg.norm(state.rel_pos))
        rel_pos_inertial = (
            np.asarray(state.rel_pos_inertial, dtype=float).copy()
            if state.rel_pos_inertial is not None
            else np.full(3, np.nan)
        )
        hill_to_inertial_dcm = (
            np.asarray(state.hill_to_inertial_dcm, dtype=float).copy()
            if state.hill_to_inertial_dcm is not None
            else np.full((3, 3), np.nan)
        )
        history["rel_pos_inertial"].append(rel_pos_inertial)
        history["hill_to_inertial_dcm"].append(hill_to_inertial_dcm)

        if should_log:
            extra = ""
            if controller_name == "Case1":
                extra = (
                    f" mode={getattr(status, 'mode', 'n/a')}"
                    f" tc={getattr(status, 'translate_counter', -1)}"
                    f" rc={getattr(status, 'rotate_counter', -1)}"
                    f" bore={getattr(status, 'boresight_error_deg', np.nan):5.2f}deg"
                    f" rate={getattr(status, 'rate_error_norm', np.nan):.4f}"
                )
            if controller_name == "Case2":
                extra = (
                    f" solve={getattr(status, 'last_solve_seconds', np.nan):6.2f}s"
                    f" status={getattr(status, 'last_solver_status', 'n/a')}"
                    f" iter={getattr(status, 'last_iter_count', None)}"
                )
            print(
                f"  finished step={step:04d} wall={step_wall_seconds:6.2f}s "
                f"wp={getattr(status, 'waypoint_index', -1)} "
                f"|r|={np.linalg.norm(state.rel_pos):6.2f}m "
                f"|F|={np.linalg.norm(cmd.force):5.2f}N "
                f"valid={cmd.valid}{extra}"
            )

        t_current += controller_dt
        step_advance(t_current)

        if getattr(controller, "done", False):
            print(f"  {controller_name} completed at t={t_current:.1f}s")
            break

    results = _finalize_history(history)
    fuel = estimate_propellant_usage(
        results["force"],
        controller_dt,
        config["spacecraft"]["mass"],
        config["spacecraft"]["isp"],
    )
    results["fuel"] = fuel
    results["vizard_path"] = vizard_path
    results["output_dir"] = output_dir
    results["backend_name"] = backend_name
    results["desired_body_axis"] = desired_body_axis.copy()
    results["completed"] = bool(getattr(controller, "done", False))
    results["sim_time_s"] = float(t_current)
    results["num_steps"] = int(results["time"].size)
    results["waypoints_reached"] = (
        int(results["waypoint_index"][-1]) if results["waypoint_index"].size else 0
    )
    results["min_distance_m"] = (
        float(np.min(results["distance"])) if results["distance"].size else np.nan
    )
    results["final_distance_m"] = (
        float(results["distance"][-1]) if results["distance"].size else np.nan
    )
    results["mean_boresight_error_deg"] = (
        float(np.mean(results["boresight_error_deg"])) if results["boresight_error_deg"].size else np.nan
    )
    results["max_boresight_error_deg"] = (
        float(np.max(results["boresight_error_deg"])) if results["boresight_error_deg"].size else np.nan
    )
    results["mean_rate_error_norm"] = (
        float(np.nanmean(results["rate_error_norm"])) if results["rate_error_norm"].size else np.nan
    )
    results["valid_fraction"] = (
        float(np.mean(results["valid"])) if results["valid"].size else np.nan
    )

    _save_run_outputs(results, config, controller_name)
    return results


def run_controller_case(
    controller_name,
    config,
    backend_name,
    max_steps=None,
    step_log_every=50,
):
    sim_cfg = config["simulation"]
    output_dir = _backend_output_dir(config, backend_name)
    os.makedirs(output_dir, exist_ok=True)

    if backend_name == "basilisk":
        from Basilisk.utilities import vizSupport
        from sim.core.bsk_environment import BskEnvironment
        from sim.core.bsk_interface import BskInterface
        from sim.core.bsk_sim import BskSim
        from sim.core.bsk_spacecraft import BskSpacecraft

        bsk = BskSim(sim_cfg["dynamics_dt"], sim_cfg["controller_dt"])
        bsk.setup()

        env = BskEnvironment(bsk, config)
        env.setup()

        sc = BskSpacecraft(bsk, config)
        sc.setup(env)
        env.attach_perturbation_models(sc)
        iface = BskInterface(sc, env)

        vizard_path = os.path.join(output_dir, f"octagon_{controller_name.lower()}.bin")
        vizSupport.enableUnityVisualization(
            bsk.scSim,
            bsk.taskName,
            [sc.scTarget, sc.scChaser],
            saveFile=vizard_path,
        )
        controller, adapter, _controller_cfg, t_end = _initialize_controller(controller_name, config, env)
        bsk.initialize()

        return _execute_run_loop(
            controller_name,
            backend_name,
            controller,
            adapter,
            config,
            t_end,
            state_reader=iface.read_state,
            state_writer=iface.write_command,
            step_advance=bsk.run,
            vizard_path=vizard_path,
            output_dir=output_dir,
            max_steps=max_steps,
            step_log_every=step_log_every,
        )

    if backend_name == "internal":
        controller, adapter, _controller_cfg, t_end = _initialize_controller(controller_name, config, env=None)
        state = _initial_internal_state(config)
        mass = float(config["spacecraft"]["mass"])
        mean_motion = _mean_motion(config)
        inertia_diag = np.asarray(config["spacecraft"]["inertia_diag"], dtype=float)

        def _read_state():
            return state

        def _write_command(_command):
            return None

        def _advance(_t_current):
            nonlocal state, _last_command
            state = _propagate_internal_state(
                state,
                _last_command,
                float(sim_cfg["controller_dt"]),
                mass,
                mean_motion,
                inertia_diag,
            )

        _last_command = None

        def _state_writer(command):
            nonlocal _last_command
            _last_command = command

        return _execute_run_loop(
            controller_name,
            backend_name,
            controller,
            adapter,
            config,
            t_end,
            state_reader=_read_state,
            state_writer=_state_writer,
            step_advance=_advance,
            vizard_path="N/A",
            output_dir=output_dir,
            max_steps=max_steps,
            step_log_every=step_log_every,
        )

    raise ValueError(f"Unknown backend_name {backend_name!r}")


def main(argv=None):
    args = parse_args(argv)
    config = build_experiment_config(args)

    controller_names = ["Case1", "Case2"] if args.controller == "both" else [args.controller]
    backend_names = ["basilisk", "internal"] if args.backend == "both" else [args.backend]

    all_results = {}
    for backend_name in backend_names:
        backend_results = {}
        for controller_name in controller_names:
            backend_results[controller_name] = run_controller_case(
                controller_name,
                config,
                backend_name=backend_name,
                max_steps=args.max_steps,
                step_log_every=args.step_log_every,
            )

        if set(backend_results) == {"Case1", "Case2"}:
            _save_comparison_outputs(backend_results, config)

        all_results[backend_name] = backend_results
        output_dir = _backend_output_dir(config, backend_name)
        print(f"Saved {backend_name} octagon inspection outputs to: {output_dir}")
        for label, result in backend_results.items():
            print(
                f"{backend_name}/{label}: completed={result['completed']} "
                f"propellant={result['fuel']['propellant_used_kg']:.6f} kg "
                f"delta_v={result['fuel']['delta_v_m_s']:.6f} m/s"
            )


if __name__ == "__main__":
    main()
