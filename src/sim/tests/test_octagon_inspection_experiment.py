import os
import sys

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_TESTS_DIR)
_SRC_DIR = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim.experiments.octagon_inspection_comparison import (
    build_experiment_config,
    compute_boresight_overlay_vectors,
    compute_case1_boresight_metrics,
    compute_case2_boresight_metrics,
    estimate_propellant_usage,
    generate_octagon_waypoints,
    generate_tangent_velocities,
)
from sim.interface.sim_state import SimState


def test_generate_octagon_waypoints_starts_at_negative_v_bar():
    waypoints = generate_octagon_waypoints(radius_m=20.0, num_points=8)

    assert waypoints.shape == (8, 3)
    np.testing.assert_allclose(waypoints[0], [0.0, -20.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(waypoints[:, :2], axis=1), 20.0)


def test_generate_tangent_velocities_are_perpendicular_to_radius():
    waypoints = generate_octagon_waypoints(radius_m=20.0, num_points=8)
    tangent_vels = generate_tangent_velocities(waypoints, period_s=600.0)

    assert tangent_vels.shape == waypoints.shape
    for wp, vel in zip(waypoints, tangent_vels, strict=True):
        assert np.dot(wp[:2], vel[:2]) == pytest.approx(0.0, abs=1e-12)

    expected_speed = 2.0 * np.pi * 20.0 / 600.0
    np.testing.assert_allclose(np.linalg.norm(tangent_vels[:, :2], axis=1), expected_speed)
    np.testing.assert_allclose(tangent_vels[0], [expected_speed, 0.0, 0.0], atol=1e-12)


def test_estimate_propellant_usage_matches_closed_form_for_constant_force():
    force_history = np.tile(np.array([[5.0, 0.0, 0.0]]), (10, 1))
    result = estimate_propellant_usage(
        force_history,
        dt_s=2.0,
        initial_mass_kg=500.0,
        isp_s=225.0,
    )

    expected_delta_v = 0.0
    expected_final_mass = 500.0
    for _ in range(10):
        step_delta_v = 5.0 * 2.0 / expected_final_mass
        expected_final_mass *= np.exp(-step_delta_v / (225.0 * 9.80665))
        expected_delta_v += step_delta_v

    assert result["total_impulse_n_s"] == pytest.approx(100.0)
    assert result["delta_v_m_s"] == pytest.approx(expected_delta_v)
    assert result["final_mass_kg"] == pytest.approx(expected_final_mass)
    assert result["propellant_used_kg"] == pytest.approx(500.0 - expected_final_mass)


def test_experiment_config_applies_case2_tuning_defaults():
    config = build_experiment_config()

    assert config["environment"]["gravity"]["enable_j2"] is True
    assert config["inspection_experiment"]["output_root"].endswith("octagon_inspection")
    assert config["simulation"]["controller_dt"] == pytest.approx(2.0)
    assert config["case1"]["horizon"] == 40
    assert config["case1"]["R_koz"] == pytest.approx(17.0)
    assert config["case1"]["cbf_use_slack"] is False
    assert config["case1"]["tau_max"] == pytest.approx([0.5, 0.5, 0.5])
    assert config["case1"]["mission"]["eps_boresight_deg"] == pytest.approx(5.0)
    assert config["case1"]["mission"]["eps_w"] == pytest.approx(0.025)
    assert config["case2"]["horizon"] == 24
    assert config["case2"]["R_koz"] == pytest.approx(17.0)
    assert config["case2"]["cbf_use_slack"] is False
    assert config["case2"]["keep_out_radius"] == pytest.approx(17.0)
    assert config["case2"]["max_force"] == pytest.approx(5.2)
    assert config["case2"]["max_torque"] == pytest.approx(0.5)
    np.testing.assert_allclose(config["case2"]["wp_tangent_vels"][0], [0.0, 0.0, 0.0])
    expected_speed = 2.0 * np.pi * 20.0 / 900.0
    assert np.linalg.norm(config["case2"]["wp_tangent_vels"][1][:2]) == pytest.approx(expected_speed)
    assert config["case2"]["weights"]["Q_wp_pos"] == pytest.approx(80000.0)
    assert config["case2"]["weights"]["Q_wp_vel"] == pytest.approx(30000.0)
    assert config["case2"]["weights"]["Q_pointing"] == pytest.approx(2000.0)
    assert config["case2"]["weights"]["Q_pointing_rate"] == pytest.approx(800.0)
    assert config["case2"]["weights"]["R_thrust"] == pytest.approx(250.0)
    assert config["case2"]["weights"]["Q_du"] == pytest.approx(250.0)


def test_case1_boresight_metric_is_zero_when_body_x_points_to_target():
    state = SimState(
        time=0.0,
        rel_pos=np.array([-1.0, 0.0, 0.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([-1.0, 0.0, 0.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    error_deg, los_body, desired_axis = compute_case1_boresight_metrics(state)

    assert error_deg == pytest.approx(0.0)
    np.testing.assert_allclose(los_body, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(desired_axis, [1.0, 0.0, 0.0], atol=1e-12)


def test_case2_boresight_metric_is_zero_when_body_z_points_to_target():
    state = SimState(
        time=0.0,
        rel_pos=np.array([0.0, 0.0, -1.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([0.0, 0.0, -1.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    error_deg, los_body, desired_axis = compute_case2_boresight_metrics(state)

    assert error_deg == pytest.approx(0.0)
    np.testing.assert_allclose(los_body, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(desired_axis, [0.0, 0.0, 1.0], atol=1e-12)


def test_case1_overlay_vectors_match_when_perfectly_pointed():
    state = SimState(
        time=0.0,
        rel_pos=np.array([-1.0, 0.0, 0.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([-1.0, 0.0, 0.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    boresight_hill, target_los_hill = compute_boresight_overlay_vectors("Case1", state)

    np.testing.assert_allclose(boresight_hill, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(target_los_hill, [1.0, 0.0, 0.0], atol=1e-12)


def test_case2_overlay_vectors_match_when_perfectly_pointed():
    state = SimState(
        time=0.0,
        rel_pos=np.array([0.0, 0.0, -1.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([0.0, 0.0, -1.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    boresight_hill, target_los_hill = compute_boresight_overlay_vectors("Case2", state)

    np.testing.assert_allclose(boresight_hill, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(target_los_hill, [0.0, 0.0, 1.0], atol=1e-12)
