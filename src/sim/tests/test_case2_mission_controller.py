"""
Smoke tests for the Case 2 mission controller adapter.
"""

import os
import sys

import numpy as np
import pytest


pytest.importorskip("casadi")


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_TESTS_DIR)
_SRC_DIR = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim.interface.sim_state import SimState
from control.Case2.Case2_Controller import Case2MissionController


def _controller_config():
    return {
        "mass": 500.0,
        "mean_motion": 0.0011,
        "controller_dt": 4.0,
        "horizon": 3,
        "inertia_diag": [100.0, 100.0, 100.0],
        "u_max_thrust": 5.0,
        "tau_max": 0.8,
        "waypoints": [[0.0, -50.0, 0.0]],
        "wp_tangent_vels": [[0.0, 0.0, 0.0]],
        "arrival_tol_pre": 3.0,
        "arrival_tol": 3.0,
        "verbose": False,
    }


def test_case2_controller_returns_finite_command():
    controller = Case2MissionController(_controller_config())
    state = SimState(
        time=0.0,
        rel_pos=np.array([0.0, -100.0, 0.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([0.0, -100.0, 0.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    cmd = controller.step(state)

    assert cmd.force.shape == (3,)
    assert cmd.torque.shape == (3,)
    assert np.all(np.isfinite(cmd.force))
    assert np.all(np.isfinite(cmd.torque))
    assert np.linalg.norm(cmd.torque, ord=np.inf) <= 0.8 + 1e-8
    assert isinstance(controller.status().waypoint_index, int)


def test_case2_waypoint_switch_uses_measured_state_not_prediction(monkeypatch):
    config = _controller_config()
    config["waypoints"] = [[10.0, 0.0, 0.0]]
    config["wp_tangent_vels"] = [[0.0, 0.0, 0.0]]
    config["pre_approach_waypoint_index"] = None
    controller = Case2MissionController(config)
    state = SimState(
        time=0.0,
        rel_pos=np.array([0.0, -100.0, 0.0]),
        rel_vel=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        omega=np.zeros(3),
        rel_pos_inertial=np.array([0.0, -100.0, 0.0]),
        hill_to_inertial_dcm=np.eye(3),
    )

    predicted_arrival = np.zeros((13, 2))
    predicted_arrival[0:3, 1] = np.array([10.0, 0.0, 0.0])

    def _fake_solve(*_args, **_kwargs):
        return (
            np.zeros(6),
            np.zeros((6, controller.N_leg)),
            predicted_arrival,
            True,
            True,
            {
                "solve_seconds": 0.0,
                "solver_status": "mocked",
                "iter_count": 0,
            },
        )

    monkeypatch.setattr(
        "control.Case2.Case2_Controller._solve_case2_step",
        _fake_solve,
    )

    controller.step(state)

    assert controller.status().waypoint_index == 0
