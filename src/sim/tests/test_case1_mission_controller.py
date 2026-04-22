"""
Smoke tests for the Case 1 mission controller adapter.
"""

import os
import sys

import numpy as np


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_TESTS_DIR)
_SRC_DIR = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim.interface.sim_state import SimState
from control.Case1.case1_mission_controller import Case1MissionController


def _controller_config():
    return {
        "mass": 500.0,
        "mean_motion": 0.0011,
        "controller_dt": 10.0,
        "horizon": 3,
        "u_max": [0.01, 0.01, 0.01],
        "Q": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0],
        "R": [0.1, 0.1, 0.1],
        "P": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
        "use_cbf": True,
        "R_koz": 15.0,
        "Kq": [1.0, 1.0, 1.0],
        "Kw": [6.0, 6.0, 6.0],
        "tau_max": [0.05, 0.05, 0.05],
        "waypoints": [[0.0, -50.0, 0.0]],
        "mission": {
            "eps_r": 2.0,
            "eps_v": 0.05,
            "eps_q": 0.05,
            "eps_w": 0.01,
            "required_count": 1,
        },
    }


def test_case1_controller_returns_finite_command():
    controller = Case1MissionController(_controller_config())
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
    assert np.linalg.norm(cmd.force, ord=np.inf) <= 5.0 + 1e-8
    assert np.allclose(cmd.torque, np.zeros(3))
    assert controller.status().mode == "TRANSLATE"
