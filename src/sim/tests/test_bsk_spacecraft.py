"""
Unit tests for Basilisk spacecraft initialization helpers.
"""

import os
import sys

import numpy as np
import pytest


pytest.importorskip("Basilisk")


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_TESTS_DIR)
_SRC_DIR = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sim.core.bsk_spacecraft import (  # noqa: E402
    _hill_to_inertial_dcm,
    _hill_velocity_to_inertial_delta,
)


def test_hill_velocity_to_inertial_delta_inverts_rotating_frame_relation():
    r_tgt = np.array([6778136.0, 0.0, 0.0])
    mu = 3.986004418e14
    n = np.sqrt(mu / np.linalg.norm(r_tgt) ** 3)
    v_tgt = np.array([0.0, np.sqrt(mu / np.linalg.norm(r_tgt)), 0.0])

    dr_hill = np.array([25.0, -100.0, 10.0])
    dv_hill = np.array([0.03, -0.02, 0.01])

    R_NH = _hill_to_inertial_dcm(r_tgt, v_tgt)
    inertial_delta_v = _hill_velocity_to_inertial_delta(r_tgt, v_tgt, dr_hill, dv_hill)

    recovered_rel_vel = (
        R_NH.T @ inertial_delta_v
        - np.cross(np.array([0.0, 0.0, n]), dr_hill)
    )

    np.testing.assert_allclose(recovered_rel_vel, dv_hill, atol=1e-12, rtol=0.0)
