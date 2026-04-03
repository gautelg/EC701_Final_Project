"""
tests/test_pd_controller.py

Unit tests for PDController.

No Basilisk dependency. Tests the PD control law directly using SimState
objects with known values and checks ControlCommand outputs analytically.

How to run (from repo root)
---------------------------
    python -m pytest src/control/tests/test_pd_controller.py -v
"""

import os
import sys

import numpy as np
import pytest

_CONTROL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/control/
_SRC_DIR     = os.path.dirname(_CONTROL_DIR)                                 # src/
for p in [_SRC_DIR, os.path.join(_SRC_DIR, "sim")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from control.pd_controller import PDController
from interface.sim_state import SimState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(rel_pos, rel_vel):
    """Build a minimal SimState with the given Hill-frame pos/vel."""
    return SimState(
        time       = 0.0,
        rel_pos    = np.array(rel_pos, dtype=float),
        rel_vel    = np.array(rel_vel, dtype=float),
        quaternion = np.array([0.0, 0.0, 0.0, 1.0]),
        omega      = np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPDControllerAtEquilibrium:
    """At the desired position with zero velocity the force should be zero."""

    def test_origin_desired_at_origin(self):
        ctrl = PDController(Kp=0.05, Kd=10.0)
        cmd  = ctrl.step(_make_state([0, 0, 0], [0, 0, 0]))
        np.testing.assert_allclose(cmd.force, [0, 0, 0], atol=1e-12)

    def test_nonzero_desired_at_desired(self):
        desired = [10.0, -5.0, 3.0]
        ctrl    = PDController(Kp=0.05, Kd=10.0, desired_rel_pos=desired)
        cmd     = ctrl.step(_make_state(desired, [0, 0, 0]))
        np.testing.assert_allclose(cmd.force, [0, 0, 0], atol=1e-12)


class TestPDControllerProportionalTerm:
    """Pure displacement (zero velocity) → force = -Kp * error, axis-by-axis."""

    @pytest.mark.parametrize("axis,idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_single_axis_displacement(self, axis, idx):
        Kp  = 0.05
        pos = [0.0, 0.0, 0.0]
        pos[idx] = 50.0
        ctrl = PDController(Kp=Kp, Kd=0.0)
        cmd  = ctrl.step(_make_state(pos, [0, 0, 0]))

        expected      = np.zeros(3)
        expected[idx] = -Kp * 50.0
        np.testing.assert_allclose(cmd.force, expected, atol=1e-12)

    def test_force_opposes_displacement(self):
        ctrl = PDController(Kp=0.1, Kd=0.0)
        cmd  = ctrl.step(_make_state([1.0, 0.0, 0.0], [0, 0, 0]))
        assert cmd.force[0] < 0, "Force should oppose positive x displacement"


class TestPDControllerDerivativeTerm:
    """Pure velocity (at desired position) → force = -Kd * vel."""

    @pytest.mark.parametrize("axis,idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_single_axis_velocity(self, axis, idx):
        Kd  = 10.0
        vel = [0.0, 0.0, 0.0]
        vel[idx] = 2.0
        ctrl = PDController(Kp=0.0, Kd=Kd)
        cmd  = ctrl.step(_make_state([0, 0, 0], vel))

        expected      = np.zeros(3)
        expected[idx] = -Kd * 2.0
        np.testing.assert_allclose(cmd.force, expected, atol=1e-12)

    def test_force_opposes_velocity(self):
        ctrl = PDController(Kp=0.0, Kd=5.0)
        cmd  = ctrl.step(_make_state([0, 0, 0], [0.0, 1.0, 0.0]))
        assert cmd.force[1] < 0, "Force should oppose positive y velocity"


class TestPDControllerSuperposition:
    """Both displacement and velocity active → forces add linearly."""

    def test_superposition(self):
        Kp, Kd = 0.05, 10.0
        pos = [20.0, -10.0, 5.0]
        vel = [0.5,   1.0, -0.3]
        ctrl = PDController(Kp=Kp, Kd=Kd)
        cmd  = ctrl.step(_make_state(pos, vel))

        expected = -Kp * np.array(pos) - Kd * np.array(vel)
        np.testing.assert_allclose(cmd.force, expected, atol=1e-12)


class TestPDControllerOutputFormat:
    """ControlCommand has correct shapes and valid flag."""

    def test_torque_is_zero(self):
        ctrl = PDController(Kp=0.05, Kd=10.0)
        cmd  = ctrl.step(_make_state([1, 2, 3], [0.1, 0.2, 0.3]))
        np.testing.assert_allclose(cmd.torque, [0, 0, 0], atol=1e-12)

    def test_valid_flag(self):
        ctrl = PDController(Kp=0.05, Kd=10.0)
        cmd  = ctrl.step(_make_state([1, 0, 0], [0, 0, 0]))
        assert cmd.valid is True

    def test_force_shape(self):
        ctrl = PDController(Kp=0.05, Kd=10.0)
        cmd  = ctrl.step(_make_state([1, 2, 3], [0, 0, 0]))
        assert cmd.force.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
