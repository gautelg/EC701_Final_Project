"""
tests/test_controller_adapter.py

Unit tests for ControllerAdapter.

No Basilisk dependency. Uses a stub controller that returns a fixed command
to isolate clipping and pass-through logic.

How to run (from repo root)
---------------------------
    python -m pytest src/sim/tests/test_controller_adapter.py -v
"""

import os
import sys

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR   = os.path.dirname(_TESTS_DIR)
_SRC_DIR   = os.path.dirname(_SIM_DIR)
for p in [_SRC_DIR, _SIM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter.controller_adapter import ControllerAdapter
from interface.sim_state import SimState
from interface.control_command import ControlCommand
from control.base_controller import BaseController


# ---------------------------------------------------------------------------
# Stub controller
# ---------------------------------------------------------------------------

class _StubController(BaseController):
    """Returns a fixed ControlCommand regardless of state."""

    def __init__(self, force, torque, valid=True):
        self._cmd = ControlCommand(
            force  = np.array(force,  dtype=float),
            torque = np.array(torque, dtype=float),
            valid  = valid,
        )

    def step(self, state: SimState) -> ControlCommand:
        return self._cmd


def _dummy_state():
    return SimState(
        time       = 0.0,
        rel_pos    = np.zeros(3),
        rel_vel    = np.zeros(3),
        quaternion = np.array([0.0, 0.0, 0.0, 1.0]),
        omega      = np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Tests — pass-through (within limits)
# ---------------------------------------------------------------------------

class TestAdapterPassThrough:
    """Commands within limits should pass through unchanged with valid=True."""

    def test_force_unchanged(self):
        ctrl    = _StubController([1.0, -2.0, 3.0], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        np.testing.assert_allclose(cmd.force, [1.0, -2.0, 3.0])

    def test_torque_unchanged(self):
        ctrl    = _StubController([0.0, 0.0, 0.0], [0.1, -0.2, 0.3])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        np.testing.assert_allclose(cmd.torque, [0.1, -0.2, 0.3])

    def test_valid_preserved(self):
        ctrl    = _StubController([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.valid is True

    def test_no_limits_passes_anything(self):
        ctrl    = _StubController([1e6, 1e6, 1e6], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {})   # defaults to inf limits
        cmd     = adapter.step(_dummy_state())
        np.testing.assert_allclose(cmd.force, [1e6, 1e6, 1e6])
        assert cmd.valid is True


# ---------------------------------------------------------------------------
# Tests — force clipping
# ---------------------------------------------------------------------------

class TestAdapterForceClipping:
    """Force components exceeding max_force should be clipped and valid=False."""

    def test_positive_force_clipped(self):
        ctrl    = _StubController([20.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.force[0] == pytest.approx(10.0)
        assert cmd.valid is False

    def test_negative_force_clipped(self):
        ctrl    = _StubController([-20.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.force[0] == pytest.approx(-10.0)
        assert cmd.valid is False

    def test_unclipped_axes_unchanged(self):
        ctrl    = _StubController([20.0, 1.0, -1.0], [0.0, 0.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.force[1] == pytest.approx(1.0)
        assert cmd.force[2] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests — torque clipping
# ---------------------------------------------------------------------------

class TestAdapterTorqueClipping:
    """Torque components exceeding max_torque should be clipped and valid=False."""

    def test_torque_clipped(self):
        ctrl    = _StubController([0.0, 0.0, 0.0], [0.0, 10.0, 0.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.torque[1] == pytest.approx(5.0)
        assert cmd.valid is False

    def test_force_ok_torque_clipped_sets_invalid(self):
        ctrl    = _StubController([1.0, 0.0, 0.0], [0.0, 0.0, 100.0])
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.valid is False


# ---------------------------------------------------------------------------
# Tests — upstream invalid propagated
# ---------------------------------------------------------------------------

class TestAdapterUpstreamInvalid:
    """If the underlying controller returns valid=False, adapter preserves it."""

    def test_upstream_invalid_preserved(self):
        ctrl    = _StubController([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], valid=False)
        adapter = ControllerAdapter(ctrl, {"max_force": 10.0, "max_torque": 5.0})
        cmd     = adapter.step(_dummy_state())
        assert cmd.valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
