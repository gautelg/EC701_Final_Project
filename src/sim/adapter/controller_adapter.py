"""
controller_adapter.py

Controller Wrapper / Adapter.

Calls the controller's step() method, clips force and torque to configured
limits, and returns a ControlCommand. Neither Basilisk internals nor
controller implementation details cross this boundary.
"""

import numpy as np

from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand
from control.base_controller import BaseController


class ControllerAdapter:

    def __init__(self, controller: BaseController, config: dict):
        """
        Parameters
        ----------
        controller : BaseController
            Any controller implementing step(SimState) -> ControlCommand.
        config : dict
            Must contain:
              'max_force'  : float  — per-axis force limit (N)
              'max_torque' : float  — per-axis torque limit (N·m)
        """
        self.controller = controller
        self.max_force = float(config.get("max_force", np.inf))
        self.max_torque = float(config.get("max_torque", np.inf))

    def step(self, state: SimState) -> ControlCommand:
        """Run one controller update and return a clipped ControlCommand."""
        cmd = self.controller.step(state)
        clipped_force = np.clip(cmd.force, -self.max_force, self.max_force)
        clipped_torque = np.clip(cmd.torque, -self.max_torque, self.max_torque)
        was_clipped = not (
            np.array_equal(clipped_force, cmd.force)
            and np.array_equal(clipped_torque, cmd.torque)
        )
        return ControlCommand(
            force=clipped_force,
            torque=clipped_torque,
            valid=cmd.valid and not was_clipped,
        )
