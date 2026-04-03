"""
pd_controller.py

PD stand-in controller for closed-loop infrastructure testing.

Drives the chaser to a desired relative position in the Hill frame using a
simple proportional-derivative law:

    F = -Kp * (r_rel - r_des) - Kd * v_rel

Replace this with the MPC by swapping the controller passed to ControllerAdapter.
No torque is commanded (attitude control not included at Stage 1).
"""

import numpy as np

from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand
from control.base_controller import BaseController


class PDController(BaseController):

    def __init__(self, Kp: float, Kd: float, desired_rel_pos=None):
        """
        Parameters
        ----------
        Kp : float
            Proportional gain (N/m).
        Kd : float
            Derivative gain (N·s/m).
        desired_rel_pos : array-like, shape (3,), optional
            Target relative position in the Hill frame (m).
            Defaults to [0, 0, 0] (rendezvous / docking point).
        """
        self.Kp = Kp
        self.Kd = Kd
        self.desired_rel_pos = (
            np.asarray(desired_rel_pos, dtype=float)
            if desired_rel_pos is not None
            else np.zeros(3)
        )

    def step(self, state: SimState) -> ControlCommand:
        pos_error = state.rel_pos - self.desired_rel_pos
        force = -self.Kp * pos_error - self.Kd * state.rel_vel
        return ControlCommand(
            force=force,
            torque=np.zeros(3),
            valid=True,
        )
