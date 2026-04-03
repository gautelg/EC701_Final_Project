"""
base_controller.py

Abstract base class for all controllers (PD stand-in, MPC, etc.).

Any controller that plugs into the sim must subclass BaseController and
implement step(). The adapter calls nothing else.
"""

from abc import ABC, abstractmethod

from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand


class BaseController(ABC):

    @abstractmethod
    def step(self, state: SimState) -> ControlCommand:
        """
        Compute the control command for the current timestep.

        Parameters
        ----------
        state : SimState
            Current simulation state (Hill-frame relative position/velocity,
            chaser attitude quaternion and angular velocity).

        Returns
        -------
        ControlCommand
            Commanded force (Hill frame, N) and torque (body frame, N·m).
        """
