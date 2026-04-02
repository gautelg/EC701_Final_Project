"""
controller_adapter.py

Controller Wrapper / Adapter.

Converts a SimState into the MPC's expected input vector, calls the MPC
solver, clips or validates the output, and returns a ControlCommand.

The MPC code never interacts with Basilisk internals directly.
"""

from interface.sim_state import SimState
from interface.control_command import ControlCommand


class ControllerAdapter:

    def __init__(self, mpc_controller, config):
        """
        Parameters
        ----------
        mpc_controller : object
            The MPC controller instance (interface TBD based on control module).
        config : dict
            Adapter configuration (force limits, torque limits, etc.).
        """
        pass

    def step(self, state: SimState) -> ControlCommand:
        """
        Run one controller update.

        Parameters
        ----------
        state : SimState
            Current simulation state.

        Returns
        -------
        ControlCommand
            Commanded force and torque for the next hold interval.
        """
        pass

    def _build_mpc_input(self, state: SimState):
        """Convert SimState to the MPC input vector format."""
        pass

    def _parse_mpc_output(self, mpc_output) -> ControlCommand:
        """Convert MPC output to ControlCommand, clipping if needed."""
        pass
