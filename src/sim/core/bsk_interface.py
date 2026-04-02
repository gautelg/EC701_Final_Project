"""
bsk_interface.py

Message bridge between Basilisk and the rest of the sim pipeline.

Reads Basilisk output messages and packages them into a SimState.
Writes a ControlCommand back into the appropriate Basilisk input message.
"""

from interface.sim_state import SimState
from interface.control_command import ControlCommand


class BskInterface:

    def __init__(self, spacecraft, environment):
        """
        Parameters
        ----------
        spacecraft : BskSpacecraft
        environment : BskEnvironment
        """
        pass

    def read_state(self) -> SimState:
        """
        Read current Basilisk output messages and return a SimState.

        Returns
        -------
        SimState
            Current simulation state (time, rel. pos, rel. vel, quaternion, omega).
        """
        pass

    def write_command(self, command: ControlCommand):
        """
        Write a ControlCommand into the appropriate Basilisk input message.

        Parameters
        ----------
        command : ControlCommand
        """
        pass
