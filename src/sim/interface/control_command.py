"""
control_command.py

Defines the ControlCommand dataclass: the data contract returned by the
controller and consumed by the simulator at each controller update step.

Units and conventions to be finalized here before implementation.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ControlCommand:
    """
    Control output returned by the MPC and applied to the spacecraft.

    Attributes
    ----------
    force : np.ndarray, shape (3,)
        Commanded force expressed in [frame TBD] (N).
    torque : np.ndarray, shape (3,)
        Commanded torque expressed in body frame (N·m).
    valid : bool
        True if the solver returned a valid solution; False if it failed or
        was clipped. Simulator should handle invalid commands gracefully.
    """
    force: np.ndarray
    torque: np.ndarray
    valid: bool = True
