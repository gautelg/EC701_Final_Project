"""
sim_state.py

Defines the SimState dataclass: the data contract passed from the simulator
to the controller at each controller update step.

Units and conventions to be finalized here before implementation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimState:
    """
    Snapshot of the simulation state passed to the controller.

    Attributes
    ----------
    time : float
        Current simulation time (s).
    rel_pos : np.ndarray, shape (3,)
        Relative position of chaser w.r.t. target, expressed in the Hill frame
        (x=R-bar radial, y=V-bar along-track, z=H-bar orbit-normal) (m).
    rel_vel : np.ndarray, shape (3,)
        Relative velocity of chaser w.r.t. target, expressed in the Hill frame (m/s).
    quaternion : np.ndarray, shape (4,)
        Chaser attitude quaternion, scalar-last convention [qx, qy, qz, qw].
    omega : np.ndarray, shape (3,)
        Chaser body angular velocity (rad/s).
    rel_pos_inertial : np.ndarray, shape (3,), optional
        Relative position of chaser w.r.t. target in the inertial frame (m).
        Used by controllers that need an inertial line of sight.
    hill_to_inertial_dcm : np.ndarray, shape (3, 3), optional
        DCM rotating vectors from Hill frame into inertial frame.
    """
    time: float
    rel_pos: np.ndarray
    rel_vel: np.ndarray
    quaternion: np.ndarray
    omega: np.ndarray
    rel_pos_inertial: np.ndarray | None = None
    hill_to_inertial_dcm: np.ndarray | None = None
