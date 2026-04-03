"""
bsk_interface.py

Message bridge between Basilisk and the rest of the sim pipeline.

read_state()   — samples live Basilisk messages and returns a SimState.
write_command() — converts a ControlCommand into extForceTorque inputs.

Frame conventions (Stage 1)
---------------------------
- rel_pos / rel_vel  : Hill frame  (x=R-bar, y=V-bar, z=H-bar)
- force in command   : Hill frame  → rotated to inertial before applying
- torque in command  : body frame  → applied directly
- quaternion         : scalar-last [qx, qy, qz, qw] converted from MRP
"""

import numpy as np

from Basilisk.utilities import macros, RigidBodyKinematics as rbk

from interface.sim_state import SimState
from interface.control_command import ControlCommand


def _hill_to_inertial_dcm(r_N, v_N):
    """Return 3x3 DCM rotating vectors from Hill frame into the inertial frame."""
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)
    e_x = r_N / np.linalg.norm(r_N)
    h   = np.cross(r_N, v_N)
    e_z = h / np.linalg.norm(h)
    e_y = np.cross(e_z, e_x)
    return np.column_stack([e_x, e_y, e_z])   # Hill → inertial


def _mrp_to_quat_scalar_last(sigma):
    """Convert a Basilisk MRP (3-vector) to a scalar-last quaternion [qx,qy,qz,qw]."""
    # rbk.MRP2EP returns scalar-first: [q0, q1, q2, q3]
    q_sf = rbk.MRP2EP(sigma)
    return np.array([q_sf[1], q_sf[2], q_sf[3], q_sf[0]])


class BskInterface:

    def __init__(self, spacecraft, environment):
        """
        Parameters
        ----------
        spacecraft  : BskSpacecraft  (must have chaserExtFT set, i.e. after setup())
        environment : BskEnvironment (unused in Stage 1, reserved for Stage 3)
        """
        self.spacecraft  = spacecraft
        self.environment = environment

    # ---------------------------------------------------------------------- #

    def read_state(self) -> SimState:
        """
        Sample live Basilisk messages and return a SimState.

        rel_pos and rel_vel are expressed in the Hill frame defined by the
        target spacecraft's current position and velocity.
        """
        tgt = self.spacecraft.scTarget.scStateOutMsg.read()
        chs = self.spacecraft.scChaser.scStateOutMsg.read()

        r_tgt = np.array(tgt.r_BN_N)
        v_tgt = np.array(tgt.v_BN_N)
        r_chs = np.array(chs.r_BN_N)
        v_chs = np.array(chs.v_BN_N)

        R_HN = _hill_to_inertial_dcm(r_tgt, v_tgt).T   # inertial → Hill

        rel_pos = R_HN @ (r_chs - r_tgt)
        # Stage 1: simple inertial-difference projected into Hill frame.
        # Does not subtract the frame-rotation term (O(n·Δr) ≈ 0.1 m/s for 100 m sep).
        rel_vel = R_HN @ (v_chs - v_tgt)

        quaternion = _mrp_to_quat_scalar_last(np.array(chs.sigma_BN))
        omega      = np.array(chs.omega_BN_B)
        time_s     = self.spacecraft.sim.scSim.TotalSim.CurrentNanos * macros.NANO2SEC

        return SimState(
            time       = time_s,
            rel_pos    = rel_pos,
            rel_vel    = rel_vel,
            quaternion = quaternion,
            omega      = omega,
        )

    # ---------------------------------------------------------------------- #

    def write_command(self, command: ControlCommand):
        """
        Apply a ControlCommand to the chaser's extForceTorque effector.

        Force is converted from Hill frame to inertial before being passed
        to Basilisk (extForce_N). Torque is applied directly in body frame.
        """
        tgt = self.spacecraft.scTarget.scStateOutMsg.read()
        r_tgt = np.array(tgt.r_BN_N)
        v_tgt = np.array(tgt.v_BN_N)

        R_HN = _hill_to_inertial_dcm(r_tgt, v_tgt)   # Hill → inertial
        f_inertial = R_HN @ command.force

        self.spacecraft.chaserExtFT.extForce_N       = f_inertial.tolist()
        self.spacecraft.chaserExtFT.extTorquePntB_B  = command.torque.tolist()
