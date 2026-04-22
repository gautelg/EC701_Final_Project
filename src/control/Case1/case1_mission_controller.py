"""
Case 1 mission controller.

This module adapts the standalone Case1 translation, CBF, and attitude
controllers to the common BaseController -> ControlCommand sim interface.
"""

from dataclasses import dataclass

import numpy as np

from sim.interface.sim_state import SimState
from sim.interface.control_command import ControlCommand
from control.base_controller import BaseController
from control.Case1.Case1_attitude_controller import (
    attitude_pd_control,
    compute_desired_pointing_quaternion,
    quat_error,
)
from control.Case1.Case1_cbf import cbf_filter_translation
from control.Case1.Case1_translation_controller import (
    discretize_system,
    hcw_matrices,
    solve_mpc,
)


def _as_array(value, shape=None):
    arr = np.asarray(value, dtype=float)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def _diag_matrix(value, size):
    arr = np.asarray(value, dtype=float)
    if arr.shape == (size, size):
        return arr
    if arr.shape == (size,):
        return np.diag(arr)
    raise ValueError(f"Expected {size} values or a {size}x{size} matrix, got {arr.shape}")


def _scalar_last_to_first(q):
    q = np.asarray(q, dtype=float)
    # Basilisk sigma_BN converts to a passive B<-N quaternion. The Case1
    # attitude prototype uses active body-to-inertial quaternions, so use the
    # conjugate while converting scalar-last -> scalar-first.
    return np.array([q[3], -q[0], -q[1], -q[2]], dtype=float)


@dataclass
class Case1MissionStatus:
    mode: str
    waypoint_index: int
    done: bool
    current_waypoint: np.ndarray | None
    translate_counter: int
    rotate_counter: int


class Case1MissionManager:
    """Mission mode logic copied from the standalone Case1 prototype."""

    def __init__(
        self,
        waypoints,
        eps_r=0.1,
        eps_v=0.01,
        eps_q=0.05,
        eps_w=0.01,
        required_count=5,
    ):
        self.waypoints = [_as_array(wp, (3,)) for wp in waypoints]
        self.eps_r = float(eps_r)
        self.eps_v = float(eps_v)
        self.eps_q = float(eps_q)
        self.eps_w = float(eps_w)
        self.required_count = int(required_count)

        self.mode = "TRANSLATE"
        self.wp_idx = 0
        self.done = False
        self.translate_counter = 0
        self.rotate_counter = 0

    def current_waypoint(self):
        if self.wp_idx >= len(self.waypoints):
            return None
        return self.waypoints[self.wp_idx]

    def translation_complete(self, x_trans):
        r_wp = self.current_waypoint()
        if r_wp is None:
            return False
        pos_err = np.linalg.norm(x_trans[:3] - r_wp)
        vel_err = np.linalg.norm(x_trans[3:])
        return (pos_err < self.eps_r) and (vel_err < self.eps_v)

    def rotation_complete(self, q_err, omega):
        att_err = np.linalg.norm(q_err[1:])
        rate_err = np.linalg.norm(omega)
        return (att_err < self.eps_q) and (rate_err < self.eps_w)

    def update_mode(self, x_trans, q_err, omega):
        if self.done:
            return

        if self.mode == "TRANSLATE":
            if self.translation_complete(x_trans):
                self.translate_counter += 1
            else:
                self.translate_counter = 0

            if self.translate_counter >= self.required_count:
                self.mode = "ROTATE"
                self.translate_counter = 0

        elif self.mode == "ROTATE":
            if self.rotation_complete(q_err, omega):
                self.rotate_counter += 1
            else:
                self.rotate_counter = 0

            if self.rotate_counter >= self.required_count:
                self.wp_idx += 1
                self.rotate_counter = 0

                if self.wp_idx >= len(self.waypoints):
                    self.done = True
                else:
                    self.mode = "TRANSLATE"

    def status(self):
        return Case1MissionStatus(
            mode=self.mode,
            waypoint_index=self.wp_idx,
            done=self.done,
            current_waypoint=self.current_waypoint(),
            translate_counter=self.translate_counter,
            rotate_counter=self.rotate_counter,
        )


class Case1MissionController(BaseController):
    """
    Combined Case1 translation, CBF, attitude, and waypoint-mode controller.

    The standalone translation controller returns acceleration in the Hill
    frame. This adapter multiplies by spacecraft mass before returning a
    ControlCommand force.
    """

    def __init__(self, config):
        self.config = dict(config)

        self.mass = float(self.config["mass"])
        self.n = float(self.config["mean_motion"])
        self.dt = float(self.config["controller_dt"])
        self.N = int(self.config.get("horizon", 20))

        self.u_max = _as_array(self.config.get("u_max", [0.01, 0.01, 0.01]), (3,))
        self.Q = _diag_matrix(self.config.get("Q", [10, 10, 10, 5, 5, 5]), 6)
        self.R = _diag_matrix(self.config.get("R", [0.1, 0.1, 0.1]), 3)
        self.P = _diag_matrix(self.config.get("P", [200, 200, 200, 100, 100, 100]), 6)

        A, B = hcw_matrices(self.n)
        self.Ad, self.Bd = discretize_system(A, B, self.dt)

        self.use_cbf = bool(self.config.get("use_cbf", True))
        self.R_koz = float(self.config.get("R_koz", 5.0))
        self.cbf_k0 = float(self.config.get("cbf_k0", 1.0))
        self.cbf_k1 = float(self.config.get("cbf_k1", 2.0))
        self.cbf_rho = float(self.config.get("cbf_rho", 1e4))
        self.cbf_use_slack = bool(self.config.get("cbf_use_slack", True))

        self.Kq = _diag_matrix(self.config.get("Kq", [1.0, 1.0, 1.0]), 3)
        self.Kw = _diag_matrix(self.config.get("Kw", [6.0, 6.0, 6.0]), 3)
        self.tau_max = _as_array(self.config.get("tau_max", [0.05, 0.05, 0.05]), (3,))

        waypoints = self.config.get("waypoints")
        if not waypoints:
            raise ValueError("Case1MissionController requires at least one waypoint")

        mission_cfg = self.config.get("mission", {})
        self.manager = Case1MissionManager(
            waypoints=waypoints,
            eps_r=mission_cfg.get("eps_r", 0.1),
            eps_v=mission_cfg.get("eps_v", 0.01),
            eps_q=mission_cfg.get("eps_q", 0.05),
            eps_w=mission_cfg.get("eps_w", 0.01),
            required_count=mission_cfg.get("required_count", 5),
        )

        self.last_accel = np.zeros(3)
        self.last_q_err = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_valid = True

    @property
    def done(self):
        return self.manager.done

    def status(self):
        return self.manager.status()

    def step(self, state: SimState) -> ControlCommand:
        if self.manager.done:
            return ControlCommand(force=np.zeros(3), torque=np.zeros(3), valid=True)

        x_trans = np.hstack([state.rel_pos, state.rel_vel])
        waypoint = self.manager.current_waypoint()
        if waypoint is None:
            self.manager.done = True
            return ControlCommand(force=np.zeros(3), torque=np.zeros(3), valid=True)

        accel, trans_valid = self._translation_accel(x_trans, waypoint)
        torque, q_err = self._attitude_torque(state)

        if self.manager.mode == "TRANSLATE":
            torque_to_apply = np.zeros(3)
        else:
            torque_to_apply = torque

        self.manager.update_mode(x_trans, q_err, state.omega)

        self.last_accel = accel
        self.last_q_err = q_err
        self.last_valid = trans_valid

        return ControlCommand(
            force=self.mass * accel,
            torque=torque_to_apply,
            valid=trans_valid,
        )

    def _translation_accel(self, x_trans, waypoint):
        x_ref = np.hstack([waypoint, np.zeros(3)])
        try:
            u_nom = solve_mpc(
                x_trans,
                x_ref,
                self.Ad,
                self.Bd,
                self.Q,
                self.R,
                self.P,
                self.N,
                self.u_max,
            )
            valid = True
        except RuntimeError:
            u_nom = np.zeros(3)
            valid = False

        if self.use_cbf:
            u = cbf_filter_translation(
                x_trans,
                u_nom,
                self.n,
                self.u_max,
                self.R_koz,
                k0=self.cbf_k0,
                k1=self.cbf_k1,
                rho=self.cbf_rho,
                use_slack=self.cbf_use_slack,
            )
        else:
            u = np.clip(u_nom, -self.u_max, self.u_max)

        return u, valid

    def _attitude_torque(self, state: SimState):
        q_current = _scalar_last_to_first(state.quaternion)
        los_inertial = self._line_of_sight_to_target_inertial(state)

        q_des = compute_desired_pointing_quaternion(
            chaser_pos=np.zeros(3),
            target_pos=los_inertial,
        )
        # The standalone attitude prototype and Basilisk torque convention use
        # opposite error directions in this integration. Swapping the arguments
        # keeps the prototype function unchanged while preserving rate damping.
        torque, q_err = attitude_pd_control(
            q_des,
            state.omega,
            q_current,
            self.Kq,
            self.Kw,
            self.tau_max,
        )
        return torque, q_err

    def _line_of_sight_to_target_inertial(self, state: SimState):
        if state.rel_pos_inertial is not None:
            los = -np.asarray(state.rel_pos_inertial, dtype=float)
        elif state.hill_to_inertial_dcm is not None:
            los = -np.asarray(state.hill_to_inertial_dcm, dtype=float) @ state.rel_pos
        else:
            los = -np.asarray(state.rel_pos, dtype=float)

        if np.linalg.norm(los) < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return los

    def pointing_error(self, state: SimState):
        q_current = _scalar_last_to_first(state.quaternion)
        los_inertial = self._line_of_sight_to_target_inertial(state)
        q_des = compute_desired_pointing_quaternion(np.zeros(3), los_inertial)
        return quat_error(q_current, q_des)
