"""
run translational MPC until ||r - r_wp|| < \epsilon_r, ||v|| < \epsilon_v

switch to attitude mode

rotate until pointing error below threshold, bodyrate below threshold

switch back to translational mode and move on to next waypoint

"""

import numpy as np


class MissionManager:
    def __init__(
        self,
        waypoints,
        eps_r=0.1,
        eps_v=0.01,
        eps_q=0.05,
        eps_w=0.01,
        max_steps=5000,
        required_count=5,
    ):
        self.waypoints = waypoints
        self.eps_r = eps_r
        self.eps_v = eps_v
        self.eps_q = eps_q      # tolerance on quaternion error vector norm
        self.eps_w = eps_w      # tolerance on angular rate norm
        self.max_steps = max_steps
        self.required_count = required_count    # chatter prevention

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
        """
        x_trans = [rx, ry, rz, vx, vy, vz]
        """
        r_wp = self.current_waypoint()
        if r_wp is None:
            return False

        pos_err = np.linalg.norm(x_trans[:3] - r_wp)
        vel_err = np.linalg.norm(x_trans[3:])

        return (pos_err < self.eps_r) and (vel_err < self.eps_v)

    def rotation_complete(self, q_err, omega):
        """
        q_err: quaternion error
        omega: body angular velocity
        """
        att_err = np.linalg.norm(q_err[1:])   # vector part
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
        return {
            "mode": self.mode,
            "wp_idx": self.wp_idx,
            "done": self.done,
            "current_waypoint": self.current_waypoint(),
            "translate_counter": self.translate_counter,
            "rotate_counter": self.rotate_counter,
        }



