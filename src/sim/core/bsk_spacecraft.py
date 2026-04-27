"""
bsk_spacecraft.py

Configures two Basilisk Spacecraft objects — target and chaser.

Sets mass properties, inertia, initial inertial state (converted from
orbital elements + Hill-frame offset), and attaches Earth gravity.
Message recorders for position, velocity, attitude, and angular velocity
are added here so the test script can extract logged data after the run.

apply_control() is a no-op in Stage 1 (zero-control test).
"""

import numpy as np

from Basilisk.simulation import spacecraft, extForceTorque
from Basilisk.utilities import orbitalMotion, macros


def _hill_to_inertial_dcm(r_N, v_N):
    """
    Return the 3x3 DCM that rotates vectors from the Hill (LVLH) frame
    into the inertial frame.

    Hill frame convention
    ---------------------
    x  : radial outward (R-bar)
    y  : along-track, approx. velocity direction for circular orbit (V-bar)
    z  : orbit normal, direction of h = r × v (H-bar)

    The columns of the returned matrix are the Hill basis vectors expressed
    in the inertial frame.
    """
    r_N = np.asarray(r_N, dtype=float)
    v_N = np.asarray(v_N, dtype=float)

    e_x = r_N / np.linalg.norm(r_N)
    h = np.cross(r_N, v_N)
    e_z = h / np.linalg.norm(h)
    e_y = np.cross(e_z, e_x)

    return np.column_stack([e_x, e_y, e_z])   # shape (3, 3)


def _hill_velocity_to_inertial_delta(r_tgt_N, v_tgt_N, dr_hill, dv_hill):
    """
    Convert Hill-frame relative position/velocity into an inertial velocity offset.

    The Hill-frame relative velocity includes the rotating-frame transport term.
    Inverting the read-side relation used by BskInterface requires adding
    omega_H x dr before rotating back to inertial coordinates.
    """
    r_tgt_N = np.asarray(r_tgt_N, dtype=float)
    v_tgt_N = np.asarray(v_tgt_N, dtype=float)
    dr_hill = np.asarray(dr_hill, dtype=float)
    dv_hill = np.asarray(dv_hill, dtype=float)

    R_HN = _hill_to_inertial_dcm(r_tgt_N, v_tgt_N)
    h_tgt = np.cross(r_tgt_N, v_tgt_N)
    n = np.linalg.norm(h_tgt) / np.linalg.norm(r_tgt_N) ** 2
    omega_H = np.array([0.0, 0.0, n], dtype=float)
    return R_HN @ (dv_hill + np.cross(omega_H, dr_hill))


class BskSpacecraft:

    def __init__(self, sim, config):
        """
        Parameters
        ----------
        sim : BskSim
        config : dict
            Top-level configuration dictionary (from sim_config.yaml).
        """
        self.sim = sim
        self.config = config

        # Spacecraft module handles (set in setup())
        self.scTarget = None
        self.scChaser = None

        # Message recorders (set in setup(), read after run())
        self.scTargetLog = None
        self.scChaserLog = None

        # External force/torque effector for chaser (used by BskInterface)
        self.chaserExtFT = None
        self.targetEnvExtFT = None
        self.chaserEnvExtFT = None
        self.target_mass = None
        self.chaser_mass = None

    def setup(self, environment):
        """
        Create both spacecraft, set initial conditions, attach gravity,
        and add message recorders to the dynamics task.

        Parameters
        ----------
        environment : BskEnvironment
            Must already be setup() so gravFactory and earth.mu are available.
        """
        sc_cfg  = self.config["spacecraft"]
        orb_cfg = self.config["orbit"]
        tgt_cfg = self.config.get("target", {})
        chs_cfg = self.config["chaser"]

        mass  = sc_cfg["mass"]
        self.target_mass = float(mass)
        self.chaser_mass = float(mass)
        I_d   = sc_cfg["inertia_diag"]
        I_mat = [[I_d[0], 0.0,    0.0   ],
                 [0.0,    I_d[1], 0.0   ],
                 [0.0,    0.0,    I_d[2]]]

        mu = environment.earth.mu   # m^3/s^2

        # ------------------------------------------------------------------ #
        # Target: initial inertial state from classical orbital elements
        # ------------------------------------------------------------------ #
        oe = orbitalMotion.ClassicElements()
        oe.a     = orb_cfg["a"]
        oe.e     = orb_cfg["e"]
        oe.i     = orb_cfg["i"]     * macros.D2R
        oe.Omega = orb_cfg["Omega"] * macros.D2R
        oe.omega = orb_cfg["omega"] * macros.D2R
        oe.f     = orb_cfg["f"]     * macros.D2R

        r_tgt, v_tgt = orbitalMotion.elem2rv(mu, oe)
        r_tgt = np.array(r_tgt)
        v_tgt = np.array(v_tgt)

        # ------------------------------------------------------------------ #
        # Chaser: apply Hill-frame offset to target's inertial state
        # ------------------------------------------------------------------ #
        R_HN = _hill_to_inertial_dcm(r_tgt, v_tgt)   # Hill → inertial

        dr_hill = np.array(chs_cfg["offset_hill"])     # m,   [radial, along-track, normal]
        dv_hill = np.array(chs_cfg["v_offset_hill"])   # m/s

        r_chs = r_tgt + R_HN @ dr_hill
        v_chs = v_tgt + _hill_velocity_to_inertial_delta(r_tgt, v_tgt, dr_hill, dv_hill)

        # ------------------------------------------------------------------ #
        # Build both spacecraft modules
        # ------------------------------------------------------------------ #
        self.scTarget = self._make_sc(
            tag    = "target",
            mass   = mass,
            I_mat  = I_mat,
            r_init = r_tgt,
            v_init = v_tgt,
            sigma  = tgt_cfg.get("attitude_mrp", [0.0, 0.0, 0.0]),
            omega  = tgt_cfg.get("omega_BN_B",   [0.0, 0.0, 0.0]),
        )
        self.scChaser = self._make_sc(
            tag    = "chaser",
            mass   = mass,
            I_mat  = I_mat,
            r_init = r_chs,
            v_init = v_chs,
            sigma  = chs_cfg.get("attitude_mrp", [0.0, 0.0, 0.0]),
            omega  = chs_cfg.get("omega_BN_B",   [0.0, 0.0, 0.0]),
        )

        # ------------------------------------------------------------------ #
        # External force/torque effector for chaser (defaults to zero)
        # BskInterface.write_command() sets these each controller step.
        # ------------------------------------------------------------------ #
        self.chaserExtFT = extForceTorque.ExtForceTorque()
        self.chaserExtFT.ModelTag = "chaserExtFT"
        self.scChaser.addDynamicEffector(self.chaserExtFT)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.chaserExtFT)

        self.targetEnvExtFT = extForceTorque.ExtForceTorque()
        self.targetEnvExtFT.ModelTag = "targetEnvExtFT"
        self.scTarget.addDynamicEffector(self.targetEnvExtFT)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.targetEnvExtFT)

        self.chaserEnvExtFT = extForceTorque.ExtForceTorque()
        self.chaserEnvExtFT.ModelTag = "chaserEnvExtFT"
        self.scChaser.addDynamicEffector(self.chaserEnvExtFT)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.chaserEnvExtFT)

        # ------------------------------------------------------------------ #
        # Attach Earth gravity to both spacecraft
        # ------------------------------------------------------------------ #
        grav_bodies = spacecraft.GravBodyVector(
            list(environment.gravFactory.gravBodies.values())
        )
        self.scTarget.gravField.gravBodies = grav_bodies
        self.scChaser.gravField.gravBodies = grav_bodies

        # ------------------------------------------------------------------ #
        # Register with the dynamics task
        # ------------------------------------------------------------------ #
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.scTarget)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.scChaser)

        # ------------------------------------------------------------------ #
        # Message recorders — log every dynamics step
        # ------------------------------------------------------------------ #
        self.scTargetLog = self.scTarget.scStateOutMsg.recorder()
        self.scChaserLog = self.scChaser.scStateOutMsg.recorder()
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.scTargetLog)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.scChaserLog)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _make_sc(tag, mass, I_mat, r_init, v_init, sigma, omega):
        """Instantiate and configure a Basilisk Spacecraft module."""
        sc = spacecraft.Spacecraft()
        sc.ModelTag = tag

        sc.hub.mHub         = mass
        sc.hub.IHubPntBc_B  = I_mat

        sc.hub.r_CN_NInit      = r_init.tolist()
        sc.hub.v_CN_NInit      = v_init.tolist()
        sc.hub.sigma_BNInit    = [[sigma[0]], [sigma[1]], [sigma[2]]]
        sc.hub.omega_BN_BInit  = [[omega[0]], [omega[1]], [omega[2]]]

        return sc

    def apply_control(self, control_command):
        """No-op — force application is handled by BskInterface.write_command()."""
        pass
