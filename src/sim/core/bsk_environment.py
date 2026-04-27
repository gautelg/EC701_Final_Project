"""
bsk_environment.py

Adds environmental models to the Basilisk scene.

Stage 1 (this file): point-mass Earth gravity only.
Stage 3 (future): J2, solar radiation pressure, atmospheric drag.

The gravFactory object is the authoritative source of gravity body data
(including Earth's mu) used by BskSpacecraft to set initial orbital elements.
"""

from Basilisk.utilities import simIncludeGravBody

try:
    from sim.core.j2_perturbation import J2PerturbationModel
except ModuleNotFoundError:  # pragma: no cover - compatibility with legacy launch paths
    from core.j2_perturbation import J2PerturbationModel


class BskEnvironment:

    def __init__(self, sim, config):
        """
        Parameters
        ----------
        sim : BskSim
            The parent simulation instance.
        config : dict
            Top-level configuration dictionary (from sim_config.yaml).
        """
        self.sim = sim
        self.config = config

        self.gravFactory = None
        self.earth = None
        self.j2_enabled = False
        self.j2_model = None

    def setup(self):
        """Create Earth point-mass gravity body."""
        env_cfg = self.config.get("environment", {})
        gravity_cfg = env_cfg.get("gravity", {})
        self.j2_enabled = bool(gravity_cfg.get("enable_j2", False))

        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.earth = self.gravFactory.createEarth()
        self.earth.isCentralBody = True
        # mu is set automatically by createEarth() to the standard GM value
        # (3.986004418e14 m^3/s^2); no override needed for Stage 1.

    def attach_perturbation_models(self, spacecraft):
        if not self.j2_enabled:
            return None

        self.j2_model = J2PerturbationModel(spacecraft)
        self.sim.scSim.AddModelToTask(self.sim.taskName, self.j2_model)
        return self.j2_model
