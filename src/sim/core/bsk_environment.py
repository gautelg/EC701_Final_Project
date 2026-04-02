"""
bsk_environment.py

Adds environmental models to the Basilisk scene.

Stage 1 (this file): point-mass Earth gravity only.
Stage 3 (future): J2, solar radiation pressure, atmospheric drag.

The gravFactory object is the authoritative source of gravity body data
(including Earth's mu) used by BskSpacecraft to set initial orbital elements.
"""

from Basilisk.utilities import simIncludeGravBody


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

    def setup(self):
        """Create Earth point-mass gravity body."""
        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.earth = self.gravFactory.createEarth()
        self.earth.isCentralBody = True
        # mu is set automatically by createEarth() to the standard GM value
        # (3.986004418e14 m^3/s^2); no override needed for Stage 1.
