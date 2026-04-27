"""
Analytic J2 perturbation model for Basilisk dynamics runs.

This Python SysModel updates external-force effectors every dynamics step so
both the target and chaser feel a J2 acceleration in addition to the existing
point-mass gravity model.
"""

from Basilisk.architecture import sysModel
from Basilisk.utilities import orbitalMotion
import numpy as np


class J2PerturbationModel(sysModel.SysModel):

    def __init__(self, spacecraft):
        super().__init__()
        self.ModelTag = "j2PerturbationModel"
        self.spacecraft = spacecraft

    @staticmethod
    def _j2_force_n(state_msg, mass_kg):
        r_m = np.asarray(state_msg.r_BN_N, dtype=float)
        accel_km_s2 = orbitalMotion.jPerturb(
            r_m / 1000.0,
            2,
            "CELESTIAL_EARTH",
        )
        accel_m_s2 = 1000.0 * np.asarray(accel_km_s2, dtype=float)
        return mass_kg * accel_m_s2

    def UpdateState(self, CurrentSimNanos):
        tgt = self.spacecraft.scTarget.scStateOutMsg.read()
        chs = self.spacecraft.scChaser.scStateOutMsg.read()

        self.spacecraft.targetEnvExtFT.extForce_N = self._j2_force_n(
            tgt,
            self.spacecraft.target_mass,
        ).tolist()
        self.spacecraft.chaserEnvExtFT.extForce_N = self._j2_force_n(
            chs,
            self.spacecraft.chaser_mass,
        ).tolist()
        self.spacecraft.targetEnvExtFT.extTorquePntB_B = [0.0, 0.0, 0.0]
        self.spacecraft.chaserEnvExtFT.extTorquePntB_B = [0.0, 0.0, 0.0]

        return super().UpdateState(CurrentSimNanos)
