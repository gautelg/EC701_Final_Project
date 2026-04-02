"""
bsk_sim.py

Basilisk SimBase setup. Defines the simulation object, a single dynamics
process, and a single dynamics task clocked at dynamics_dt.

The controller_dt is stored here for reference by other components but is
not used in the zero-control test (no controller loop is run).
"""

from Basilisk.utilities import SimulationBaseClass, macros


class BskSim:

    def __init__(self, dynamics_dt, controller_dt):
        """
        Parameters
        ----------
        dynamics_dt : float
            Basilisk dynamics integration time step (seconds).
        controller_dt : float
            Controller update interval (seconds). Stored for later use.
        """
        self.dynamics_dt = dynamics_dt
        self.controller_dt = controller_dt

        self.scSim = None
        self.dynProcess = None

        # Names used when adding models and recorders to the task
        self.processName = "dynProcess"
        self.taskName = "dynTask"

    def setup(self):
        """Initialize Basilisk SimBase, processes, and tasks."""
        self.scSim = SimulationBaseClass.SimBaseClass()

        self.dynProcess = self.scSim.CreateNewProcess(self.processName)
        dynTask = self.scSim.CreateNewTask(
            self.taskName, macros.sec2nano(self.dynamics_dt)
        )
        self.dynProcess.addTask(dynTask)

    def initialize(self):
        """Finalize module connections. Call after all models have been added."""
        self.scSim.InitializeSimulation()

    def run(self, t_end):
        """Advance the simulation to t_end (seconds)."""
        self.scSim.ConfigureStopTime(macros.sec2nano(t_end))
        self.scSim.ExecuteSimulation()

    def reset(self):
        """Reset the simulation to t=0."""
        self.scSim.ResetSimulation()
