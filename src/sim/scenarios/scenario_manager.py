"""
scenario_manager.py

Scenario and Experiment Manager.

Loads a scenario YAML config, sets initial conditions in Basilisk, drives
the closed-loop execution, and manages run lifecycle (start, step, stop).
"""


class ScenarioManager:

    def __init__(self, scenario_config_path, sim, adapter, logger):
        """
        Parameters
        ----------
        scenario_config_path : str
            Path to the scenario YAML file.
        sim : BskSim
            The Basilisk simulation instance.
        adapter : ControllerAdapter
            The controller adapter instance.
        logger : SimLogger
            The logger instance.
        """
        pass

    def load(self):
        """Load and validate the scenario YAML config."""
        pass

    def initialize(self):
        """Apply initial conditions to the Basilisk sim."""
        pass

    def run(self):
        """Drive the closed loop from t=0 to t_end."""
        pass
