"""
sim_logger.py

Records simulation states, control commands, and solver metadata at each
timestep. Writes output to file for post-run analysis.

Output format (CSV, HDF5, etc.) to be decided at implementation time.
"""


class SimLogger:

    def __init__(self, output_path, config):
        """
        Parameters
        ----------
        output_path : str
            Path to the output log file.
        config : dict
            Logger configuration (fields to log, output format, etc.).
        """
        pass

    def log(self, state, command):
        """
        Record one timestep entry.

        Parameters
        ----------
        state : SimState
        command : ControlCommand
        """
        pass

    def save(self):
        """Flush and write all logged data to file."""
        pass
