import os

import numpy as np
import yaml


class Experiment:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.config["yearlist"] = np.arange(
            self.config["yearlist_begin"], self.config["yearlist_end"]
        )
