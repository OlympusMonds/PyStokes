__author__ = 'Luke'

import numpy as np


def create_domain(domain):
    domain["dx"] = (domain["xmax"] - domain["xmin"]) / (domain["nx"] - 1)
    domain["dy"] = (domain["ymax"] - domain["ymin"]) / (domain["ny"] - 1)

    domain["x"] = np.linspace(domain["xmax"], domain["xmin"], domain["nx"])
    domain["y"] = np.linspace(domain["ymax"], domain["ymin"], domain["ny"])
    domain["grid"] = np.meshgrid(domain["x"], domain["y"])

    return domain


def create_variable_mesh(domain):
    return np.zeros((domain["ny"], domain["nx"]), dtype=np.float64)