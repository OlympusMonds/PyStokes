__author__ = 'Luke'

import numpy as np
from pyevtk.hl import imageToVTK, pointsToVTK, VtkGroup
import glob

def generate_temporal_vtk(output_path):
    for dtype in ("mesh", "particles"):
        group = VtkGroup("{}/pseudo_temporal_{}".format(output_path, dtype))
        dfiles = sorted(glob.glob("{}/{}*".format(output_path, dtype)))
        for t, df in enumerate(dfiles):
            group.addFile(filepath = df, sim_time = t)
        group.save()


def write_mesh(output_path, timestep, domain, point_data, mesh_group, current_time):
    mpath = imageToVTK("{}/mesh{:04d}".format(output_path, timestep),
                       spacing = (domain["dx"], domain["dy"], 1.0),
                       pointData = point_data)
    mesh_group.addFile(filepath = mpath, sim_time = current_time)


def write_particles(output_path, timestep, particles, particles_group, current_time):
    ppath = pointsToVTK("{}/particles{:04d}".format(output_path, timestep), particles[:,0], particles[:,1], np.zeros_like(particles[:,0]), data = {"visc": particles[:,2]})
    particles_group.addFile(filepath = ppath, sim_time = current_time)