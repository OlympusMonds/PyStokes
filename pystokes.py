__author__ = 'Luke'

import matplotlib.pyplot as plt
import numpy as np
import sys
from pyevtk.hl import VtkGroup

from particles import *
from helpers import calc_dt, interp_particles_to_mesh
from solvers import solve_flow
from output import write_mesh, write_particles

"""
speed = np.sqrt(u*u + v*v)

fig = plt.figure(figsize=(11,7), dpi=100)
plt.streamplot(x, y, u, v, color=speed, density=2)
plt.colorbar()
plt.xlim(0, 2.)
plt.ylim(0, 2.)
"""

def main():
    output_path = "D:\PyStokes\low_contrast"
    output_time_interval = 0.1
    output_timestep_interval = 1000
    logging = False

    xmin, xmax = 0., 2.
    ymin, ymax = 0., 2.

    nx, ny = 41, 41

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    dt = 0.0001
    nt = 5000
    max_time = 20.
    nit = 50

    c = 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X,Y = np.meshgrid(x, y)

    rho = 1

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    particles = generate_particles(nx, ny)

    timestep = 0
    current_time = 0
    dt_since_output = 0
    mesh_group = VtkGroup("{}/temporal_mesh".format(output_path))
    particles_group = VtkGroup("{}/temporal_particles".format(output_path))

    while timestep < nt and current_time < max_time:
        print current_time,
        nu = interp_particles_to_mesh(particles, X, Y)

        u, v, p = solve_flow(u, v, dt, dx, dy, p, rho, nu, nit)
        p -= np.min(p)  # set min pressure to be 0

        dt = calc_dt(0.01, u, v, dx, dy)
        current_time += dt
        dt_since_output += dt

        # Interpolate velocities back to particles
        particles = interp_mesh_to_particles(particles, u, v, dx, dy)

        # Advect particles
        particles = advect_particles_rk2(dt, particles, xmin, xmax, ymin, ymax, u, v, dx, dy)

        if timestep % output_timestep_interval == 0 or dt_since_output > output_time_interval:
            point_data = {"visc" : nu[:,:,np.newaxis],
                          "u" : u[:,:,np.newaxis],
                          "v" : v[:,:,np.newaxis]}
            write_mesh(output_path, timestep, dx, dy, point_data, mesh_group, current_time)
            write_particles(output_path, timestep, particles, particles_group, current_time)

            dt_since_output = 0

        timestep += 1

    mesh_group.save()
    particles_group.save()

    return 0


if __name__ == '__main__':
    sys.exit(main())
