__author__ = 'Luke'

import sobol
import numpy as np
from helpers import interp_mesh_to_particles

def generate_particles(nx, ny, particles_per_cell = 30, seed = 112):
    total_particles = nx * ny * particles_per_cell
    particles = []

    # Particle format is: x, y, visc, xvel, yvel

    for _ in xrange(total_particles):
        xy, seed = sobol.i4_sobol(2, seed)
        xy = xy * 2  # This needs to be xmax and xmin, because the sobol returns from 0 - 1.

        # Custom rheology
        visc = 0.1 if xy[0] > 1. else 0.025

        particles.append([xy[0], xy[1], visc, 0., 0.])

    particles = np.array(particles)

    return particles


def advect_particles(dt, particles, xmin, xmax, ymin, ymax):
    # Crappy Euler(?) method:
    # x+1 = x + v*dt
    particles[:,0] = particles[:,0] + (particles[:,3] * dt)
    particles[:,1] = particles[:,1] + (particles[:,4] * dt)

    return particles


def advect_particles_rk2(dt, particles, xmin, xmax, ymin, ymax, u, v, dx, dy):
    """
    Runge-Kutta 2nd order
    Make a copy of the particles, and advect them with x_t+1 = x_t + v*dt
    EXCEPT that we use dt/2, to go half the distance.
    Then find out the velocity at this new location, and use that to advect
    the particle.
    """
    particles2 = particles.copy()
    particles2[:,0] = particles[:,0] + (particles[:,3] * dt / 2.)
    particles2[:,1] = particles[:,1] + (particles[:,4] * dt / 2.)

    particles2 = interp_mesh_to_particles(particles2, u, v, dx, dy)

    particles[:,0] = particles[:,0] + (particles2[:,3] * dt)
    particles[:,1] = particles[:,1] + (particles2[:,4] * dt)

    return particles

def advect_particles_rk2_ralsons(dt, particles, xmin, xmax, ymin, ymax, u, v, dx, dy):
    """
    """
    particles2 = particles.copy()
    particles2[:,0] = particles[:,0] + (particles[:,3] * dt / 2.)
    particles2[:,1] = particles[:,1] + (particles[:,4] * dt / 2.)

    particles2 = interp_mesh_to_particles(particles2, u, v, dx, dy)

    particles[:,0] = particles[:,0] + (1./3. * particles[:,3] + 2./3. * particles2[:,3]) * dt
    particles[:,1] = particles[:,1] + (1./3. * particles[:,4] + 2./3. * particles2[:,4]) * dt

    return particles