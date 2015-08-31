__author__ = 'Luke'

import sobol
import numpy as np
from helpers import interp_mesh_to_particles

def generate_particles(domain, particles_per_cell = 30, seed = 112):
    total_particles = domain["nx"] * domain["ny"] * particles_per_cell
    particles = []

    # Particle format is: x, y, visc, xvel, yvel

    for _ in xrange(total_particles):
        xy, seed = sobol.i4_sobol(2, seed)
        xy = xy * 2  # This needs to be xmax and xmin, because the sobol returns from 0 - 1.

        # Custom rheology
        visc = 0.1 if xy[0] > 1. else 0.5

        particles.append([xy[0], xy[1], visc, 0., 0.])

    particles = np.array(particles)

    return particles


def check_particles(particles, xmin, xmax, ymin, ymax):
    """
    Check to make sure that all particles are OK. Are they outside the domain, are they motionless, etc.
    :param particles:
    :return: particles
    """
    lt_than_xmin = particles[:,0] < xmin
    gt_than_xmax = particles[:,0] > xmax
    lt_than_ymin = particles[:,1] < ymin
    gt_than_ymax = particles[:,1] > ymax

    for i in (lt_than_xmin, gt_than_xmax, lt_than_ymin, gt_than_ymax):
        if np.any(i):
            print "Needed to fix positions"

    particles[lt_than_xmin][:,0] = xmin
    particles[gt_than_xmax][:,0] = xmax
    particles[lt_than_ymin][:,1] = ymin
    particles[gt_than_ymax][:,1] = ymax

    return particles


def advect_particles(dt, particles):
    # Crappy Euler(?) method:
    # x+1 = x + v*dt
    particles[:,0] = particles[:,0] + (particles[:,3] * dt)
    particles[:,1] = particles[:,1] + (particles[:,4] * dt)

    return particles


def advect_particles_rk2(dt, particles, u, v, domain):
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

    particles2 = interp_mesh_to_particles(particles2, u, v, domain)

    particles[:,0] = particles[:,0] + (particles2[:,3] * dt)
    particles[:,1] = particles[:,1] + (particles2[:,4] * dt)

    return particles

def advect_particles_rk2_ralsons(dt, particles, u, v, domain):
    """
    """
    particles2 = particles.copy()
    particles2[:,0] = particles[:,0] + (particles[:,3] * dt / 2.)
    particles2[:,1] = particles[:,1] + (particles[:,4] * dt / 2.)

    particles2 = interp_mesh_to_particles(particles2, u, v, domain)

    particles[:,0] = particles[:,0] + (1./3. * particles[:,3] + 2./3. * particles2[:,3]) * dt
    particles[:,1] = particles[:,1] + (1./3. * particles[:,4] + 2./3. * particles2[:,4]) * dt

    return particles