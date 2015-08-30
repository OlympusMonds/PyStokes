__author__ = 'Luke'

from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import numpy as np


def interp_particles_to_mesh(particles, domain):
    """
    Improve to using either a weighted method, or cubic or whatever.
    Needs to be continuous over the whole field though.
    Could do cubic, and then find the nans, and use nearest to fill them?
    """
    X, Y = domain["grid"]
    return griddata(particles[:,:2], particles[:,2], (X, Y), method='nearest')


def interp_mesh_to_particles(particles, u, v, domain):
    # Get particle coords back into data space
    # TODO: make sure safe with domains that are not 0 -> nx
    scaled_x = particles[:,0] / domain["dx"]
    scaled_y = particles[:,1] / domain["dy"]

    particles[:,3] = ndimage.map_coordinates(u, [scaled_y, scaled_x])
    particles[:,4] = ndimage.map_coordinates(v, [scaled_y, scaled_x])

    return particles


def calc_dt(courant, u, v, domain):
    min_spacing = min((domain["dx"], domain["dy"]))
    max_u = np.max(np.absolute(u))
    max_v = np.max(np.absolute(v))
    max_vel = max(max_u, max_v)

    """
    print "Calc DT: minspacing {}, max_vel {}".format(min_spacing, max_vel)
    print "\tDT = {}".format(courant * ( min_spacing / max_vel ))
    """

    return courant * ( min_spacing / max_vel )
