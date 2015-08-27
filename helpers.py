__author__ = 'Luke'

from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import numpy as np



def interp_particles_to_mesh(particles, X, Y):
    """
    Improve to using either a weighted method, or cubic or whatever.
    Needs to be continuous over the whole field though.
    Could do cubic, and then find the nans, and use nearest to fill them?
    """
    return griddata(particles[:,:2], particles[:,2], (X, Y), method='nearest')


def interp_mesh_to_particles(particles, u, v, dx, dy):
    # Get particle coords back into data space
    # TODO: make sure safe with domains that are not 0 -> nx
    scaled_x = particles[:,0] / dx
    scaled_y = particles[:,1] / dy

    particles[:,3] = ndimage.map_coordinates(u, [scaled_y, scaled_x])
    particles[:,4] = ndimage.map_coordinates(v, [scaled_y, scaled_x])

    return particles






def calc_dt(courant, u, v, dx, dy):
    return courant * (min((dx, dy)) / max((np.max(np.absolute(u)), np.max(np.absolute(v)))))
