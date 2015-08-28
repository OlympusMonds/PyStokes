__author__ = 'Luke'

import numpy as np
from mesh import solve_stokes_momentum, solve_pressure_poisson

def apply_boundary_conditions(u, v, p):
    # Left wall
    u[:,0] = 0
    # v[:,0] = 0  # No slip
    v[:,0] = v[:,1]  # Free-slip

    # Right wall
    u[:,-1] = 0
    #v[:,-1] = 0   # No slip
    v[:,-1] = v[:,-2]   # Free slip

    # Apply boundary conditions
    # Bottom wall
    u[0,:] = -1
    v[0,:] = 0

    # Top wall
    u[-1,:] = 1
    v[-1,:] = 0


    # Internal BC
    u[21,10:21] = -0.5

    # Corner BCs

    # Bottom left
    u[0, 0] = 0
    v[0, 0] = 0.5

    # Bottom right
    u[0, -1] = -0.5
    v[0, -1] = 0

    # Top left
    u[-1, 0] = 0.5
    v[-1, 0] = 0

    # Top right
    u[-1, -1] = 0
    v[-1, -1] = -0.5

    return u, v, p


def solve_flow(u, v, dt, dx, dy, p, rho, nu, nit):
    diff = 1000.
    stepcount = 0

    while True:
        if diff < 1e-5 and stepcount >= 10:
            break
        if stepcount >= 5000:
            break

        un = u.copy()
        vn = v.copy()

        p = solve_pressure_poisson(p, rho, dx, dy, dt, u, v, nit)
        u, v = solve_stokes_momentum(u, v, p, un, vn, dt, dx, dy, rho, nu)

        u, v, p = apply_boundary_conditions(u, v, p)

        # Check if in steady-state
        udiff = np.abs((np.sum(np.abs(u[:])-np.abs(un[:])))/np.sum(np.abs(un[:])))
        vdiff = np.abs((np.sum(np.abs(v[:])-np.abs(vn[:])))/np.sum(np.abs(vn[:])))
        diff = max(udiff, vdiff)
        stepcount += 1

    print "\tstepcount: {}\tDiff: {}".format(stepcount, diff)

    return u, v, p
