__author__ = 'Luke'

import numpy as np


def solve_pressure_poisson(p, rho, dx, dy, dt, u, v, nit):
    b = np.empty_like(p)
    two_dx = 2. * dx
    two_dy = 2. * dy

    u_i1_j =  u[1:-1, 2:]
    u_in1_j = u[1:-1, :-2]
    u_i_j1 =  u[2:, 1:-1]
    u_i_jn1 = u[:-2, 1:-1]

    v_i_j1 =  v[2:, 1:-1]
    v_i_jn1 = v[:-2, 1:-1]
    v_i1_j =  v[1:-1, 2:]
    v_in1_j = v[1:-1, :-2]

    b[1:-1, 1:-1] = 1/dt * \
                    (((u_i1_j - u_in1_j) / two_dx) + ((v_i_j1 - v_i_jn1) / two_dy)) - \
                    np.power(((u_i1_j - u_in1_j) / two_dx), 2.) - \
                    2. * \
                    (((u_i_j1 - u_i_jn1) / two_dy) * ((v_i1_j - v_in1_j) / two_dx)) - \
                    np.power(((v_i_j1 - v_i_jn1) / two_dy), 2.)

    # Non-p dependant terms
    dx2 = dx * dx
    dy2 = dy * dy
    part2 = ((rho * dx2 * dy2) / (2 * (dx2 + dy2))) * b[1:-1,1:-1]

    for its in range(nit):
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2 + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2 ) / \
                       (2 * (dx2 + dy2)) - part2

        p[0,:] = p[1,:]    # dp/dy = 0 at y = 0
        p[-1,:] = p[-2,:]  # dp/dy = 0 at y = 2
        p[:,0] = p[:,1]    # dp/dx = 0 at x = 0
        p[:,-1] = p[:,-2]  # dp/dx = 0 at x = 2

    return p


def solve_momentum(u, v, p, un, vn, dt, dx, dy, rho, nu):
    u[1:-1,1:-1] = un[1:-1,1:-1] - \
                   un[1:-1,1:-1] * (dt / dx) * (un[1:-1,1:-1] - un[1:-1,:-2]) - \
                   vn[1:-1,1:-1] * (dt / dy) * (un[1:-1,1:-1] - un[:-2,1:-1]) - \
                   (dt / (rho * 2 * dx)) * (p[1:-1,2:] - p[1:-1,:-2]) + \
                   nu[1:-1,1:-1] * (((dt/dx**2) * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])) +
                                    ((dt/dy**2) * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]))
                                   )

    v[1:-1,1:-1] = vn[1:-1,1:-1] - \
                   un[1:-1,1:-1] * (dt / dx) * (vn[1:-1,1:-1] - vn[1:-1,:-2]) - \
                   vn[1:-1,1:-1] * (dt / dy) * (vn[1:-1,1:-1] - vn[:-2,1:-1]) - \
                   (dt / (rho * 2 * dy)) * (p[2:,1:-1] - p[:-2,1:-1]) + \
                   nu[1:-1,1:-1] * (((dt/dx**2) * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])) +
                                    ((dt/dy**2) * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]))
                                   )
    return u, v


def solve_stokes_momentum(u, v, p, F, un, vn, dt, dx, dy, rho, nu):
    un_i_j =   un[1:-1, 1:-1]
    un_i1_j =  un[1:-1, 2:]
    un_in1_j = un[1:-1, :-2]
    un_i_j1 =  un[2:, 1:-1]
    un_i_jn1 = un[:-2, 1:-1]

    vn_i_j =   vn[1:-1, 1:-1]
    vn_i1_j =  vn[1:-1, 2:]
    vn_in1_j = vn[1:-1, :-2]
    vn_i_j1 =  vn[2:, 1:-1]
    vn_i_jn1 = vn[:-2, 1:-1]

    p_i1_j = p[1:-1,2:]
    p_in1_j = p[1:-1,:-2]

    nu_i_j = nu[1:-1,1:-1]

    u[1:-1,1:-1] = un_i_j - (dt / (rho * 2. * dx)) * (p_i1_j - p_in1_j) + \
                   nu_i_j * ((dt / dx) * (un_i1_j - 2.*un_i_j + un_in1_j) +
                             (dt / dy) * (un_i_j1 - 2.*un_i_j + un_i_jn1)) + \
                   F[0] * dt

    v[1:-1,1:-1] = vn_i_j - (dt / (rho * 2. * dx)) * (p_i1_j - p_in1_j) + \
                   nu_i_j * ((dt / dx) * (vn_i1_j - 2.*vn_i_j + vn_in1_j) +
                             (dt / dy) * (vn_i_j1 - 2.*vn_i_j + vn_i_jn1)) + \
                   F[1] * dt
    return u, v


def apply_boundary_conditions(u, v, p):
    # Left wall
    u[:,0] = 0
    # v[:,0] = 0  # No slip
    v[:,0] = v[:,1]  # Free-slip

    # Right wall
    u[:,-1] = 0
    # v[:,-1] = 0   # No slip
    v[:,-1] = v[:,-2]   # Free slip

    # Apply boundary conditions
    # Bottom wall
    u[0,1:-2] = -1
    v[0,1:-2] = 0

    # Top wall
    u[-1,1:-2] = 1
    v[-1,1:-2] = 0


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


def solve_flow(u, v, dt, p, rho, nu, nit, domain):
    diff = 1000.
    stepcount = 0

    while True:
        if diff < 1e-5 and stepcount >= 10:
            break
        if stepcount >= 5000:
            break

        if np.any(np.isnan(u)):
            print "u nan"
            print stepcount
            sys.exit()
        if np.any(np.isnan(v)):
            print "v nan"
            sys.exit()
        if np.any(np.isnan(p)):
            print "p nan"
            sys.exit()

        un = u.copy()
        vn = v.copy()

        if np.abs(u).max() > 1.:
            print stepcount

        p = solve_pressure_poisson(p, rho, domain["dx"], domain["dy"], dt, u, v, nit)
        u, v = solve_stokes_momentum(u, v, p, domain["gravity"], un, vn, dt, domain["dx"], domain["dy"], rho, nu)

        u, v, p = apply_boundary_conditions(u, v, p)

        # Check if in steady-state
        udenom = np.sum(np.abs(un[:]))
        udiff = np.abs((np.sum(np.abs(u[:])-np.abs(un[:])))/udenom) if udenom != 0. else 1.
        vdenom = np.sum(np.abs(vn[:]))
        vdiff = np.abs((np.sum(np.abs(v[:])-np.abs(vn[:])))/vdenom) if vdenom != 0. else 1.
        diff = max(udiff, vdiff)
        stepcount += 1

    print "\tstepcount: {}\tDiff: {}".format(stepcount, diff)

    return u, v, p
