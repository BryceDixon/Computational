

import numpy as np
from astropy.constants import G as G_grav
from astropy.constants import c as c_speed



def geodesic(tau, y):
    """
    Function to compute the geodesic equations for t, r, and phi given an array y, assumes theta is pi/2, and c and G are both 1.
    
    Parameters 
    ----------
    tau : Nonetype
        empty variable, exists only so the function is of correct form to be used with the RK45 integrtor
    y : array of float
        1D array of floats representing the parameters to solve the geodesic equations in the form:
        y[0] = t
        y[1] = dt
        y[2] = r
        y[3] = dr
        y[4] = phi
        y[5] = dphi
        y[6] = mass
    
    Returns
    -------
    F : array of floats
        1D array of floats of the 1st and 2nd order proper time derivatives obtained from the geodesic equations in the form:
        F[0] = dt
        F[1] = d^2t
        F[2] = dr
        F[3] = d^2r
        F[4] = dphi
        F[5] = d^2phi
        F[6] = mass change (0)
    """
    F = np.zeros_like(y, dtype=float)
    G = 1
    c = 1
    M = y[6]
    R = (2*G*M)/c**2
    F[0] = y[1]
    F[1] = -(R/((y[2]**2)*(1-(R/y[2]))))*y[3]*y[1]
    F[2] = y[3]
    F[3] = (y[2]-R)*y[5]**2 + (y[3]**2)*(R/(2*(y[2]**2)*(1-(R/y[2])))) - ((G*M)/y[2]**2)*(1-(R/y[2]))*y[1]**2
    F[4] = y[5]
    F[5] = -(2/y[2])*y[3]*y[5]
    F[6] = 0
    
    return F


def geodesic_const(tau, y):
    """
    Function to compute the geodesic equations for t, r, and phi given an array y, assumes theta is pi/2, and c and G come from the astropy constants library.
    
    Parameters 
    ----------
    tau : Nonetype
        empty variable, exists only so the function is of correct form to be used with the RK45 integrtor
    y : array of float
        1D array of floats representing the parameters to solve the geodesic equations in the form:
        y[0] = t
        y[1] = dt
        y[2] = r
        y[3] = dr
        y[4] = phi
        y[5] = dphi
        y[6] = mass
    
    Returns
    -------
    F : array of floats
        1D array of floats of the 1st and 2nd order proper time derivatives obtained from the geodesic equations in the form:
        F[0] = dt
        F[1] = d^2t
        F[2] = dr
        F[3] = d^2r
        F[4] = dphi
        F[5] = d^2phi
        F[6] = mass change (0)
    """
    F = np.zeros_like(y, dtype=float)
    G = G_grav
    c = c_speed
    M = y[6]
    R = (2*G*M)/c**2
    F[0] = y[1]
    F[1] = -(R/((y[2]**2)*(1-(R/y[2]))))*y[3]*y[1]
    F[2] = y[3]
    F[3] = (y[2]-R)*y[5]**2 + (y[3]**2)*(R/(2*(y[2]**2)*(1-(R/y[2])))) - ((G*M)/y[2]**2)*(1-(R/y[2]))*y[1]**2
    F[4] = y[5]
    F[5] = -(2/y[2])*y[3]*y[5]
    F[6] = 0
    
    return F


def initial_dt(m, r, dphi, dr, use_const):
    if use_const == False:
        G = 1
        c = 1
    if use_const == True:
        G = G_grav
        c = c_speed
    R = (2*G*m)/c**2
    dt = np.sqrt(((c**2+((r**2)*(dphi**2)))/(1-R/r))+((dr**2)/((1-R/r)**2)))
    return dt

def initial_y(t, r, phi, m, use_const = False):
    """
    Function to generate the initial y matrix to be used in the geodesic equation functions based off the conditions for a stable keplerian orbit

    Parameters
    ----------
    t : float
        initial time in seconds
    r : float
        initial radial distance from the black hole in meters
    phi : float
        initial azimuthal angle around the black hole in radians
    m : float
        mass of the black hole in kg
    use_const : bool, optional
        if False, sets G and c to 1 when required in calculations, if True, uses G and c values from the astropy constants library in calculations, defaults to False

    Returns
    -------
    y : array of floats
        1D array of the initial conditions for a stable keplerian orbit in the form:
        y[0] = t
        y[1] = dt
        y[2] = r
        y[3] = dr
        y[4] = phi
        y[5] = dphi
        y[6] = mass
    """
    y = np.zeros(shape = (7), dtype = float)
    if use_const == False:
        G = 1
    if use_const == True:
        G = G_grav
        
    dphi = (np.sqrt((G*m)/r))/r
    y[0] = t
    y[1] = initial_dt(m, r, dphi, 0, use_const)
    y[2] = r
    y[3] = 0
    y[4] = phi
    y[5] = dphi
    y[6] = m
    return y