

import numpy as np
from astropy.constants import G as G_grav
from astropy.constants import c as c_speed



def geodesic(tau, y):
    """
    Function to compute the geodesic equations for t, r, and phi given an array y, assumes theta is pi/2, and c and G are both 1.
    
    Parameters 
    ----------
    tau : Nonetype
        empty variable, exists only so the function is of correct form to be used with the RK45 integrator
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
        empty variable, exists only so the function is of correct form to be used with the RK45 integrator
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


def initial_dt(m, r, dphi, dr, use_const = False, massless = False):
    """
    Function to generate the initial time velocity to be used as inputs for the first set of geodesic equations in the integration

    Parameters
    ----------
    m : float
        mass of the black hole in kg
    r : float
        radial distance from the black hole in meters
    dphi : float
        angular velocity of the particle in rad/s
    dr : float
        radial velocity of the particle in m/s
    use_const : bool, optional
        if False, sets G and c to 1 when required in calculations, if True, uses G and c values from the astropy constants library in calculations, defaults to False

    Returns
    -------
    dt : float
        time velocity of the particle in m/s
    """
    if use_const == False:
        G = 1
        c = 1
    if use_const == True:
        G = G_grav
        c = c_speed
    R = (2*G*m)/c**2
    if massless == False:
        dt = np.sqrt(((c**2+((r**2)*(dphi**2)))/(1-R/r))+((dr**2)/((1-R/r)**2)))
    if massless == True:
        dt = c
    return dt

def initial_y(t, r, phi, m, use_const = False, massless = False):
    """
    Function to generate the initial y matrix to be used in the geodesic equation functions based off the conditions for a stable circular keplerian orbit

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
        1D array of the initial conditions for a stable circulalr keplerian orbit in the form:
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
    y[1] = initial_dt(m, r, dphi, 0, use_const, massless)
    y[2] = r
    y[3] = 0
    y[4] = phi
    y[5] = dphi
    y[6] = m
    return y


def newtonian(tau, y):
    """
    Function to compute the newtonian accelerations and velocities for a particle given an initial array y of x and y positions and velocities, takes value for G as 1. 

    Parameters
    ----------
    tau : Nonetype
        empty variable, exists only so the function is of correct form to be used with the RK45 integrator
    y : array of float
        1D array of floats representing the parameters for newtonian gravititaiton in the form:
        y[0] = x
        y[1] = dx
        y[2] = y
        y[3] = dy
        y[4] = mass
    
    Returns
    -------
    F : array of float
        1D array of the 1st and 2nd order time derivatives due to newtonian accelaration in the form:
        F[0] = dx
        F[1] = d^2x
        F[2] = dy
        F[3] = d^2
        F[4] = mass change (0)
    """
    
    G = 1
    pos = np.array([y[0], y[2]])
    r = np.linalg.norm(pos)
    a = (-(G * y[4])/(r**3)) * pos
    
    F = np.zeros_like(y, dtype=float)
    F[0] = y[1]
    F[1] = a[0]
    F[2] = y[3]
    F[3] = a[1]
    F[4] = 0
    
    return F


def newtonian_const(tau, y):
    """
    Function to compute the newtonian accelerations and velocities for a particle given an initial array y of x and y positions and velocities, takes value for G from the astropy constants library. 

    Parameters
    ----------
    tau : Nonetype
        empty variable, exists only so the function is of correct form to be used with the RK45 integrator
    y : array of float
        1D array of floats representing the parameters for newtonian gravititaiton in the form:
        y[0] = x
        y[1] = dx
        y[2] = y
        y[3] = dy
        y[4] = mass
    
    Returns
    -------
    F : array of float
        1D array of the 1st and 2nd order time derivatives due to newtonian accelaration in the form:
        F[0] = dx
        F[1] = d^2x
        F[2] = dy
        F[3] = d^2
        F[4] = mass change (0)
    """
    
    G = G_grav
    pos = np.array([y[0], y[2]])
    r = np.linalg.norm(pos)
    a = (-(G * y[4])/(r**3)) * pos
    
    F = np.zeros_like(y, dtype=float)
    F[0] = y[1]
    F[1] = a[0]
    F[2] = y[3]
    F[3] = a[1]
    F[4] = 0
    
    return F
    
    