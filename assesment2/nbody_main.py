"""
Assessment 2 - N-body code
Main code for the N-body algorithm

Contains:


Author: Bryce Dixon
Version: 15/11/2023    
"""


import numpy as np
from astropy.constants import G as G_grav


def a(positions, masses):
    """

    Parameters
    ----------
    positions : array of floats
        array of floats with 3 columns representing x,y,z positions and N rows representing each particle
    masses : array of floats
        array of particle masses in order m_1, m_2, ..., etc
    
    Returns
    -------
    a : array of floats
        array of accellerations with 3 columns representing x,y,z accelerations and N rows representing each particle
    """
    G_grav = 1
    
    positions = np.array(positions, dtype=float)
    masses = np.array(masses, dtype=float)
    a = np.zeros_like(positions)
        
    for i in range(len(positions)):
        
        for k in range(len(positions)):
            
            # calculate difference in x,y,z positions for particle k and i, will be 0 for when k=i and so not contribute to acceleration
            dif_pos = positions[k,:] - positions[i,:]
            
            # calculate r^3 using vector differences, include the softening factor
            r = ((np.linalg.norm(dif_pos))**2 + 0.00001**2)**(3/2)
            
            # calculate the x,y,z components of acceleration and add them to the acceleration of the ith particle
            a[i,:] += ((G_grav * masses[k])/r)*dif_pos
    
    return a 


class N_body:
    
    def __init__(self, h, x_0, v_0, m):
        
        x_0 = np.array(x_0)
        v_0 = np.array(v_0)
        self.h = h
        self.x_0 = x_0
        self.v_0 = v_0
        self.m = m
    
    
    def verlet(self):
        
        v_step = self.v_0 + 1/2 * self.h * a(self.x_0, self.m)
        pos = self.x_0 + self.h * v_step
        v = v_step + 1/2 * self.h * a(pos, self.m)
        