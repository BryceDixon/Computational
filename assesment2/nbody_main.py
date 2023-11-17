"""
Assessment 2 - N-body code
Main code for the N-body algorithm

Contains:


Author: Bryce Dixon
Version: 15/11/2023    
"""


import numpy as np
from astropy.constants import G as G_grav
import matplotlib.pyplot as plt


def a(positions, masses, softening, fixed_mass = None, use_grav = True):
    """
    Function to calculate the gravitational accelerations acting on N number of particles at given positions

    Parameters
    ----------
    positions : array of floats
        2D array of floats with 3 columns representing x,y,z positions and N rows representing each particle
    masses : array of floats
        1D array of particle masses in order m_1, m_2, ..., etc
    use_grav : bool, optional
        If True, the gravitational constant G will be used in acceleration calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
    
    Returns
    -------
    a : array of floats
        2D array of accellerations with 3 columns representing x,y,z accelerations and N rows representing each particle
    """
    
    if use_grav is True:
        G = G_grav.value
    if use_grav is False:
        G = 1
        
    positions = np.array(positions, dtype=float)
    masses = np.array(masses, dtype=float)
    
    assert len(positions[0,:]) == 3, "Number of columns in position must be equal to 3, representing x,y,z positions"
    assert np.shape(masses) == (len(positions[:,0]),), "masses must be a 1D array or list of the particle masses and the number of particle masses given must match number of particles in position array"
    
    if fixed_mass is None:
        fixed_mass = np.zeros_like(masses, dtype=float)
    fixed_mass = np.array(fixed_mass)
    
    a = np.zeros_like(positions)
        
    for i in range(len(positions[:,0])):
        
        if fixed_mass[i] == 1.:
            
            a[i,:] += 0.
        
        else:
        
            for k in range(len(positions[:,0])):
            
                # calculate difference in x,y,z positions for particle k and i, will be 0 for when k=i and so not contribute to acceleration
                dif_pos = positions[k,:] - positions[i,:]
            
                # calculate r^3 using vector differences, include the softening factor
                r = ((np.linalg.norm(dif_pos))**2 + softening**2)**(3/2)
            
                # calculate the x,y,z components of acceleration and add them to the acceleration of the ith particle
                a[i,:] += ((G * masses[k])/r)*dif_pos
    
    return a 


class N_body:
    
    def __init__(self, h, pos_0, v_0, m):
        
        pos_0 = np.array(pos_0)
        v_0 = np.array(v_0)
        self.h = h
        self.pos_0 = pos_0
        self.v_0 = v_0
        self.m = m

        fig = plt.figure(figsize = (10,7))
        self.ax = fig.add_subplot(2,1,1)
        self.ax2 = fig.add_subplot(2,1,2)
    
    def verlet(self, softening, fixed_mass, use_grav):
        
        print(self.v_0)
        print(self.pos_0)
        print(a(self.pos_0, self.m, softening, fixed_mass, use_grav))
        v_step = self.v_0 + 0.5 * self.h * a(self.pos_0, self.m, softening, fixed_mass, use_grav)
        pos = self.pos_0 + self.h * v_step
        v = v_step + 0.5 * self.h * a(pos, self.m, softening, fixed_mass, use_grav)
    
        self.v = v
        self.pos = pos
        print(self.v)
        print(self.pos)
        print(a(pos, self.m, softening, fixed_mass, use_grav))
    
    def plot(self, iteration):
        
        for i in range(len(self.pos[:,0])):
            self.ax.plot([self.pos_0[i,0],self.pos[i,0]], [self.pos_0[i,1], self.pos[i,1]], color = 'C'+str(i))
        
        for i in range(len(self.pos[:,0])):
            L_0 = np.cross(self.pos_0[i,:], self.m[i] * self.v_0[i,:])
            mod_L0 = np.linalg.norm(L_0)
            L = np.cross(self.pos[i,:], self.m[i] * self.v[i,:])
            mod_L = np.linalg.norm(L)

            self.ax2.plot([iteration, iteration+1], [mod_L0,mod_L], color = 'C'+str(i))
        
        
        self.pos_0 = self.pos
        self.v_0 = self.v