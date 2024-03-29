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
from mpl_toolkits import mplot3d


def a(positions, masses, softening, fixed_mass = None, use_grav = True):
    """
    Function to calculate the gravitational accelerations acting on N number of particles at given positions

    Parameters
    ----------
    positions : array of floats
        2D array of floats with 3 columns representing x,y,z positions and N rows representing each particle
    masses : array of floats
        1D array of particle masses in order m_1, m_2, ..., etc
    softening : float
        float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small
    fixed_mass : array of ints, optional
        1D array of 1s and 0s equal in length to the mass array, where 1 indicates the mass in that position is to be fixed and 0 indicates it is not, defaults to None 
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
    
    a = np.zeros_like(positions, dtype=float)
        
    for i in range(len(positions[:,0])):
        if fixed_mass[i] == 1.:
            a[i,:] += 0.
        else:
            for k in range(len(positions[:,0])):
                if i == k:
                    a[i,:] += 0.
                else:
                    # calculate difference in x,y,z positions for particle k and i, will be 0 for when k=i and so not contribute to acceleration
                    dif_pos = positions[k,:] - positions[i,:]
                    # calculate r^3 using vector differences, include the softening factor
                    r = (np.sqrt((np.linalg.norm(dif_pos))**2 + softening**2))**3
                    # calculate the x,y,z components of acceleration and add them to the acceleration of the ith particle
                    a[i,:] += ((G * masses[k])/r)*dif_pos
    
    return a 


def E_pot(positions, masses, softening, use_grav = True):
    """
    Function to calculate the gravitational potential energy the system at the given particle positions

    Parameters
    ----------
    positions : array of floats
        2D array of floats with 3 columns representing x,y,z positions and N rows representing each particle
    masses : array of floats
        1D array of particle masses in order m_1, m_2, ..., etc
    use_grav : bool, optional
        If True, the gravitational constant G will be used in the energy calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
    
    Returns
    -------
    E : float
        gravitational potential energy of the system
    """
    
    positions = np.array(positions, dtype=float)
    masses = np.array(masses, dtype=float)
    if use_grav is True:
        G = G_grav.value
    if use_grav is False:
        G = 1.
    
    E = 0
    
    for i in range(len(masses)):
        for k in range(i+1, len(masses)):
            dif_pos = positions[k,:] - positions[i,:]
            r = (np.linalg.norm(dif_pos)**2 + softening**2)**0.5
            E -= ((G * masses[k] * masses[i])/r)
    
    return E


def E_kin(velocities, masses):
    """
    Function to calculate the total kinetic energy of the system at the given particle velocities

    Parameters
    ----------
    velocities : array  of floats
        2D array of floats with 3 columns representing initial x,y,z velocities and N rows representing each particle
    masses : array of floats
        1D array of particle masses in order m_1, m_2, ..., etc

    Returns
    -------
    E : float
        total kinetic energy of the system
    """
    
    velocities = np.array(velocities, dtype=float)
    masses = np.array(masses, dtype=float)
    
    E = 0
    
    for i in range(len(masses)):
        E += 0.5 * masses[i] * np.dot(velocities[i,:], velocities[i,:])
    
    return E


class N_body:
    
    def __init__(self, h, pos_0, v_0, m, use_grav, fixed_mass, softening):
        """Class to perform the N-body simulation on a system of particles given the initial positions and velocities of the particles

        Parameters
        ----------
        h : float
            size of the time step to use in velocity verlet method
        pos_0 :  array of floats
            2D array of floats with 3 columns representing initial x,y,z positions and N rows representing each particle
        v_0 : array  of floats
            2D array of floats with 3 columns representing initial x,y,z velocities and N rows representing each particle
        m : array of floats
            1D array of particle masses in order m_1, m_2, ..., etc
        use_grav : bool, optional
            If True, the gravitational constant G will be used in acceleration and potential energy calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
        fixed_mass : array of ints, optional
            1D array of 1s and 0s equal in length to the mass array, where 1 indicates the mass in that position is to be fixed and 0 indicates it is not, defaults to None 
        """
        
        pos_0 = np.array(pos_0)
        v_0 = np.array(v_0)
        self.h = h
        self.pos_0 = pos_0
        self.v_0 = v_0
        self.m = m
        self.use_grav = use_grav
        self.fixed_mass = fixed_mass
        self.softening = softening
        
        self.a_0 = a(self.pos_0, self.m, self.softening, self.fixed_mass, self.use_grav)
            

        fig = plt.figure(figsize = (10,16))
        if np.any(self.v_0[:,2]) == True:
            self.ax = fig.add_subplot(3,1,1, projection = '3d')
            self.ax.set_zlabel('Z Position')
        else:
            self.ax = fig.add_subplot(3,1,1)
        self.ax2 = fig.add_subplot(3,1,2)
        self.ax3 = fig.add_subplot(3,1,3)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Orbital Paths for the N-body system')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Angular Momentum Error (%)')
        self.ax2.set_title('Percentage Angular Momentum Error of the N-body system against iterations')
        self.ax3.set_xlabel('Iteration')
        self.ax3.set_ylabel('Energy Error (%)')
        self.ax3.set_title('Percentage Energy Error of the N-body system against iterations')
    
    def verlet(self):
        """Function to calculate the new velocity and position of the masses using the velocity verlet algorithm

        Parameters
        ----------
        softening : float
            float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small
        """
        
        v_step = self.v_0 + 0.5 * self.h * self.a_0
        pos = self.pos_0 + self.h * v_step
        a_cur = a(pos, self.m, self.softening, self.fixed_mass, self.use_grav)
        v = v_step + 0.5 * self.h * a_cur
        
        if 1 in self.fixed_mass:
            pass
        else:
            com = 0
            msum = 0
            com_0 = 0
            for i in range(len(self.m)):
                com_0 += self.m[i] * self.pos_0[i,:]
                com += self.m[i] * pos[i,:]
                msum += self.m[i]
            comdif = (com/msum) - (com_0/msum)
            pos = pos - comdif
    
        self.v = v
        self.pos = pos
        self.a_0 = a_cur
    
    def plot(self, iteration):
        """Function to plot the positions of the masses over the orbits, the angular momentum of the system over the orbits, and the energy of the system over the orbits

        Parameters
        ----------
        iteration : int
            current iteration of the N-body code
        """
        
        L = 0
        L_0 = 0
        
        for i in range(len(self.pos[:,0])):
            if self.fixed_mass[i] == 1 and iteration == 0:
                self.ax.scatter(self.pos_0[i,0], self.pos_0[i,1], color = 'C'+str(i), s = 5)
            else:
                if np.any(self.v_0[:,2]) == True or np.any(self.pos_0[:,2]) == True:
                    self.ax.plot3D([self.pos_0[i,0],self.pos[i,0]], [self.pos_0[i,1], self.pos[i,1]], [self.pos_0[i,2], self.pos[i,2]], color = 'C'+str(i))
                else:
                    self.ax.plot([self.pos_0[i,0],self.pos[i,0]], [self.pos_0[i,1], self.pos[i,1]], color = 'C'+str(i))
            
            L_0 += np.cross(self.pos_0[i,:], self.m[i] * self.v_0[i,:])
            L += np.cross(self.pos[i,:], self.m[i] * self.v[i,:])
        
        E_0 = E_pot(self.pos_0, self.m, self.softening, self.use_grav) + E_kin(self.v_0, self.m)
        E = E_pot(self.pos, self.m, self.softening, self.use_grav) + E_kin(self.v, self.m)
        
        mod_L0 = np.linalg.norm(L_0)
        mod_L = np.linalg.norm(L)
        
        if iteration == 0:
            self.E_initial = E_0
            self.L_initial = mod_L0
        
        if self.L_initial != 0:
            errmod_L0 = np.sqrt(((mod_L0 - self.L_initial)/self.L_initial)**2)*100
            errmod_L = np.sqrt(((mod_L - self.L_initial)/self.L_initial)**2)*100
            self.ax2.plot([iteration, iteration+1], [errmod_L0,errmod_L], color = 'C0')
        else:
            self.ax2.plot([iteration, iteration+1], [mod_L0,mod_L], color = 'C0')
            self.ax2.set_ylabel('Angular Momentum L')
            self.ax2.set_title('Angular Momentum of the N-body system against iterations')
        
        errE_0 = np.sqrt(((E_0 - self.E_initial)/self.E_initial)**2)*100
        errE = np.sqrt(((E - self.E_initial)/self.E_initial)**2)*100
        self.ax3.plot([iteration, iteration+1], [errE_0,errE], color = 'C0')
            
        self.pos_0 = self.pos
        self.v_0 = self.v
    


def nbody(iterations, h, position_init, velocity_init, masses, use_grav = True, fixed_mass = None, softening = 0.001, save_folder = None, savefilename = None):
    """Function to perform the full N-body simulation for a given set of masses with their positions and velocities for a given number of iterations

    Parameters
    ----------
    iterations : int
        number of iterations to run the N-body code for
    h : float
        size of the time step to use in velocity verlet method
    position_init : array of floats
        2D array of floats with 3 columns representing initial x,y,z positions and N rows representing each particle
    velocity_init : array of floats
        2D array of floats with 3 columns representing initial x,y,z velocities and N rows representing each particle
    masses : array of floats
        1D array of particle masses in order m_1, m_2, ..., etc
    use_grav : bool, optional
        If True, the gravitational constant G will be used in acceleration and potential energy calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
    fixed_mass : array of ints, optional
        1D array of 1s and 0s equal in length to the mass array, where 1 indicates the mass in that position is to be fixed and 0 indicates it is not, defaults to None
    softening : float, optional
        float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small, defaults to 0.001
    """
    
    if fixed_mass is None:
        fixed_mass = np.zeros_like(masses, dtype=float)
    fixed_mass = np.array(fixed_mass)
        
    N = N_body(h, position_init, velocity_init, masses, use_grav, fixed_mass, softening)
    for i in range(iterations):
        N.verlet()
        N.plot(i)
    
    # save the plot provided  save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved      
    if save_folder is not None and savefilename is not None:
        plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
    elif save_folder is None and savefilename is None:
        # no message displayed if both save_folder and savefilename are none, it's assumed the user did not intend to save the plot
        pass
    else:
        print("Figure not saved, you need to provide both savefilename and save_folder to save the figure") 