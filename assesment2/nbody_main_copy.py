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
    
    def __init__(self, h, pos_0, v_0, m, use_grav, fixed_mass, softening, tot_iterations, plotting_num):
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
        softening : float
            float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small 
        tot_iterations : int
            total number of iterations to run the N_body code for
        plotting_num : int
            number of points to plot on the graphs, this will be used to determine how often the results are saved to an array for plotting, this may vary around the given value depending on the given value and total iterations

        
        Raises
        ------
        assertionError:
            Raised if total number of iterations is smaller than 2 times plotting_num
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
        
        assert tot_iterations >= (plotting_num * 2), "Total iterations must be larger or equal to 2 times plotting_num"
        self.step = 0
        self.interval = int(tot_iterations/plotting_num)
        self.plotting_num = len(np.arange(0,tot_iterations,self.interval))
        self.pos_array = np.zeros(shape = (self.plotting_num, len(self.m), 3))
        self.v_array = np.zeros(shape = (self.plotting_num, len(self.m), 3))
        self.tot_iterations = tot_iterations
        self.iteration_list = []
        self.it_step = 0
    
    def verlet(self):
        """Function to calculate the new velocity and position of the masses using the velocity verlet algorithm
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
    
    def save(self, iteration):
        """Function to save the specific pos and v values to an array for plotting depending on the number of plotted points desired

        Parameters
        ----------
        iteration : int
            current iteration of the N-body code
        """
        
        if iteration == 0:
            self.pos_array[self.it_step,:,:] = self.pos_0
            self.v_array[self.it_step,:,:] = self.v_0
            self.step += int(self.interval)
            self.iteration_list.append(iteration)
            self.it_step += 1
        elif iteration == self.step:
            self.pos_array[self.it_step,:,:] = self.pos
            self.v_array[self.it_step,:,:] = self.v
            self.step += int(self.interval)
            self.iteration_list.append(iteration)
            self.it_step += 1
        elif iteration == (self.tot_iterations - 1):
            # add on the last iteration if it hasn't already ended
            self.pos_array = np.concatenate([self.pos_array, self.pos[None]]) 
            self.v_array = np.concatenate([self.v_array, self.v[None]])
            self.iteration_list.append(iteration)
            self.plotting_num += 1
        else:
            pass
        
        self.pos_0 = self.pos
        self.v_0 = self.v
    
    def plot(self):
        """Function to plot the positions of the masses over the orbits, the angular momentum of the system over the orbits, and the energy of the system over the orbits
        """
        
        fig = plt.figure(figsize = (10,16), dpi = 200)
        if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
            ax = fig.add_subplot(3,1,1, projection = '3d')
            ax.set_zlabel('Z Position')
        else:
            ax = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Orbital Paths for the N-body system')
        ax2.set_xlabel('Iteration')
        ax3.set_xlabel('Iteration')
        
        L = np.zeros(self.plotting_num, dtype = float)
        E = np.zeros(self.plotting_num, dtype = float)
        
        for i in range(len(self.m)):
            if self.fixed_mass[i] == 1:
                ax.scatter(self.pos_array[0,i,0], self.pos_array[0,i,1], color = 'C'+str(i), s = 5)
            else:
                if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
                    ax.plot3D(self.pos_array[:,i,0], self.pos_array[:,i,1], self.pos_array[:,i,2], color = 'C'+str(i))
                else:
                    ax.plot(self.pos_array[:,i,0], self.pos_array[:,i,1], color = 'C'+str(i))
        
        for l in range(self.plotting_num):
            L_cur = 0
            for i in range(len(self.m)):
                L_cur += np.cross(self.pos_array[l,i,:], self.m[i] * self.v_array[l,i,:])
            L[l] = np.linalg.norm(L_cur)
        
        for i in range(self.plotting_num):
            E[i] = E_pot(self.pos_array[i,:,:], self.m, self.softening, self.use_grav) + E_kin(self.v_array[i,:,:], self.m)
        
        if L[0] != 0:
            errmod_L = np.zeros(self.plotting_num, dtype = float)
            for i in range(self.plotting_num):
                errmod_L[i] = np.sqrt(((L[i] - L[0])/L[0])**2)*100
            ax2.plot(self.iteration_list, errmod_L, color = 'C0')
            ax2.set_ylabel('Angular Momentum Error (%)')
            ax2.set_title('Percentage Angular Momentum Error of the N-body system against iterations')
        else:
            ax2.plot(self.iteration_list, L, color = 'C0')
            ax2.set_ylabel('Angular Momentum L')
            ax2.set_title('Angular Momentum of the N-body system against iterations')

        if E[0] != 0:
            errE = np.zeros(self.plotting_num, dtype = float)
            for i in range(self.plotting_num):
                errE[i] = np.sqrt(((E[i] - E[0])/E[0])**2)*100
            ax3.plot(self.iteration_list, errE, color = 'C0')
            ax3.set_ylabel('Energy Error (%)')
            ax3.set_title('Percentage Energy Error of the N-body system against iterations')
        else:
            ax3.plot(self.iteration_list, E, color = 'C0')
            ax3.set_ylabel('Energy')
            ax3.set_title('Energy of the N-body system against iterations')
            
    


def nbody(iterations, h, position_init, velocity_init, masses, use_grav = True, fixed_mass = None, softening = 0.001, plotting_num = 1000, save_folder = None, savefilename = None):
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
    plotting_num : int, optional
        number of points to plot on the graphs, this will be used to determine how often the results are saved to an array for plotting, this may vary around the given value depending on the given value and total iterations, defaults to 1000
    save_folder: str, optional
        directory to save the convergence plot to, defaults to None
    savefilename: str, optional
        filename for the convergence plot, defaults to None
    """
    
    if fixed_mass is None:
        fixed_mass = np.zeros_like(masses, dtype=float)
    fixed_mass = np.array(fixed_mass)
        
    N = N_body(h, position_init, velocity_init, masses, use_grav, fixed_mass, softening, iterations, plotting_num)
    for i in range(iterations):
        N.verlet()
        N.save(i)
    N.plot()
    
    # save the plot provided  save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved      
    if save_folder is not None and savefilename is not None:
        plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
    elif save_folder is None and savefilename is None:
        # no message displayed if both save_folder and savefilename are none, it's assumed the user did not intend to save the plot
        pass
    else:
        print("Figure not saved, you need to provide both savefilename and save_folder to save the figure") 