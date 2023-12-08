"""
Assessment 2 - N-body code
Main code for the N-body algorithm

Contains:
    accel function
    E_pot function
    E_kin function
    N_body class
        verlet function
        save function
        plot function
        return_period funciton
        return_semi_major_axis function
    nbody function

Author: Bryce Dixon
Version: 07/12/2023    
"""


import numpy as np
from astropy.constants import G as G_grav
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def accel(positions, masses, softening, use_grav = True):
    """
    Function to calculate the gravitational accelerations acting on N number of masses at given positions

    Parameters
    ----------
    positions : array of floats
        2D array of floats with 3 columns representing x,y,z positions and N rows representing each mass
    masses : array of floats
        1D array of object masses in order m_1, m_2, ..., etc
    softening : float
        float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small
    use_grav : bool, optional
        If True, the gravitational constant G will be used in acceleration calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
    
    Returns
    -------
    a : array of floats
        2D array of accellerations with 3 columns representing x,y,z accelerations and N rows representing each mass
    """
    
    if use_grav is True:
        G = G_grav.value
    if use_grav is False:
        G = 1
        
    positions = np.array(positions, dtype=float)
    masses = np.array(masses, dtype=float)
    
    assert len(positions[0,:]) == 3, "Number of columns in position must be equal to 3, representing x,y,z positions"
    assert np.shape(masses) == (len(positions[:,0]),), "masses must be a 1D array or list of the object masses and the number of masses given must match number of masses in position array"
    
    a = np.zeros_like(positions, dtype=float)
        
    for i in range(len(positions[:,0])):
        for k in range(len(positions[:,0])):
            # calculate difference in x,y,z positions for mass k and i, will be 0 for when k=i and so not contribute to acceleration
            dif_pos = positions[k,:] - positions[i,:]
            # calculate r^3 using vector differences, include the softening factor
            r = (np.sqrt((np.linalg.norm(dif_pos))**2 + softening**2))**3
            # calculate the x,y,z components of acceleration and add them to the acceleration of the ith mass
            a[i,:] += ((G * masses[k])/r)*dif_pos
    
    return a 


def E_pot(positions, masses, softening, use_grav = True):
    """
    Function to calculate the gravitational potential energy of the system at the given mass positions

    Parameters
    ----------
    positions : array of floats
        2D array of floats with 3 columns representing x,y,z positions and N rows representing each mass
    masses : array of floats
        1D array of object masses in order m_1, m_2, ..., etc
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
    Function to calculate the total kinetic energy of the system at the mass velocities

    Parameters
    ----------
    velocities : array  of floats
        2D array of floats with 3 columns representing initial x,y,z velocities and N rows representing each mass
    masses : array of floats
        1D array of object masses in order m_1, m_2, ..., etc

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
    
    def __init__(self, h, pos_0, v_0, m, use_grav, softening, tot_iterations, plotting_num, return_period, per_tolerance):
        """Class to perform the N-body simulation on a system of masses given the initial positions and velocities of the masses

        Parameters
        ----------
        h : float
            size of the time step to use in velocity verlet method
        pos_0 :  array of floats
            2D array of floats with 3 columns representing initial x,y,z positions and N rows representing each mass
        v_0 : array  of floats
            2D array of floats with 3 columns representing initial x,y,z velocities and N rows representing each mass
        m : array of floats
            1D array of object masses in order m_1, m_2, ..., etc
        use_grav : bool, optional
            If True, the gravitational constant G will be used in acceleration and potential energy calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
        softening : float
            float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small 
        tot_iterations : int
            total number of iterations to run the N_body code for
        plotting_num : int
            number of points to plot on the graphs, this will be used to determine how often the results are saved to an array for plotting, this may vary around the given value depending on the given value and total iterations
        return_period : bool
            if True, sets up and performs the calculations for the period of the N-body system, if False, does not calculate a period
        per_tolerance : float
            the tolerance on the period estimate, should either be None to set to 0.5% or a sensible value chosen based on the initial positions
        
        Raises
        ------
        assertionError:
            Raised if pos_0 does not have 3 columns
        assertionError:
            Raised if pos_0 and v_0 are different shapes
        assertionError:
            Raised if m is not the same length as the number or rows in pos_0
        assertionError:
            Raised if total number of iterations is smaller than 2 times plotting_num
        """
        
        
        self.h = h
        self.pos_0 = np.array(pos_0)
        self.v_0 = np.array(v_0)
        self.m = np.array(m)
        self.use_grav = use_grav
        self.softening = softening
        
        assert len(self.pos_0[0,:]) == 3, "pos_0 must have 3 columns (position elements x,y,z)"
        assert np.shape(self.pos_0) == np.shape(self.v_0), "pos_0 and v_0 must have the same shape, 3 columns representing x,y,z position or velocities, and rows representing masses"
        assert np.shape(self.m) == (len(self.pos_0[:,0]),), "m should be a 1D array of object masses of the same length of the number of rows as pos_0 and v_0"
        
        # calculate initial acceleration to be used in the first step of the verlet algorithm
        self.a_0 = accel(self.pos_0, self.m, self.softening, self.use_grav)
        
        # set up variables for plotting, this is designed so the user can input a plotting number much smaller than the total iterations, this number will be used to determine how many points are plotted
        assert tot_iterations >= (plotting_num * 2), "Total iterations must be larger or equal to 2 times plotting_num"
        self.step = 0
        # a point will be saved to an array at every interval iteration
        self.interval = int(tot_iterations/plotting_num)
        # due to rounding based on the values chosen, self.plotting_num will be used as the number of points plotted and therefore the length of the arrays, this will be similar or equal to the inputted plotting number
        self.plotting_num = len(np.arange(0,tot_iterations,self.interval))
        self.pos_array = np.zeros(shape = (self.plotting_num, len(self.m), 3))
        self.v_array = np.zeros(shape = (self.plotting_num, len(self.m), 3))
        self.tot_iterations = tot_iterations
        self.iteration_list = []
        self.it_step = 0
        
        # sets up calculating the period 
        if return_period == True:
            self.period_array = []
            # picks the lowest mass to calculate the period for, the period is the same for all masses however depending on the size of the larger masses, their orbits may be very small which may make calculating a period unreliable with the method used
            self.period_mass = self.m.argmin()
            if per_tolerance == None:
                # sets up a range around the initial position to accept as a potential period value
                self.range_up = self.pos_0[self.period_mass,:] * 1.05
                self.range_lo = self.pos_0[self.period_mass,:] * 0.95
                for i in range(len(self.range_up)):
                    if self.range_up[i] == 0:
                        # also apply a range to any zeros in the position
                        self.range_up[i] += np.linalg.norm(self.pos_0[self.period_mass,:])/20
                        self.range_lo[i] -= np.linalg.norm(self.pos_0[self.period_mass,:])/20
            else:
                # if a range is specified, use that instead
                self.range_up = self.pos_0[self.period_mass,:] * 1.
                self.range_lo = self.pos_0[self.period_mass,:] * 1.
                for i in range(len(self.range_up)):
                    self.range_up[i] += per_tolerance
                    self.range_lo[i] -= per_tolerance
    
    def verlet(self):
        """Function to calculate the new velocity and position of the masses using the velocity verlet algorithm
        """
        
        # self.a_0 is used to ensure acceleration is only calculated once to speed up the code
        v_step = self.v_0 + 0.5 * self.h * self.a_0
        pos = self.pos_0 + self.h * v_step
        # calculate the new acceleration, this will become the old acceleration at the end of this function
        a_cur = accel(pos, self.m, self.softening, self.use_grav)
        v = v_step + 0.5 * self.h * a_cur
        
        # calculate the change in the center of mass and minus it from the current positions
        com = 0
        msum = 0
        com_0 = 0
        for i in range(len(self.m)):
            com_0 += self.m[i] * self.pos_0[i,:]
            com += self.m[i] * pos[i,:]
            msum += self.m[i]
        comdif = (com/msum) - (com_0/msum)
        pos = pos - comdif
    
        # current positions and velocities are updated, and the old acceleration is set to the current
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
            # at the 0th iteration, add the initial positions and velocities to the saving arrays
            self.pos_array[self.it_step,:,:] = self.pos_0
            self.v_array[self.it_step,:,:] = self.v_0
            self.step += int(self.interval)
            self.iteration_list.append(iteration*self.h)
            self.it_step += 1
        elif iteration == self.step:
            # when the iteration is equal to the desired interval, add the positions and velocities to the saving arrays
            self.pos_array[self.it_step,:,:] = self.pos
            self.v_array[self.it_step,:,:] = self.v
            self.step += int(self.interval)
            self.iteration_list.append((iteration+1)*self.h)
            self.it_step += 1
        elif iteration == (self.tot_iterations - 1):
            # add on the last iteration if it hasn't already ended
            self.pos_array = np.concatenate([self.pos_array, self.pos[None]]) 
            self.v_array = np.concatenate([self.v_array, self.v[None]])
            self.iteration_list.append((iteration+1)*self.h)
            self.plotting_num += 1
        else:
            pass
        
        # if the period calculation has been set up, check if all the x,y,z positions for the selected mass are within the specified range and append the iteration to a list if that is the case
        try:
            if self.pos[self.period_mass,0] <= self.range_up[0] and self.pos[self.period_mass,1] <= self.range_up[1] and self.pos[self.period_mass,2] <= self.range_up[2] and self.pos[self.period_mass,0] >= self.range_lo[0] and self.pos[self.period_mass,1] >= self.range_lo[1] and self.pos[self.period_mass,2] >= self.range_lo[2]:
                self.period_array.append(iteration+1)
                # period_array should end up being a list of every iteration within the range, this will include points before and after the period, points aat multiples of periods, and points at the beginning
        except:
            pass
        
        # update the initial positions and velocities
        self.pos_0 = self.pos
        self.v_0 = self.v
        
    
    def plot(self, ref_frame, percent_L, percent_E):
        """Function to plot the positions of the masses over the orbits, the angular momentum of the system over the orbits and the energy of the system over the orbits
        
        Parameters
        ----------
        ref_frame : int
            integer indicating the frame of reference to plot the system in, 1 to set the frame of reference to mass 1, 2 to set the frame of reference to mass 2, ..., etc
        percent_L : bool
            if True, return the angular momentum plot as a percentage difference from the initial value (if possible), if False, return the angular momentum plot as the actual values
        percent_E : bool
            if True, return the energy plot as a percentage difference from the initial value (if possible), if False, return the energy plot as the actual values

        """
        
        # set up plots, if a z component is present in the positions or velocities then plot the orbits in 3D
        fig = plt.figure(figsize = (10,16), dpi = 200)
        if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
            ax = fig.add_subplot(3,1,1, projection = '3d')
            ax.set_zlabel('Z Position')
        else:
            ax = fig.add_subplot(3,1,1)
            ax.axis('equal')
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Orbital Paths for the N-body system')
        ax2.set_xlabel('Time (Seconds)')
        ax3.set_xlabel('Time (Seconds)')
        
        L = np.zeros(self.plotting_num, dtype = float)
        E = np.zeros(self.plotting_num, dtype = float)
        
        # plot the orbits depending on whether they should be 3d or not and if they are to be plotted based on a certain frame of reference
        for i in range(len(self.m)):
            if ref_frame is not None:
                if i == ref_frame-1:
                    if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
                        ax.scatter3D(self.pos_array[0,i,0]-self.pos_array[0,i,0], self.pos_array[0,i,1]-self.pos_array[0,i,1], self.pos_array[0,i,2]-self.pos_array[0,i,2], color = 'C'+str(i), s = 5, label = "Mass {}, {} kg".format(i+1, self.m[i]))
                    else:
                        ax.scatter(self.pos_array[0,i,0]-self.pos_array[0,i,0], self.pos_array[0,i,1]-self.pos_array[0,i,1], color = 'C'+str(i), s = 5, label = "Mass {}, {} kg".format(i+1, self.m[i]))
                else:
                    if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
                        ax.plot3D(self.pos_array[:,i,0]-self.pos_array[:,ref_frame-1,0], self.pos_array[:,i,1]-self.pos_array[:,ref_frame-1,1], self.pos_array[:,i,2]-self.pos_array[:,ref_frame-1,2], color = 'C'+str(i), label = "Mass {}, {} kg".format(i+1, self.m[i]))
                    else:
                        ax.plot(self.pos_array[:,i,0]-self.pos_array[:,ref_frame-1,0], self.pos_array[:,i,1]-self.pos_array[:,ref_frame-1,1], color = 'C'+str(i), label = "Mass {}, {} kg".format(i+1, self.m[i]))
                    
            else:
                if np.any(self.v_array[:,:,2]) == True or np.any(self.pos_array[:,:,2]) == True:
                    ax.plot3D(self.pos_array[:,i,0], self.pos_array[:,i,1], self.pos_array[:,i,2], color = 'C'+str(i), label = "Mass {}, {} kg".format(i+1, self.m[i]))
                else:
                    ax.plot(self.pos_array[:,i,0], self.pos_array[:,i,1], color = 'C'+str(i), label = "Mass {}, {} kg".format(i+1, self.m[i]))
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize = 10)
        
        # calculate the angular momentum, adding the components for each mass then normalising
        for l in range(self.plotting_num):
            L_cur = 0
            for i in range(len(self.m)):
                L_cur += np.cross(self.pos_array[l,i,:], self.m[i] * self.v_array[l,i,:])
            L[l] = np.linalg.norm(L_cur)
        
        # calculate energy
        for i in range(self.plotting_num):
            E[i] = E_pot(self.pos_array[i,:,:], self.m, self.softening, self.use_grav) + E_kin(self.v_array[i,:,:], self.m)
        
        if L[0] != 0 and percent_L == True:
            # if the initial angular momentum is not 0, calculate percentage error in angular momentum and plot
            errmod_L = np.zeros(self.plotting_num, dtype = float)
            for i in range(self.plotting_num):
                errmod_L[i] = np.sqrt(((L[i] - L[0])/L[0])**2)*100
            ax2.plot(self.iteration_list, errmod_L, color = 'C0')
            ax2.set_ylabel('Angular Momentum Error (%)')
            ax2.set_title('Percentage Angular Momentum Error of the N-body system against time')
        else:
            ax2.plot(self.iteration_list, L, color = 'C0')
            ax2.set_ylabel('Angular Momentum L')
            ax2.set_title('Angular Momentum of the N-body system against time')

        if E[0] != 0 and percent_E == True:
            # similar for the energy
            errE = np.zeros(self.plotting_num, dtype = float)
            for i in range(self.plotting_num):
                errE[i] = np.sqrt(((E[i] - E[0])/E[0])**2)*100
            ax3.plot(self.iteration_list, errE, color = 'C0')
            ax3.set_ylabel('Energy Error (%)')
            ax3.set_title('Percentage Energy Error of the N-body system against time')
        else:
            ax3.plot(self.iteration_list, E, color = 'C0')
            ax3.set_ylabel('Energy')
            ax3.set_title('Energy of the N-body system against time')
            
        
    
    def return_period(self):
        """Function to return the period of the N-body system
        """

        per_val = 0
        per_list = []
        for i in range(len(self.period_array)):
            # pick out points only around the first period
            if self.period_array[i] - self.period_array[i-1] != 1 and i != 0:
                per_val += 1
                # when the iterations are no longer consecutive numbers, we are on the next period, at this point we can add 1 to per_val
            if per_val == 1:
                # per_val will only be 1 when we are around the first period so append those iterations to another list
                per_list.append(self.period_array[i])
        # the median of this list should be closest to the actual value of the period
        self.period = np.median(per_list) * self.h
        print("Period = {} Seconds".format(self.period))
        
    
    def return_semi_major_axis(self):
        """Function to return the semi-major axis of each mass in the N-body system
        """
        
        # obtain the index of the iteration list closest to the period adn use that to mark out when one period has been completed
        iter_per = (np.abs(self.iteration_list - self.period)).argmin()
        for k in range(len(self.m)):
            axis_list = []
            for i in range(len(self.iteration_list[0:iter_per+1])):
                for l in range(len(self.iteration_list[0:iter_per+1])):
                    # use this one period to iterate through every point on the orbit calculating the difference between each point, the maximum difference will be the major axis
                    # this can become very time consuming for large pos_arrays so it is recommended to keep the plotting number reasonably low when calculating period
                    dif = np.linalg.norm(self.pos_array[i, k, :] - self.pos_array[l, k, :])
                    axis_list.append(dif)
            major_axis = np.max(axis_list)
            print("Mass {} Semi-Major Axis = {}".format(k+1, major_axis/2))
            
            
    


def nbody(time, h, position_init, velocity_init, masses, use_grav = True, ref_frame = None, softening = 0.001, plotting_num = 1000, return_period = False, per_tolerance = None, return_semi_major_axis = False, percent_L = True, percent_E = True, save_folder = None, savefilename = None):
    """Function to perform the full N-body simulation for a given set of masses with their positions and velocities for a given number of iterations

    Parameters
    ----------
    time : int
        total time in seconds to simulate the system over
    h : float
        size of the time step to use in velocity verlet method in seconds
    position_init : array of floats
        2D array of floats with 3 columns representing initial x,y,z positions in meters and N rows representing each mass
    velocity_init : array of floats
        2D array of floats with 3 columns representing initial x,y,z velocities in meters per second and N rows representing each mass
    masses : array of floats
        1D array of object masses in kg in order m_1, m_2, ..., etc
    use_grav : bool, optional
        If True, the gravitational constant G will be used in acceleration and potential energy calculation, If False, G will be treated as 1, essentially giving the answers in terms of G, defaults to True
    ref_frame : int, optional
        integer indicating the frame of reference to plot the system in, 1 to set the frame of reference to mass 1, 2 to set the frame of reference to mass 2, ..., etc, defaults to None
    softening : float, optional
        float value to define the softening of the system, this ensure correct behaviour of the bodies when the distance between them is particularlly small, defaults to 0.001
    plotting_num : int, optional
        number of points to plot on the graphs, this will be used to determine how often the results are saved to an array for plotting, this may vary around the given value depending on the given value and total iterations, defaults to 1000
    return_period : bool, optional
        if True, return the period of the N-body system, if False, do not return the period, defaults to False
    per_tolerance : float, optional
        tolerance on the period estimate, if None is entered tolerance will be 0.5%, should choose a sensible value based on the initial position, for example if the initial position is 1, a sensible value would be 0.05, this is not a percentage, if this is set too low a period may not be found, this must be set if the initial position the lowest mass is 0,0,0, defaults to None
    return_semi_major_axis : bool, optional
        if True, return the semi-major axis of each mass and return_period must also be True, if False, do not return the semi-major axes, will calculate for systems larger than 2 bodies but will be incorrect if the orbit is not elliptical, defaults to False
    percent_L : bool, optional
        if True, return the angular momentum plot as a percentage difference from the initial value (if possible), if False, return the angular momentum plot as the actual values, defaults to True
    percent_E : bool, optional
        if True, return the energy plot as a percentage difference from the initial value (if possible), if False, return the energy plot as the actual values, defaults to True
    save_folder: str, optional
        directory to save the convergence plot to, defaults to None
    savefilename: str, optional
        filename for the convergence plot, defaults to None
    
    Raises
    ------
    AssertionError:
        Raised if ref_frame is not an integer
    ValueError:
        Raised if ref_frame does not correspond to any of the given masses
    ValueError:
        Raised if return_period is False and return_semi_major_axis is True, period must be returned to return semi-major axis
    """
    
    if ref_frame is not None:
        assert type(ref_frame) == int, "ref_frame must be an integer"
        if ref_frame > len(masses) or ref_frame < 1:
            raise ValueError("ref_frame must be a non-zero positive integer corresponding to one of the masses")
    
    if return_period == False and return_semi_major_axis == True:
        raise ValueError("The period must be returned in order to return the semi-major axis")
            
    iterations = int(time/h)
        
    N = N_body(h, position_init, velocity_init, masses, use_grav, softening, iterations, plotting_num, return_period, per_tolerance)
    for i in range(iterations):
        N.verlet()
        N.save(i)
    N.plot(ref_frame, percent_L, percent_E)
    if return_period == True:
        N.return_period()
    if return_semi_major_axis == True:
        N.return_semi_major_axis()
    
    
    # save the plot provided  save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved      
    if save_folder is not None and savefilename is not None:
        plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
    elif save_folder is None and savefilename is None:
        # no message displayed if both save_folder and savefilename are none, it's assumed the user did not intend to save the plot
        pass
    else:
        print("Figure not saved, you need to provide both savefilename and save_folder to save the figure") 