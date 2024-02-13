

import numpy as np
from scipy.integrate import RK45 as RK
import matplotlib.pyplot as plt
from astropy.constants import G as G_grav
from astropy.constants import c as c_speed
import blackhole_aux as aux
import os


class bh_path:

    def __init__(self, init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const = False):
        """
        Class to obtain the trajectories of a particle around a black hole

        Parameters
        ----------
        init_t : float
            initial time in seconds
        init_r : float
            initial radial distance from the black hole in meters
        init_phi : float
            initial azimuthal angle around the black hole in radians
        mass : float
            mass of the black hole in kg
        init_proper_t : float
            initial proper time in seconds
        t_bound : float
            maximum proper time in seconds, boundary time of the integration, the integration will not continue beyond this
        max_step : float
            maximum time step to be used by the adaptive time step integrator
        first_step : float
            initial time step to be used by the integrator
        atol : float
            absolute tolerance of the integrator, controls the absolute accuracy (number of correct decimal places)
        rtol : float
            relative tolderance of the integrator, controls the relative accuracy (number of correct digits)
        use_const : bool, optional
            if False, sets G and c to 1 when required in calculations, if True, uses G and c values from the astropy constants library in calculations, defaults to False
        """
    
        y = aux.initial_y(init_t, init_r, init_phi, mass, use_const)
        self.y_values = [y]
        self.propt_values = [init_proper_t]
        if use_const == False:
            self.runga = RK(fun = aux.geodesic, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        if use_const == True:
            self.runga = RK(fun = aux.geodesic_const, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        self.t_bound = t_bound
        self.use_const = use_const
    
    def run(self, save_interval = None):
        """
        Function to run the 4th order runge-kutta method using the scipy RK45 algorithm to generate the particle velocities

        Parameters
        ----------
        save_interval : int, optional
            sets how often append the results to the saving arrays, if None will save every step, this is not recommended for a large number of steps, defaults to None
        """
        
        assert type(save_interval) == int, "save_interval must be an integer, represents the interval between steps that the data should be saved"
        num = 0
        save_num = save_interval
        while self.propt_values[-1] <= self.t_bound:
            # step the integrator forward
            self.runga.step()
            if save_interval == None:
                # append the results to teh saving arrays based on the saving interval
                self.y_values.append(self.runga.y)
                self.propt_values.append(self.runga.t)
            else:
                num += 1
                if num == save_num:
                    self.y_values.append(self.runga.y)
                    self.propt_values.append(self.runga.t)
                    save_num += save_interval
            # continue the loop until the integrator finishes or fails
            if self.runga.status == 'finished':
                break
            elif self.runga.status == 'failed':
                print("Runga Kutta method failed, generating results prior to failing")
                break
            else:
                pass
            
    def plot(self, conserve_plots = True):
        """
        Function to plot the trajectories around the black hole and to plot the conservation checks if the user requests it

        Parameters
        ----------
        conserve_plots : bool, optional
            if True, plots the conservation checks, if False, does not, defaults to True
        """
        
        fig = plt.figure(figsize = (10,16))
        if conserve_plots == True:
            ax = fig.add_subplot(2,1,1)
            ax1 = fig.add_subplot(2,1,2)
        else:
            ax = fig.add_subplot(1,1,1)
        
        # plot the main trajectories
        y_values = np.array(self.y_values)
        r = y_values[:,2]
        phi = y_values[:,4]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        ax.plot(x,y, color = 'b', label = 'Particle Trajectory')
        ax.scatter(0,0, color = 'k', label = 'Black Hole Position')
        ax.axis('equal')
        ax.set_xlabel('X Position /m')
        ax.set_ylabel('Y Position /m')
        ax.set_title('Trajectory of a Particle Orbiting a Black Hole')
        
        if conserve_plots == True:
            # conservation plots will require maths using c and G, set these up accordingly
            if self.use_const == True:
                c = c_speed
                G = G_grav
            else:
                c = 1
                G = 1
            # get constants M and R (schwartschild radius)
            M = y_values[0,6]
            R = (2*G*M)/(c**2)
            # c^2 is conserved based on the same invariant equation used to calclate the initial t velocity, this time taking into acocunt dt, dr, and dphi
            c_conserve = (1-R/r)*y_values[:,1]**2 - (1/(1-R/r))*y_values[:,3]**2 - r**2 * y_values[:,5]**2
            c_conserve_perc = ((c_conserve - c_conserve[0])/c_conserve[0]) * 100
            
            ax1.plot(self.propt_values, c_conserve_perc, color = 'b')
            ax1.set_xlabel('Proper Time /s')
            ax1.set_ylabel('Percentage Difference /%')
            ax1.set_title('Percentage Difference in the Conserved Quantity $c^2$ against Proper Time')
    
    def save(self, savefolder):
        """
        Function to save the plots and the data

        Parameters
        ----------
        savefolder : str
            name of the directory to save the data to
        """
        
        plt.savefig(str(savefolder)+'/blackhole_plots.png', bbox_inches='tight')
        trajfilename = os.path.join(savefolder, 'trajectories.txt')
        proptfilename = os.path.join(savefolder, 'proper_times.txt')
        np.savetxt(trajfilename, self.y_values)
        np.savetxt(proptfilename, self.propt_values)
        



def trajectory(init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const = False, save_interval = None, conserve_plots = True, savefolder = None):
    """
    Function to run the bh_path class to generate and plot the trajectories of a particle around a black hole using a 4th order runge-kutta integrator method

    Parameters
    ----------
    init_t : float
        initial time in seconds
    init_r : float
        initial radial distance from the black hole in meters
    init_phi : float
        initial azimuthal angle around the black hole in radians
    mass : float
        mass of the black hole in kg
    init_proper_t : float
        initial proper time in seconds
    t_bound : float
        maximum proper time in seconds, boundary time of the integration, the integration will not continue beyond this
    max_step : float
        maximum time step to be used by the adaptive time step integrator
    first_step : float
        initial time step to be used by the integrator
    atol : float
        absolute tolerance of the integrator, controls the absolute accuracy (number of correct decimal places)
    rtol : float
        relative tolderance of the integrator, controls the relative accuracy (number of correct digits)
    use_const : bool, optional
        if False, sets G and c to 1 when required in calculations, if True, uses G and c values from the astropy constants library in calculations, defaults to False
    save_interval : int, optional
        sets how often append the results to the saving arrays, if None will save every step, this is not recommended for a large number of steps, defaults to None
    conserve_plots : bool, optional
        if True, plots the conservation checks, if False, does not, defaults to True
    savefolder : str
        name of the directory to save the data to
    """
    
    bh = bh_path(init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const)
    bh.run(save_interval)
    bh.plot(conserve_plots)
    if savefolder is not None:
        bh.save(savefolder)
            
            
            
            
    
    



    