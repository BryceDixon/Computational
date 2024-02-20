

import numpy as np
from scipy.integrate import RK45 as RK
import matplotlib.pyplot as plt
from astropy.constants import G as G_grav
from astropy.constants import c as c_speed
import blackhole_aux as aux
import os
import yaml


class bh_path:

    def __init__(self, y, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const = False):
        """
        Class to obtain the trajectories of a particle around a black hole

        Parameters
        ----------
        y : array of floats
            1D array of the initial conditions for a keplerian orbit in the form:
            y[0] = t
            y[1] = dt
            y[2] = r
            y[3] = dr
            y[4] = phi
            y[5] = dphi
            y[6] = mass
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

        self.y_values = [y]
        self.propt_values = [init_proper_t]
        if use_const == False:
            self.runga = RK(fun = aux.geodesic, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        if use_const == True:
            self.runga = RK(fun = aux.geodesic_const, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        self.t_bound = t_bound
        self.use_const = use_const
        
        # plots will require maths using c and G, set these up accordingly
        if self.use_const == True:
            self.c = c_speed
            self.G = G_grav
        else:
            self.c = 1
            self.G = 1
    
    def run(self, save_interval = None):
        """
        Function to run the 4th order runge-kutta method using the scipy RK45 algorithm to generate the particle velocities

        Parameters
        ----------
        save_interval : int, optional
            sets how often append the results to the saving arrays, if None will save every step, this is not recommended for a large number of steps, defaults to None
        """
        
        assert type(save_interval) == int or save_interval == None, "save_interval must be an integer or None, represents the interval between steps that the data should be saved"
        num = 0
        save_num = save_interval
        while self.propt_values[-1] <= self.t_bound:
            # step the integrator forward
            self.runga.step()
            if save_interval == None:
                # append the results to the saving arrays based on the saving interval
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
        
        self.y_values = np.array(self.y_values)
        self.propt_values = np.array(self.propt_values)
            
    def plot(self, conserve_plots = False):
        """
        Function to plot the trajectories around the black hole and to plot the conservation checks if the user requests it

        Parameters
        ----------
        conserve_plots : bool, optional
            if True, plots the conservation checks, if False, does not, defaults to False
        """
        
        if conserve_plots == True:
            fig = plt.figure(figsize = (10,26))
            ax = fig.add_subplot(5,1,1)
            ax1 = fig.add_subplot(5,1,2)
            ax2 = fig.add_subplot(5,1,3)
            ax3 = fig.add_subplot(5,1,4)
            ax4 = fig.add_subplot(5,1,5)
        else:
            fig = plt.figure(figsize = (10,16))
            ax = fig.add_subplot(1,1,1)
        
        # plot the main trajectories
        r = self.y_values[:,2]
        phi = self.y_values[:,4]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        ax.plot(x,y, color = 'b', label = 'Particle Trajectory')
        ax.scatter(0,0, color = 'k', label = 'Black Hole Position')
        ax.axis('equal')
        ax.set_xlabel('X Position /m')
        ax.set_ylabel('Y Position /m')
        ax.set_title('Trajectory of a Particle Orbiting a Black Hole')
        
        if conserve_plots == True:
            # get constants M and R (schwartschild radius)
            M = self.y_values[0,6]
            R = (2*self.G*M)/(self.c**2)
            # c^2 is conserved based on the same invariant equation used to calclate the initial t velocity, this time taking into acocunt dt, dr, and dphi
            c_conserve = (1-R/r)*self.y_values[:,1]**2 - (1/(1-R/r))*self.y_values[:,3]**2 - r**2 * self.y_values[:,5]**2
            c_conserve_perc = np.sqrt(((c_conserve - c_conserve[0])/c_conserve[0])**2) * 100
            
            ax1.plot(self.propt_values, c_conserve_perc, color = 'b')
            ax1.set_xlabel('Proper Time /s')
            ax1.set_ylabel('Percentage Difference /%')
            ax1.set_title('Percentage Difference in the Conserved Quantity $c^2$ against Proper Time')
            
            # angular momentum and energy conservation plots
            m = M * 1e-10
            dphi = self.y_values[:,5]
            dr = self.y_values[:,3]
            energy = (-self.G * M * m)/(r) + (0.5 * m * r**2 * dphi**2) + (0.5 * m * dr**2)
            energy_perc = np.sqrt(((energy - energy[0])/energy[0])**2) * 100
            
            ax2.plot(self.propt_values, energy_perc, color = 'b')
            ax2.set_xlabel('Proper Time /s')
            ax2.set_ylabel('Percentage Difference /%')
            ax2.set_title('Percentage Difference in the Total Energy against Proper Time')
            
            ang_mom = m * dphi * r**2
            ang_mom_perc = np.sqrt(((ang_mom - ang_mom[0])/ang_mom[0])**2) * 100
            
            ax3.plot(self.propt_values, ang_mom_perc, color = 'b')
            ax3.set_xlabel('Proper Time /s')
            ax3.set_ylabel('Percentage Difference /%')
            ax3.set_title('Percentage Difference in the Angular Momentum against Proper Time')
            
            # plot computed accelerations against newtonian accelerations
            M = self.y_values[0,6]
            r = self.y_values[:,2]
            newton = -(self.G * M)/(r**2)
        
            r_accel = []
            t_step = []
            for i in range(len(r)):
                if i == 0:
                    pass
                else:
                    r_accel.append(r[i]-r[i-1])
                    t_step.append(self.propt_values[i]-self.propt_values[i-1])
            r_accel = np.array(r_accel)
            t_step = np.array(t_step)
            r_accel = r_accel/t_step
            
            ax4.plot(self.propt_values, newton, color = 'red', label = 'Newtonian Acceleration')
            ax4.plot(self.propt_values[1:], r_accel, color = 'blue', label = 'Computed GR Acceleration')
            ax4.set_xlabel('Proper Time /s')
            ax4.set_ylabel('Acceleration $m/s^2$')
            ax4.set_title('Acceleration against Proper Time')
            ax4.legend(bbox_to_anchor=(0.01, 0.01), loc='lower left', borderaxespad=0, fontsize = 10)
            
    
    def kep_check(self):
        
        period_phi = self.y_values[0,4] + (np.pi * 2)
        iter_per = (np.abs(self.y_values[:,4] - period_phi)).argmin()
        period = self.propt_values[iter_per] - self.propt_values[0]
        print("Period = {} Seconds".format(period))
        
        semi_major_axis = np.max(self.y_values[:,2])
        print("Semi Major Axis = {} Meters".format(semi_major_axis))
        
        M = self.y_values[0,6]
        kep = (4 * np.pi**2)/(self.G * M)
        com_kep = (np.float64(period)**2)/(np.float64(semi_major_axis)**3)
        print("Expected Kepler's 3rd Law ratio = {}".format(kep))
        print("Computed Kepler's 3rd Law ratio = {}".format(com_kep))
        
    
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
        



def trajectory(input_file_dir):
    """
    Function to run the bh_path class to generate and plot the trajectories of a particle around a black hole using a 4th order runge-kutta integrator method

    Parameters
    ----------
    input_file_dir : str
        name of the file path to the bh_inputs.yaml config file
    """
    
    para_file = open(str(input_file_dir), 'r')
    params = yaml.safe_load(para_file)
    
    savefolder = params['save_directory']
    auto_kep = params['auto_keplerian']
    use_const = params['use_const']
    init_t = params['initial_time']
    init_r = params['initial_r']
    init_phi = params['initial_phi']
    mass = params['black_hole_mass']
    init_proper_t = params['initial_proper_t']
    t_bound = params['max_proper_t']
    max_step = params['max_step']
    first_step = params['initial_step']
    atol = np.float64(params['atol'])
    rtol = np.float64(params['rtol'])
    save_interval = params['save_interval']
    conserve_plots = params['conserve_plots']
    keplerian_check = params['keplerian_check']
    
    if auto_kep == True:
        y = aux.initial_y(init_t, init_r, init_phi, mass, use_const)
    else:
        init_dr = params['initial_dr']
        init_dphi = params['initial_dphi']
        init_dt = aux.initial_dt(mass, init_r, init_dphi, init_dr, use_const)
        y = np.array([init_t, init_dt, init_r, init_dr, init_phi, init_dphi, mass])
    
    bh = bh_path(y, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const)
    bh.run(save_interval)
    bh.plot(conserve_plots)
    if keplerian_check is True:
        bh.kep_check()
    if savefolder is not None:
        bh.save(savefolder)
            
            
            
            
    
    



    