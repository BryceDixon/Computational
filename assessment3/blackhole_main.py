

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
        # initialise the runge kutta class with the correct geodesic function based on the values for G and c
        if use_const == False:
            self.runga = RK(fun = aux.geodesic, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        if use_const == True:
            self.runga = RK(fun = aux.geodesic_const, t0 = init_proper_t, y0 = y, t_bound = t_bound, max_step = max_step, first_step = first_step, atol = atol, rtol = rtol)
        self.t_bound = t_bound
        self.use_const = use_const
        self.t_bound = t_bound
        self.max_step = max_step
        self.first_step = first_step
        self.atol = atol
        self.rtol = rtol
        self.use_const = use_const
        
        # plots will require maths using c and G, set these up accordingly
        if self.use_const == True:
            self.c = c_speed
            self.G = G_grav
        else:
            self.c = 1
            self.G = 1
        # Schwarzschild radius will be used in plotting
        self.R = (2 * self.G * y[6])/(self.c**2)
    
    def run(self, save_interval = None):
        """
        Function to run the 4th order runge-kutta method using the scipy RK45 algorithm to generate the particle velocities

        Parameters
        ----------
        save_interval : int, optional
            sets how often append the results to the saving arrays, if None will save every step, this is not recommended for a large number of steps, defaults to None
        """
        
        assert type(save_interval) == int or save_interval == 'None', "save_interval must be an integer or None, represents the interval between steps that the data should be saved"
        num = 0
        save_num = save_interval
        while self.propt_values[-1] <= self.t_bound:
            # step the integrator forward
            self.runga.step()
            if save_interval == 'None':
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
                print("The particle is within 1 Schwarzschild Radius of the black hole so has crossed the event horizon")
                break
            else:
                pass
        
        self.y_values = np.array(self.y_values)
        self.propt_values = np.array(self.propt_values)
        self.save_interval = save_interval
            
    def plot(self, conserve_plots = False, massless = False):
        """
        Function to plot the trajectories around the black hole and to plot the conservation checks if the user requests it

        Parameters
        ----------
        conserve_plots : bool
            if True, plots the conservation checks, if False, does not, defaults to False
        """
        
        if conserve_plots == True:
            if massless == False:
                fig = plt.figure(figsize = (10,26))
                ax = fig.add_subplot(4,1,1)
                ax1 = fig.add_subplot(4,1,2)
                ax2 = fig.add_subplot(4,1,3)
                ax3 = fig.add_subplot(4,1,4)
            else:
                fig = plt.figure(figsize = (10,20))
                ax = fig.add_subplot(2,1,1)
                ax1 = fig.add_subplot(2,1,2)
        else:
            fig = plt.figure(figsize = (10,16))
            ax = fig.add_subplot(1,1,1)
        
        # convert the trajectories to cartesian and plot along with the black hole position, the schwarzschild radius, and 3 times the schwarzschild radius (the last stable orbit radius)
        r = self.y_values[:,2]
        phi = self.y_values[:,4]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        # units of schwarzschild radius so divide x and y by R
        ax.plot(x/self.R, y/self.R, color = 'b', label = 'Particle Trajectory')
        ax.scatter(0,0, color = 'k', label = 'Black Hole Position')
        bh_pos = plt.Circle((0, 0), 1, color='k', fill=False, label = 'Schwarzschild Radius')
        R_3 = plt.Circle((0, 0), 3, color='orange', fill=False, label = '3 Schwarzschild Radii')
        ax.add_patch(bh_pos)
        ax.add_patch(R_3)
        ax.axis('equal')
        ax.set_xlabel('X Position /Schwarzschild Radii')
        ax.set_ylabel('Y Position /Schwarzschild Radii')
        ax.set_title('Trajectory of a Particle Orbiting a Black Hole')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize = 10)
        
        if conserve_plots == True:
            # get constants M and R (schwartschild radius)
            M = self.y_values[0,6]
            R = self.R
            # c^2 is conserved based on the same invariant equation used to calclate the initial t velocity, this time taking into acocunt dt, dr, and dphi
            c_conserve = (1-R/r)*self.y_values[:,1]**2 - (1/(1-R/r))*self.y_values[:,3]**2 - r**2 * self.y_values[:,5]**2
            if massless == True:
                c_conserve_perc = c_conserve
                ax1.set_ylabel('$g_{\\mu \\nu} U^{\\mu} U^{\\nu}$')
                ax1.set_title('Conserved Quantity, $g_{\\mu \\nu} U^{\\mu} U^{\\nu}$, against Proper Time')
            else:
                c_conserve_perc = np.sqrt(((c_conserve - c_conserve[0])/c_conserve[0])**2) * 100
                ax1.set_ylabel('Percentage Difference /%')
                ax1.set_title('Percentage Difference in the Conserved Quantity $c^2$ against Proper Time')
            
            ax1.plot(self.propt_values, c_conserve_perc, color = 'b')
            ax1.set_xlabel('Proper Time /s')
            
            if massless == False:
                # angular momentum and energy conservation plots
                m = M * 1e-10
                dphi = self.y_values[:,5]
                dr = self.y_values[:,3]
                # potential plus angular kinetic and radial kinetic energy
                energy = (-self.G * M * m)/(r) + (0.5 * m * r**2 * dphi**2) + (0.5 * m * dr**2)
                # plot as percentage difference if the initial value is not 0
                if energy[0] != 0:
                    energy_perc = np.sqrt(((energy - energy[0])/energy[0])**2) * 100
                    energy_plot = energy_perc
                    etit = 'Percentage Difference in the Total Energy against Proper Time'
                    elab = 'Percentage Difference /%'
                else:
                    energy_plot = energy
                    etit = 'Total Energy against Proper Time'
                    elab = 'Total Energy /J'
            
                ax2.plot(self.propt_values, energy_plot, color = 'b')
                ax2.set_xlabel('Proper Time /s')
                ax2.set_ylabel(elab)
                ax2.set_title(etit)
            
                ang_mom = m * dphi * r**2
                # similarly for angular momentum
                if ang_mom[0] != 0:
                    ang_mom_perc = np.sqrt(((ang_mom - ang_mom[0])/ang_mom[0])**2) * 100
                    ang_mom_plot = ang_mom_perc
                    atit = 'Percentage Difference in the Angular Momentum against Proper Time'
                    alab = 'Percentage Difference /%'
                else:
                    ang_mom_plot = ang_mom
                    atit = 'Angular Momentum against Proper Time'
                    alab = 'Total Angular Momentum /$kgm^2s^{-1}$' 
            
                ax3.plot(self.propt_values, ang_mom_plot, color = 'b')
                ax3.set_xlabel('Proper Time /s')
                ax3.set_ylabel(alab)
                ax3.set_title(atit)
        
        self.fig = fig
        self.ax = ax
        self.plotx = x
        self.ploty = y
            
    
    def kep_check(self):
        """
        Function to return the period and semi major axis of the orbit and compare the ratio to the expected value accoridng to Newton's third law
        """
        
        # get period by adding 1 rotation (2pi) to the initial phi value and finding the proper time of that point
        period_phi = self.y_values[0,4] + (np.pi * 2)
        iter_per = (np.abs(self.y_values[:,4] - period_phi)).argmin()
        period = self.propt_values[iter_per] - self.propt_values[0]
        print("Period = {} Seconds".format(period))
        
        axis_list = []
        for i in range(len(self.propt_values[0:iter_per+1])):
            for l in range(len(self.propt_values[0:iter_per+1])):
                # iterate through every point in one period calculating the difference between each point, the maximum difference will be the major axis
                # this can become very time consuming for large arrays so it is recommended to set a reasonable save interval when calculating period
                dif = np.sqrt((self.plotx[i] - self.plotx[l])**2 + (self.ploty[i] - self.ploty[l])**2)
                axis_list.append(dif)
        semi_major_axis = (np.max(axis_list))/2
        print("Semi-Major Axis = {} Metres".format(semi_major_axis))
        
        # compare p^2/a^3 with 4pi^2/G*M from Newton's third law
        M = self.y_values[0,6]
        kep = (4 * np.pi**2)/(self.G * M)
        com_kep = (np.float64(period)**2)/(np.float64(semi_major_axis)**3)
        print("Expected Kepler's 3rd Law ratio = {}".format(kep))
        print("Computed Kepler's 3rd Law ratio = {}".format(com_kep))
        
    
    def newton_check(self, newt_plot_type):
        """
        Function to the compare general relativistic orbits to newtonian orbits. Calculates newtonian trajectories using the universal law of gravitation. Plots newtonian trajectories over the GR ones and plots r against time for newtonian orbits over that of GR.
        """
        
        # the newtonian trajectories are calculated using the same runge kutta method with a much simpler newtonian equation for acclerations, this requires the use of cartesian x, y, dx, dy coordinates
        init_y = self.y_values[0,:]
        y_array = np.zeros(shape = (5), dtype=float)
        # convert spherical r, phi, dr, dphi to x, y, dx, dy
        y_array[0] = init_y[2] * np.cos(init_y[4]) # x
        y_array[2] = init_y[2] * np.sin(init_y[4]) # y
        y_array[1] = (init_y[3] * np.cos(init_y[4])) - (init_y[2] * init_y[5] * np.sin(init_y[4])) # dx
        y_array[3] = (init_y[3] * np.sin(init_y[4])) + (init_y[2] * init_y[5] * np.cos(init_y[4])) # dy
        y_array[4] = init_y[6] # mass
        
        # initialise the correct runge kutta method based on the values for G and c
        if self.use_const is True:
            runga = RK(fun = aux.newtonian_const, t0 = self.propt_values[0], y0 = y_array, t_bound = self.t_bound, max_step = self.max_step, first_step = self.first_step, atol = self.atol, rtol = self.rtol)
        else:
            runga = RK(fun = aux.newtonian, t0 = self.propt_values[0], y0 = y_array, t_bound = self.t_bound, max_step = self.max_step, first_step = self.first_step, atol = self.atol, rtol = self.rtol)
        
        # run the method in the exact same way as before
        newt_array = [y_array]
        newt_time = [self.propt_values[0]]
        num = 0
        save_num = self.save_interval
        while newt_time[-1] <= self.t_bound:
            # step the integrator forward
            runga.step()
            if self.save_interval == 'None':
                # append the results to the saving arrays based on the saving interval
                newt_array.append(runga.y)
                newt_time.append(runga.t)
            else:
                num += 1
                if num == save_num:
                    newt_array.append(runga.y)
                    newt_time.append(runga.t)
                    save_num += self.save_interval
            # continue the loop until the integrator finishes or fails
            if runga.status == 'finished':
                break
            elif runga.status == 'failed':
                print("Runga Kutta method failed, generating results prior to failing")
                break
            else:
                pass
        newt_array = np.array(newt_array)
        newt_time = np.array(newt_time)
        
        # create 20 points equally spaced in time along the newtonian trajectories to use for plotting
        if newt_plot_type == 'dots':
            points = np.linspace(0, len(newt_time)-1, 20)
            newt_points =  []
            time_points = []
            for i in points:
                newt_points.append(newt_array[int(i),:])
                time_points.append(newt_time[int(i)])
            newt_points = np.array(newt_points)
            time_points = np.array(time_points)
        
        newt_fig = plt.figure(figsize=(10,16))
        ax = newt_fig.add_subplot(2,1,1)
        ax1 = newt_fig.add_subplot(2,1,2)
        
        # plot the trajectories, black hole position, and schwarzschild radius as previously
        ax.plot(self.plotx/self.R, self.ploty/self.R, color = 'b', label = 'General Relativistic', zorder = 15)
        try:
            # if the user specified to use dots then the newtonian trajectories will be plotted as the 20 equally spaced points in time, this is not ideal for multiple orbits as the points tend to clump together
            ax.scatter(newt_points[:,0]/self.R, newt_points[:,2]/self.R, color = 'red', label = 'Newtonian', zorder = 20)
        except:
            # if the user specified a particular linestyle, the newtonian trajectories will be plotted as that
            ax.plot(newt_array[:,0]/self.R, newt_array[:,2]/self.R, color = 'red', linestyle = newt_plot_type, label = 'Newtonian', zorder = 20)
        ax.scatter(0,0, color = 'k', label = 'Black Hole Position', zorder = 0)
        bh_pos = plt.Circle((0, 0), 1, color='k', fill=False, label = 'Schwarzschild Radius', zorder = 5)
        R_3 = plt.Circle((0, 0), 3, color='orange', fill=False, label = '3 Schwarzschild Radii', zorder = 10)
        ax.add_patch(bh_pos)
        ax.add_patch(R_3)
        ax.axis('equal')
        ax.set_xlabel('X Position /Schwarzschild Radii')
        ax.set_ylabel('Y Position /Schwarzschild Radii')
        ax.set_title('Trajectory of a Particle Orbiting a Black Hole')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize = 10)
        
        # plot radius against proper time for both GR and newtonian
        r = self.y_values[:,2]
        try:
            # try plotting the equally spaced points, this is not ideal for a large number of orbits as it is very difficult to see the trend
            newt_r = np.sqrt(newt_points[:,0]**2 + newt_points[:,2]**2)
            ax1.scatter(time_points, newt_r/self.R, color = 'red', label = 'Newtonian', zorder = 5)
        except:
            # plot instead with the user specified linestyle
            newt_r = np.sqrt(newt_array[:,0]**2 + newt_array[:,2]**2)
            ax1.plot(newt_time, newt_r/self.R, color = 'red', linestyle = newt_plot_type, label = 'Newtonian', zorder = 5)
        ax1.plot(self.propt_values, r/self.R, color = 'b', label = 'General Relativistic', zorder = 0)
        ax1.set_xlabel('Proper Time /s')
        ax1.set_ylabel('Radial Distance /Schwarzschild Radii')
        ax1.set_title('Radial Distance against Proper Time for a Particle around a Black Hole')
        ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize = 10)
        
        self.newt_fig = newt_fig
        self.newt_array = newt_array
        self.newt_time = newt_time
        
    
    def save(self, savefolder):
        """
        Function to save the plots and the data

        Parameters
        ----------
        savefolder : str
            name of the directory to save the data to
        """
        
        # save the main figures and trajectories
        self.fig.savefig(str(savefolder)+'/blackhole_plots.png', bbox_inches='tight')
        trajfilename = os.path.join(savefolder, 'trajectories.txt')
        proptfilename = os.path.join(savefolder, 'proper_times.txt')
        np.savetxt(trajfilename, self.y_values)
        np.savetxt(proptfilename, self.propt_values)
        try:
            # if they exist, save the newtonian comparison figures and newtonian trajectories
            self.newt_fig.savefig(str(savefolder)+'/blackhole_newtonian_plots.png', bbox_inches='tight')
            newt_trajfilename = os.path.join(savefolder, 'newtonian_trajectories.txt')
            newt_proptfilename = os.path.join(savefolder, 'newtonian_times.txt')
            np.savetxt(newt_trajfilename, self.newt_array)
            np.savetxt(newt_proptfilename, self.newt_time)
        except:
            pass



def trajectory(input_file_dir):
    """
    Function to run the bh_path class to generate and plot the trajectories of a particle around a black hole using a 4th order runge-kutta integrator method

    Parameters
    ----------
    input_file_dir : str
        name of the file path to the bh_inputs.yaml config file
    """
    
    # read out all the relevant parameters from the input file
    para_file = open(str(input_file_dir), 'r')
    params = yaml.safe_load(para_file)
    
    savefolder = params['save_directory']
    use_const = params['use_const']
    mass = np.float64(params['black_hole_mass'])
    init_t = np.float64(params['initial_time'])
    if use_const is True:
        c = c_speed
        G = G_grav
    else:
        c = 1
        G = 1
    R = ((2*G*mass)/(c**2))
    # change r to si units so it can be used in the integrator
    init_r = np.float64(params['initial_r']) * R
    init_phi = np.float64(params['initial_phi'])
    init_proper_t = np.float64(params['initial_proper_t'])
    t_bound = np.float64(params['max_proper_t'])
    max_step = np.float64(params['max_step'])
    first_step = np.float64(params['initial_step'])
    atol = np.float64(params['atol'])
    rtol = np.float64(params['rtol'])
    save_interval = params['save_interval']
    conserve_plots = params['conserve_plots']
    keplerian_check = params['keplerian_check']
    newton_check = params['newton_check']
    newt_plot_type = params['newt_plot_type']
    massless = params['massless_particle']
    
    # convert dr to si units so it can be used in the integrator
    init_dr = np.float64(params['initial_dr']) * R
    init_dphi = np.float64(params['initial_dphi'])
    
    if massless is True:
        max_dphi = np.sqrt(((c**2)*(1-R/init_r))/(init_r**2))
        max_dr = np.sqrt((c**2)*((1-R/init_r)**2))
        assert init_dphi < max_dphi, "The photon is travelling faster than the speed of light, please enter a lower velocity"
        assert init_dr < max_dr, "The photon is travelling faster than the speed of light, please enter a lower velocity"
        if init_dr == np.float64(0):
            init_dr = -np.sqrt((c**2)*((1-R/init_r)**2) - (init_r**2)*(init_dphi**2)*(1-R/init_r))
        elif init_dphi == np.float64(0):
            init_dphi = np.sqrt((((c**2)*(1-R/init_r))/(init_r**2))-((init_dr**2)/((init_r**2)*(1-R/init_r))))
        else:
            raise ValueError("Only dphi or dr should be specified, please set the other to zero and it will be calculated to ensure the total space velocity is equal to c")
    # initial_dt calculates the time velocity given the other positions and velocities
    init_dt = aux.initial_dt(mass, init_r, init_dphi, init_dr, use_const, massless)
    y = np.array([init_t, init_dt, init_r, init_dr, init_phi, init_dphi, mass])
    
    # initialise the bh_path class using and call class functions using the input parameters
    bh = bh_path(y, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const)
    bh.run(save_interval)
    bh.plot(conserve_plots, massless)
    if keplerian_check is True:
        bh.kep_check()
    if newton_check is True:
        bh.newton_check(newt_plot_type)
    if savefolder != 'None':
        bh.save(savefolder)
            
            
            
            
    
    



    