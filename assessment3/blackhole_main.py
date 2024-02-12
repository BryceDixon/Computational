

import numpy as np
from scipy.integrate import RK45 as RK
import matplotlib.pyplot as plt
from astropy.constants import G as G_grav
from astropy.constants import c as c_speed
import blackhole_aux as aux


class bh_path:

    def __init__(self, init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const = False):
    
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
        
        num = 0
        save_num = save_interval
        while self.propt_values[-1] <= self.t_bound:
            self.runga.step()
            if save_interval == None:
                self.y_values.append(self.runga.y)
                self.propt_values.append(self.runga.t)
            else:
                num += 1
                if num == save_num:
                    self.y_values.append(self.runga.y)
                    self.propt_values.append(self.runga.t)
                    save_num += save_interval
            if self.runga.status == 'finished':
                break
            elif self.runga.status == 'failed':
                print("Runga Kutta method failed, generating results prior to failing")
                break
            else:
                pass
            
    def plot(self, conserve_plots = True):
        
        fig = plt.figure(figsize = (10,16))
        if conserve_plots == True:
            ax = fig.add_subplot(2,1,1)
            ax1 = fig.add_subplot(2,1,2)
        else:
            ax = fig.add_subplot(1,1,1)
            
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
            
            if self.use_const == True:
                c = c_speed
                G = G_grav
            else:
                c = 1
                G = 1
            
            M = y_values[0,6]
            R = (2*G*M)/(c**2)
            
            c_conserve = (1-R/r)*y_values[:,1]**2 - (1/(1-R/r))*y_values[:,3]**2 - r**2 * y_values[:,5]**2
            c_conserve_perc = ((c_conserve - c_conserve[0])/c_conserve[0]) * 100
            
            ax1.plot(self.propt_values, c_conserve_perc, color = 'b')
            ax1.set_xlabel('Proper Time /s')
            ax1.set_ylabel('Percentage Difference /%')
            ax1.set_title('Percentage Difference in the Conserved Quantity $c^2$ against Proper Time')



def trajectory(init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const = False, save_interval = None, conserve_plots = True):
    
    bh = bh_path(init_t, init_r, init_phi, mass, init_proper_t, t_bound, max_step, first_step, atol, rtol, use_const)
    bh.run(save_interval)
    bh.plot(conserve_plots)
            
            
            
            
    
    



    