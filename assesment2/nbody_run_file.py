"""
File to run all the code for the N body assessment
Please ensure that nbody_main.py is saved in the same folder as this file otherwise the import will not work

Author: Bryce Dixon
Version: 07/12/2023  
"""

import numpy as np
import nbody_main as nbody


# for this example we will run a very simple 2 body simulation using mostly default parameters
# the nbody function is the only function the user will need to interact with, this has an extensive number of parameters allowing users to fully customise their output
# position, mass and velocity arrays must be created and entered into the nbody function, position and velocity must be arrays with 3 columns representing x,y,z and any number of rows each representing a mass
# the mass list should match the number of rows in the velocity and position arrays effectively designating each row with that mass, all values should be given i SI units
pos = np.array([(1,0,0),(0,0,0)])
mass = (1,1)
v = np.array([(0,1,0),(0,0,0)])
total_time = 10
time_step = 0.001
# we run this for a total time of 10 seconds, this is not the real time that the code runs for, it is used with time step to dictate the number of iterations and then used with plotting so graphs against time can be plotted
# all of the parameters are listed below in the function, the only required parameters are time, h, position_init, velocity_init, and masses
# the rest of there parameters are set to their default other than use_grav, this is set to false so this fake system can be plotted with G = 1
# a full desciption of each parameter is available in the function docstring
nbody.nbody(time = total_time, h = time_step, position_init = pos, velocity_init = v, masses = mass, use_grav = False, ref_frame = None, softening = 0.001, plotting_num = 1000, return_period = False, per_tolerance = None, return_semi_major_axis = False, percent_L = True, percent_E = True, save_folder = None, savefilename = None)
# testing a different system is very easy and simply requires changing or extending the position, mass and velocity arrays, and changing the other input parameters to fit the desired output