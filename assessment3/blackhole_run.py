"""
File to run the code for the black hole trajectories
Please ensure blackhole_main.py and blackhole_aux.py, and blackhole_run.py are saved in the same folder otheriwse the imports will not work

Author : Bryce Dixon
Version : 26/03/24
"""

import blackhole_main as bh

# this code is very simple to use and runs off of the settings in the bh_inputs.yaml file
# these should be set accordingly in the file, all of the parameter descriptions are located in that file
# the bh.trajectory function is the only function that the user needs to call, the only parameter for this is the directory of the settings file
bh.trajectory(input_file_dir = 'C:/Users/bryce/Python/Computational/assessment3/bh_inputs.yaml')