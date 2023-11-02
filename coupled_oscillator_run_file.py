"""
File to run all the code for the coupled oscillator assessment in and produce all the required outputs
Please ensure that qu_decomp.py and os_plot.py are saved in the same folder as this file otherwise the imports will not work
For clarity of output it could be recommended to run each section individually either by commenting out the code lines in the other sections or by creating cells using # %% if the editor supports it

Author: Bryce Dixon
Version: 20/10/2023  
"""

import numpy as np
import qu_decomp as qu
import os_plot as os


########## Test Array ##########


# the QU algorithm to find eigenvalues of any square matrix is designed to be run by the qu.calculate function, it will return the matrix whose diagonals are equal to the eigenvalues
# lets test this with a random matrix for this first example
matrix = np.array([(5,4,6),(3,9,2),(4,3,7)])
# the only essential input for this is the matrix, we will run with default values for every other input but for clarity they are displyed in the function below
A = qu.calculate(matrix = matrix, max_it = 100, acceptance = 0.001, print_conv = True, save_folder = None, savefilename = None, message_out = True)
# this also prints a plot of the convergence chains which is very useful to see if the chains are non converging
# some arrays may have non converging or not real eigenvalues, this will normally show up as an oscillation in the convergence chain
# if convergence is not reached but the chains are not oscillating, it is likely more iterations are simply needed to reach convergence


########## Coupled Oscillator Test ##########


# lets not work with the coupled oscillator, we can use the os.os_matrix function to generate an M matrix for the eigenvalue equation MA=-w^2A
# we give a mass and a spring constant
oscillator_matrix = os.os_matrix(mass = 10, spring_const = 5)
# alternatively, we can give this 2 different mass values in a list to get the matrix for unequal masses
# we can then plug this into the qu.calculate function to calculate the eigenvalues which will be equal to -w^2
# run with default parameters
A_os = qu.calculate(oscillator_matrix)


########## Frequency against Mass ##########


# os.mass_plot enables us to see how the frequency of the coupled oscillator changed with the mass of the particles for a given spring constant
# lets first set up a mass array to plot over, I chose a geomspace due to the root 1/m expected relationship so to ensure smoothness of the line its best to have more numbers near 1
# a linspace would work fine but the plot would not be very smooth at low numbers unless a large amount of points were used
mass_range = np.geomspace(1,100,100)
# mass_range and spring_const are the only essential parameters so we will run with default parameters
os.mass_plot(mass_range = mass_range, spring_const = 5, eig_max_it = 100, eig_acceptance = 0.001, mass2_range = None, analytical = False, save_folder = None, savefilename = None)
# this function also allows us to plot for differing masses by providing another mass array as the mass2_range parameter and we can plot the analytical solution using the analytical parameter
# when plotting with multiple mass ranges it is highly recommended to use linspaces to ensure the axes are equivalent and the plots are the correct shape
# os.spring_const_plot will produce a simlar frequency vs spring constant plot for a given range of spring constants at a fixed mass, parameter information can be found in the function docstring