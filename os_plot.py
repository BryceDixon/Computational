"""
Assessmetn 1 - Coupled Oscillators
Code for generating and plotting the coupled oscillators
    
Contains:
    os_matrix function
    plot function
    
Author: Bryce Dixon
Version: 18/10/2023
"""

import numpy as np
import matplotlib.pyplot as plt
import qu_decomp as qu

def os_matrix(mass, spring_const):
    """Function to generate the 2x2 M matrix for a pair of equal mass coupled oscillators with equal spring constants for the eigenvalue equation:
    
        M A = -omega^2 A
    
    In the form:

        M = (-2k/m   k/m )
            ( k/m   -2k/m)
    
    Parameters
    ----------
    mass : float
        mass, m, of an individual particle
    spring_const : float
        spring constant, k, of the spring connecting the particles
    
    Returns
    -------
    M : array of floats
        M array for the coupled oscillators
    """    
    
    m = mass
    k = spring_const
    
    M = np.zeros(shape = (2,2))
    M[0,0] = M[1,1]= -(2 * k)/m
    M[0,1] = M[1,0] = k/m
    
    return M


def plot(mass_range, spring_const, eig_max_it = 100, eig_acceptance = 0.001, save_folder = None, savefilename = None):
    """Function to generate a plot of frequency against mass for a system of equal mass coupled oscillators

    Parameters
    ----------
    mass_range : array or list of floats
        array of masses, m, of an individual particle to be plotted against frequency
    spring_const : float
        spring constant, k, of the spring connecting the particles
    eig_max_it : int, optional
        maximum number of iterations to perform when computing the eigenvalues using the QU algorithm, defaults to 100
    eig_acceptance : float, optional
        acceptance value for convergence of the QU algorithm, percentage difference between the current and previous eigenvalue as a decimal, defaults to 0.001
    save_folder : _type_, optional
        _description_, by default None
    savefilename : _type_, optional
        _description_, by default None
    """
    
    A_array = np.zeros(shape = (len(mass_range),2,2))
    
    for i in range(len(mass_range)):

        M = os_matrix(mass_range[i], spring_const)
        A = qu.calculate(M, eig_max_it, eig_acceptance, False)
        A_array[i,:,:] = A
    
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Particle Mass ($kg$)')
    ax.set_ylabel('Frequency ($Rads^{-1}$)')
    
    for i in range(len(A_array[0,:,:])):
        # -omega = square root of the eigenvalue so to get omega we must flip the sign of the eigenvalue and square root
        omega = np.sqrt(A_array[:,i,i]*-1)
        ax.plot(mass_range, omega, color = 'C'+str(i), label = 'Eigenvalue '+str(i+1))
    
    ax.legend()

    
    
 