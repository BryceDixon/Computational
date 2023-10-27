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
    mass : float/int or list/tuple of floats/ints
        mass, m, of an individual particle or list or tuple of each particle in the 2 particle coupled oscillator in the form (m1, m2)
    spring_const : float
        spring constant, k, of the spring connecting the particles
    
    Returns
    -------
    M : array of floats
        M array for the coupled oscillators
    """    
    
    k = spring_const
    M = np.zeros(shape = (2,2))
    
    if type(mass) == float or type(mass) == int or type(mass) == np.float64:
        m = mass

        M[0,0] = M[1,1]= -(2 * k)/m
        M[0,1] = M[1,0] = k/m
    
    if type(mass) == list or type(mass) == tuple or type(mass) == np.ndarray:
        mass = np.array(mass)
        m_1 = mass[0]
        m_2 = mass[1]
        
        M[0,0] = -(2 * k)/m_1
        M[0,1] = k/m_1
        M[1,0] = k/m_2
        M[1,1] = -(2 * k)/m_2
          
    return M


def mass_plot(mass_range, spring_const, eig_max_it = 100, eig_acceptance = 0.001, mass2_range = None, analytical = False, save_folder = None, savefilename = None):
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
    mass2_range : array or list of floats, optional
        array of masses, m_2, of the 2nd particle in the two particle coupled oscillator, (assumes mass_range is the 1st paricle), defaults to None
    analytical : bool, optional
        If True, plot the analytical solution derived using the characteristic equation on the graph, if False do not plot that, only works for a single mass_range, defaults to False
    save_folder : _type_, optional
        _description_, by default None
    savefilename : _type_, optional
        _description_, by default None
    """
    
    A_array = np.zeros(shape = (len(mass_range),2,2))
    
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_ylabel('Frequency ($Rads^{-1}$)')
    ax.set_title('Frequency against Mass for a coupled oscillator with spring constant of {} N/m'.format(spring_const))
    
    if mass2_range is None:
    
        f_1 = []
        f_2 = []
    
        for i in range(len(mass_range)):

            M = os_matrix(mass_range[i], spring_const)
            A = qu.calculate(M, eig_max_it, eig_acceptance, False, message_out=False)
            A_array[i,:,:] = A
            if analytical is True:
                f_1.append(np.sqrt(3*spring_const/mass_range[i]))
                f_2.append(np.sqrt(spring_const/mass_range[i]))
    
        ax.set_xlabel('Particle Mass ($kg$)')
        labels = ['Eigenvalue 1, $\\omega_1 = \\sqrt{\\frac{3k}{m}}$', 'Eigenvalue 2, $\\omega_2 = \\sqrt{\\frac{k}{m}}$']
        for i in range(len(A_array[0,:,:])):
            # -omega = square root of the eigenvalue so to get omega we must flip the sign of the eigenvalue and square root
            omega = np.sqrt(A_array[:,i,i]*-1)
            ax.plot(mass_range, omega, color = 'C'+str(i), label = labels[i])
        if analytical is True:    
            ax.plot(mass_range, f_1, linestyle = '--', color = 'k', label = 'Analytical $\\omega_1 = \\sqrt{\\frac{3k}{m}}$')
            ax.plot(mass_range, f_2, linestyle = ':', color = 'k', label = 'Analytical $\\omega_2 = \\sqrt{\\frac{k}{m}}$')
    
    if mass2_range is not None:
        
        assert len(mass2_range) == len(mass_range), 'mass2_range must be the same length as mass_range'
        
        for i in range(len(mass_range)):

            M = os_matrix((mass_range[i],mass2_range[i]), spring_const)
            A = qu.calculate(M, eig_max_it, eig_acceptance, False, message_out=False)
            A_array[i,:,:] = A
            
        ax.set_xlabel('Particle 1 Mass ($kg$)')
        for i in range(len(A_array[0,:,:])):
            # -omega = square root of the eigenvalue so to get omega we must flip the sign of the eigenvalue and square root
            omega = np.sqrt(A_array[:,i,i]*-1)
            ax.plot(mass_range, omega, color = 'C'+str(i), label = 'Eigenvale '+str(i+1))
            if max(mass2_range) - min(mass2_range) != 0:
                ax2 = ax.twiny()
                ax2.plot(mass2_range, omega, color = 'C'+str(i))
        try:
            ax2.set_xlabel('Particle 2 Mass ($kg$)')
        except:
            pass

    ax.legend()
        
    # save the plot provided  save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved      
    if save_folder is not None and savefilename is not None:
        plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
    elif save_folder is None and savefilename is None:
        # no message displayed if both save_folder and savefilename are none, it's assumed the user did not intend to save the plot
        pass
    else:
        print("Figure not saved, you need to provide both savefilename and save_folder to save the figure")

    
    
def spring_const_plot(spring_const_range, mass, eig_max_it = 100, eig_acceptance = 0.001, save_folder = None, savefilename = None):
    """Function to generate a plot of frequency against mass for a system of equal mass coupled oscillators

    Parameters
    ----------
    spring_const_range : array or list of floats
        array of spring constants, k, of the system to be plotted against frequency
    mass : float
        mass, m, of an individual particle
    eig_max_it : int, optional
        maximum number of iterations to perform when computing the eigenvalues using the QU algorithm, defaults to 100
    eig_acceptance : float, optional
        acceptance value for convergence of the QU algorithm, percentage difference between the current and previous eigenvalue as a decimal, defaults to 0.001
    save_folder : _type_, optional
        _description_, by default None
    savefilename : _type_, optional
        _description_, by default None
    """
    
    A_array = np.zeros(shape = (len(spring_const_range),2,2))
    
    for i in range(len(spring_const_range)):

        M = os_matrix(mass, spring_const_range[i])
        A = qu.calculate(M, eig_max_it, eig_acceptance, False, message_out=False)
        A_array[i,:,:] = A
    
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Spring Constant ($Nm^{-1}$)')
    ax.set_ylabel('Frequency ($Rads^{-1}$)')
    ax.set_title('Frequency against Spring Constant for a coupled oscillator with particle mass of {} kg'.format(mass))
    
    labels = ['Eigenvalue 1, $\\omega_1 = \\sqrt{\\frac{3k}{m}}$', 'Eigenvalue 2, $\\omega_2 = \\sqrt{\\frac{k}{m}}$']
    for i in range(len(A_array[0,:,:])):
        # -omega = square root of the eigenvalue so to get omega we must flip the sign of the eigenvalue and square root
        omega = np.sqrt(A_array[:,i,i]*-1)
        ax.plot(spring_const_range, omega, color = 'C'+str(i), label = labels[i])
    
    ax.legend()

    # save the plot provided  save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved      
    if save_folder is not None and savefilename is not None:
        plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
    elif save_folder is None and savefilename is None:
        # no message displayed if both save_folder and savefilename are none, it's assumed the user did not intend to save the plot
        pass
    else:
        print("Figure not saved, you need to provide both savefilename and save_folder to save the figure") 