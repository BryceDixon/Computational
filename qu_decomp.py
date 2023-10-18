'''
Assessment 1 - Coupled Oscillators
Code for the the QU algorithm

Contains:
    QU class
        decompose function
        compare function
        conv_plot function
    calculate function

Author: Bryce Dixon
Version: 18/10/2023
'''

import numpy as np
import matplotlib.pyplot as plt


# define a class to contain the functions for the QU algorithm

class QU:
    
    def __init__(self, matrix, print_cov):
        """
        Class to preform the QU decomposition method on a 2D square matrix to obtain its eigenvalues

        Parameters
        ----------
        matrix: array of floats
            square matrix to be decomposed
        print_cov: bool
            If true, creates a plot of the eigenvalue convergence chains
        
        Raises
        ------
        assertionError:
            Raised if the matrix is not a square matrix
        """
        
        # ensure matrix is the correct form
        self.matrix = np.array(matrix, dtype=float)
        assert self.matrix.shape == (len(self.matrix), len(self.matrix)), "Matrix must be a square matrix"
        
        # setup the figure for the convergence plot if print_cov is true
        self.print_cov = print_cov
        if self.print_cov is True:
            fig = plt.figure(figsize = (10,7))
            ax = fig.add_subplot(1,1,1)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title('Eigenvalue Convergence Chains')
            self.ax = ax
        
        
    def decompose(self):
        """
        Function to perform the QU decomposition on the matrix
        """

        # setup empty matrices for Q and U which will be modified throughout the loop
        Q = np.zeros_like(self.matrix)
        U = np.zeros_like(self.matrix)
        
        # loop through the columns of the matrix
        for i in range(len(self.matrix)):
            # f here represents c_1, c_2, ..., c_i from the formula (the columns)
            f = self.matrix[:,i]
            # setup a loop to deal with the upper right matrix, the range of i represents the number of filled out positions in the U matrix (1 in 1st column, 2 in 2nd column,...)
            # the diagonals are not dealt with here as this loop will never have n=i
            for n in range(i):
                # f can be dealt with in terms of Q = f/|f| as the subtraction in the formula for f can simplify to (c_i . q_n) * q_n where q_n is the nth column of the Q matrix
                # f is therefore the orthogonal column vector to the ith column of the origial matrix
                # as this loop does not run for i = 0, Q[:,0] can be populated after for f_1 then the loop will begin modifying f
                f = f - (np.dot(self.matrix[:,i], Q[:,n]) * Q[:,n])
                # fill out the off diagonal elements of the U matrix by following (c_i . q_n) where q_n is the nth column of the Q matrix
                U[n,i] = np.dot(self.matrix[:,i], Q[:,n])
            # fill out the Q matrix where each column is the normalised f vector
            Q[:,i] = f/np.linalg.norm(f)
            # fill out the diagonal elements of the U matrix as just the magnitude of vector f    
            U[i,i] = np.linalg.norm(f)
        
        # matrix A is the product of U and Q, this will contain the eigenvalues in the diagonal once convergence is reached
        self.A = np.dot(U, Q)
        
    
    def compare(self, acceptance):
        """
        Function to compare the current A matrix with the previous A matrix
        
        Parameters
        ----------
        acceptance: float
            Acceptance value for convergence, percentage difference between the current and previous eigenvalues as a decimal
        
        Returns
        -------
        A: array of floats
            matrix A, the result of the product of U and Q after decomposition, once convergence is reached, the diagonals will be equal to the eigenvalues of the original matrix
        val: int
            returns 1 if the eigenvalues have converged, 0 otherwise
        """
        val = 0
        for i in range(len(self.A)):
            # for each diagonal element of the current matrix, calculate the percentage difference between it and that of the previous matrix
            conv = (((self.A[i,i] - self.matrix[i,i])**2)**0.5)/self.matrix[i,i]
            conv_abs = (conv**2)**0.5
            # modify val by 1 if the percentage difference is less than the acceptance value for the eigenvalue in position i,i
            if conv_abs < acceptance:
                val += 1
            else:
                val += 0
        
        # redefine the preivous matrix as the current matrix to be used in the next iteration and return the current matrix and val
        self.matrix = self.A
        A = self.A
        return A, val

    def conv_plot(self, iteration):
        """
        Function to plot the eigenvalue convergence chains
        
        Parameters
        ----------
        iteration: int
            the current iteration of the QU algorithm
        
        Raises
        ------
        assertionError:
            Raised if print_cov is not True, this function will only run if print_cov is set to True
        """
        # this function will only run if print_cov is True
        assert self.print_cov is True, "print_cov must be set to True to plot the convergence chains"

        # the convergence chains are plotted by taking the diagonal elements of the current and previous matrices and plotting them at each iteration
        # this would be easier and cleaner to do using 3d matrices and storring all iterations in them however that would become very slow for extremely large matrices as all the iterations will need to be saved in the memory
        x = (iteration, iteration+1)
        for i in range(len(self.A)):
            # y is defined as the previous and current matrix diagonal element for each element seperately
            y = (self.matrix[i,i], self.A[i,i])
            # assign a colour to each element
            c = 'C'+str(i)
            if x[0] == 0:
                # for the first iteration, setup the legend and label
                self.ax.plot(x,y, c=c, label = 'Eigenvalue '+str(i+1))
                self.ax.legend(bbox_to_anchor=(1.2, 1), loc='upper right', borderaxespad=0, fontsize = 10)
            else:
                # subsequent iterations plotted without a label        
                self.ax.plot(x,y, c=c)
                    
    
        

def calculate(matrix, max_it = 100, acceptance = 0.001, print_conv = True, save_folder = None, savefilename = None):
    """
    Function to calculate the eigenvalues of a given 2 x 2 matrix
    
    Parameters
    ----------
    matrix: array of floats
        2x2 matrix to operate on via QU decomposition to obtain its eigenvalues
    max_it: int, optional
        Maximum number of iterations to perform, defaults to 100
    acceptance: float or int, optional
        Acceptance value for convergence, percentage difference between the current and previous eigenvalues as a decimal, defaults to 0.001, iterations will stop once convergence is reached for 2 consecutive steps
    print_conv: bool, optional
        If True, print a plot of the eigenvalue convergence chains, default to True
    save_folder: str, optional
        directory to save the convergence plot to, defaults to None
    savefilename: str, optional
        filename for the convergence plot, defaults to None
    
    Raises
    ------
    AssertionError:
        Raised if max_int is not an integer
    AssertionError:
        Raised if acceptance is not a float or integer
    
    Returns:
    --------
    A: array of floats
        Matrix A, if convergence is reached, the diagonals will be equal to the eigenvalues of the original matrix
    """
    
    # check the inputs are in the correct format
    assert type(max_it) == int, "max_it must be an integer"
    assert type(acceptance) == float or type(acceptance) == int, "acceptance must be a float or integer"
    
    # define an instance of the QU class
    decomp = QU(matrix, print_conv)
    acc_val = 0
    # loop through the maximum number of iterations to run each function of the class for each iteration
    # a maximum number of iterations is used to prevent the code from running indefinitely in the case that convergence is not reached
    for i in range(max_it):
        decomp.decompose()
        if print_conv is True:
            decomp.conv_plot(i)
        A, val = decomp.compare(acceptance)
        # val should be equal to the length of the matrix if the percentage difference between each eigenvalue in position i,i is less than the acceptance value
        if val == len(matrix):
            acc_val += val
            # this code requires the percentage difference to be less than the acceptance value for 2 consecutive iterations for convergence to be reached
            if acc_val == len(matrix)*2:
                # onc econvergence is reached, print the eigenvalues and stop the loop
                print("Reached Covergence after {} iterations".format(i+1))
                for i in range(len(A)):
                    print("Eigenvalue {} = {}".format(i+1, round(A[i,i], 3)))
                break
            else:
                pass
        else:
            # setting acc_val back to zero here ensures acc_val will only be 2 times the length of the matrix if two consecutive iterations are accepted
            acc_val = 0
            if i == max_it-1:
                print("No convergence reached after maximum number of iterations, consider increasing the maximum iterations or the acceptance value")
                break
            else:
                pass
    
    # save the plot provided print_cov is True and save_folder and savefilename are given, if they are not given, the code will still run but the plot will not be saved
    if print_conv is True:        
        if save_folder is not None and savefilename is not None:
            plt.savefig(str(save_folder)+"/"+str(savefilename)+".png", bbox_inches='tight')
        elif save_folder is None and savefilename is None:
            pass
        else:
            print("Figure not saved, you need to provide both savefilename and save_folder to save the figure")
    else:
        if save_folder is not None or savefilename is not None:
            print("No figure to save, print_conv must be set to True to generate the figure")
        
    return A


