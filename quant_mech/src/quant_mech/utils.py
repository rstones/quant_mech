'''
Created on 31 Jul 2013

@author: rstones

Contains various utility functions for performing quantum mechanical calculations.
eg. Partial traces, constructing step up/down operators etc...

TODO:
1) Functions to test physicality of density matrix eg. check trace of rho = 1, rho**2 <= 1 over time
2) Unit tests

'''

import numpy as np
import scipy.linalg as la
import copy

EV_TO_WAVENUMS = 8065.5
EV_TO_JOULES = 1.6022e-19
KELVIN_TO_WAVENUMS = 0.695

def lowering_operator(basis_size=2):
    op = np.zeros((basis_size, basis_size))
    for i in range(1, basis_size):
        op[i-1, i] = np.sqrt(i)
    return op
    
def raising_operator(basis_size=2):
    return lowering_operator(basis_size).T
    
def orthog_basis_set(basisSize):
    basis_set = []
    array = np.zeros((1, basisSize))

    for i in range(0, basisSize):
        array[0][i] = 1
        temp_array = copy.deepcopy(array)
        basis_set.append(temp_array.T)
        array[0][i] = 0

    return basis_set

#########################################################################################################    
# Function to perform a partial trace over a density matrix which exists in a product space
# of two Hilbert spaces
#
# @param density_matrix (array)
# @param trace_basis (List) The basis to trace over
# @param order (int) The order in the tensor product which the basis to
#                    be traced out was used. Should take value 1 or 2.
#
# @returns redDM (array) 
#########################################################################################################    
def partial_trace(density_matrix, trace_basis, order=2):
     
    dim_reduced_dm = density_matrix.shape[0] / trace_basis[0].shape[0]
    reduced_dm = np.zeros((dim_reduced_dm, dim_reduced_dm))
    I_sys = np.eye(dim_reduced_dm, dim_reduced_dm)

    for i in range(len(trace_basis)):
        trace_state = np.kron(I_sys, trace_basis[i]) if order == 2 else np.kron(trace_basis[i], I_sys)
        reduced_dm += np.dot(trace_state.T, np.dot(density_matrix, trace_state))

    return reduced_dm
  
def commutator(operator_1, operator_2):
    return np.dot(operator_1, operator_2) - np.dot(operator_2, operator_1)

def anticommutator(operator_1, operator_2):
    return np.dot(operator_1, operator_2) + np.dot(operator_2, operator_1)
    
########################################################################################################
#
# Returns Planck distribution for a mode of a given freq (in wavenumbers) at given temperature (in K)
#
########################################################################################################    
def planck_distribution(freq, temperature):
    return (np.exp(freq/(0.695*temperature)) - 1) ** -1
    
def hamiltonian_to_picosecs(hamiltonian):
    return 0.06*np.pi*hamiltonian

def thermal_state(freq, temp, basis_size):
    kB = 0.695  # boltzmann constant in wavenumbers
    density_matrix = np.zeros((basis_size, basis_size))
    Z = 0  #init normalisation constant
    
    for i in range(basis_size):
        x = np.exp(-((i + 0.5)*freq) / (kB*freq))
        density_matrix[i,i] = x
        Z += x
    
    return density_matrix / Z

# find stationary state of Liouvillian by diagonalisation
# assumes Liouvillian is in basis which results from density_matrix.flatten()
# maybe put in open_systems module
def stationary_state(liouvillian):
    stationary_state = stationary_state_unnormalised(liouvillian)
    
    dimDM = np.sqrt(stationary_state.shape[0])
    stationary_state.shape = (dimDM, dimDM)    
    return (stationary_state / np.trace(stationary_state)).flatten()

def stationary_state_unnormalised(liouvillian):
    evalues, evectors = la.eig(liouvillian)
    currentLargest = float('-inf')
    currentLargestIndex = 0
    for i,e in enumerate(evalues):
        if e > currentLargest:
            currentLargest = e
            currentLargestIndex = i
    return evectors[:, currentLargestIndex]

# find stationary state of liouvillian for system of classical rate equations
# (ie. density matrix for system should be diagonal)
def classical_stationary_state(liouvillian):
    stationary_state = stationary_state_unnormalised(liouvillian)
    stationary_state = np.diag(stationary_state)
    stationary_state = stationary_state / np.trace(stationary_state)
    return np.diagonal(stationary_state)
