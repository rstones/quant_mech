'''
Created on 31 Jul 2013

@author: rstones

Contains various utility functions for performing quantum mechanical calculations.
eg. Partial traces, constructing step up/down operators etc...

TODO:
1) Functions to test physicality of density matrix eg. check trace of rho = 1, rho**2 <= 1 over time
2) Unit tests

Test comment again

'''

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

EV_TO_WAVENUMS = 8065.5
EV_TO_JOULES = 1.6022e-19
KELVIN_TO_WAVENUMS = 0.6949
WAVENUMS_TO_INVERSE_PS = 0.06*np.pi
WAVENUMS_TO_JOULES = 1.98630e-23
WAVENUMS_TO_NM = 10000000.

def lowering_operator(basis_size=2):
    op = np.zeros((basis_size, basis_size))
    for i in range(1, basis_size):
        op[i-1, i] = np.sqrt(i)
    return op
    
def raising_operator(basis_size=2):
    return lowering_operator(basis_size).T
    
def orthog_basis_set(basisSize):
    basis_set = []
    array = np.zeros(basisSize)

    for i in range(0, basisSize):
        array[i] = 1
        temp_array = array.copy()
        basis_set.append(temp_array)
        array[i] = 0

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
    reduced_dm = np.zeros((dim_reduced_dm, dim_reduced_dm), dtype=density_matrix.dtype)
    I_sys = np.eye(dim_reduced_dm, dim_reduced_dm, dtype=density_matrix.dtype)

    for i in range(len(trace_basis)):
        trace_state = np.kron(I_sys, trace_basis[i]) if order == 2 else np.kron(trace_basis[i], I_sys)
        reduced_dm += np.dot(trace_state, np.dot(density_matrix, trace_state.T))

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
    return (np.exp(freq/(KELVIN_TO_WAVENUMS*temperature)) - 1) ** -1
    
def hamiltonian_to_picosecs(hamiltonian):
    return 0.06*np.pi*hamiltonian

########################################################################################################
#
# Returns thermal state of harmonic oscillator
#
######################################################################################################## 
def thermal_state(freq, temp, basis_size):
    kB = 0.695  # boltzmann constant in wavenumbers
    density_matrix = np.zeros((basis_size, basis_size))
    Z = 0  #init normalisation constant
    
    for i in range(basis_size):
        x = np.exp(-((i + 0.5)*freq) / (kB*temp))
        density_matrix[i,i] = x
        Z += x
    
    return density_matrix / Z

def general_thermal_state(hamiltonian, temp):
    kB = 0.695  # boltzmann constant in wavenumbers
    basis_size = hamiltonian.shape[0]
    density_matrix = np.zeros((basis_size, basis_size))
    Z = 0  #init normalisation constant
    
    for i in range(basis_size):
        x = np.exp(-hamiltonian[i,i] / (kB*temp))
        density_matrix[i,i] = x
        Z += x
    
    return density_matrix / Z

def general_thermal_state_beta(hamiltonian, beta):
    basis_size = hamiltonian.shape[0]
    density_matrix = np.zeros((basis_size, basis_size))
    Z = 0  #init normalisation constant
    
    for i in range(basis_size):
        x = np.exp(-hamiltonian[i,i] * beta)
        density_matrix[i,i] = x
        Z += x
    
    return density_matrix / Z

'''
Function to find stationary state of Liouvillian by diagonalisation
Assumes Liouvillian is in basis which results from density_matrix.flatten()

TODO: add check for multiple stationary states ie. warn about evectors that have evalues of same magnitude as smallest evalue
'''
def stationary_state(liouvillian, populations=None):
    stationary_state = stationary_state_unnormalised(liouvillian)
    
    dimDM = np.sqrt(stationary_state.shape[0])
    if dimDM % 1 == 0: # check that dimension is an integer, therefore a square number
        stationary_state.shape = (dimDM, dimDM)    
        return (stationary_state / np.trace(stationary_state)).flatten()
    elif dimDM % 1 != 0 and populations.all():
        return classical_stationary_state(liouvillian)
    elif populations.any(): # check that populations mask has been provided
        return stationary_state / np.dot(populations, stationary_state)        
    else:
        raise Exception("stationary_state is not square and populations is not defined!")

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

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.
    
    The algorithm used by this function is based on the singular value
    decomposition of `A`.
    
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    
    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """
    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    while True:
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        if ns.shape[1] == 0:
            atol *= 10.
            #print "stationary_state_svd function increasing atol to " + str(atol) + " to find nullspace"
        elif ns.shape[1] == 1:
            break
        elif ns.shape[1] and ns.shape[1] > 1:
            print "WARNING: more than one stationary state found using nullspace function!"
            return ns.T[1]
    return ns

'''
Find stationary state of Liouvillian using a singular value decomposition which is much faster than diagonalisation
and finding the eigenvector associated with the zero eigenvalue. An array indicating the position of populations of 
the density vector allows correct normalization (1 for position elements, 0 for coherence elements).
'''
def stationary_state_svd(liouvillian, density_vector_populations):
    ss = nullspace(liouvillian); ss = ss.flatten()
    return ss / np.sum(ss * density_vector_populations) # normalise

# differentiates a function defined for points on an array
# f is dependent variable
# x is independent variable
def differentiate_function(f, x):
    dx = x[1] - x[0]
    result = np.empty(f.shape, dtype='complex128')
    for i,v in enumerate(f):
        result[i] = (f[i+1] - f[i]) / dx if i < f.size-1 else 0
    return result

'''
Sorts set of evals and evecs in order from lowest to highest energy 

evecs must have evectors defined along rows
'''
def sort_evals_evecs(evals, evecs):
    evals_sorted = np.sort(evals)
    evecs_sorted = []
    for i in evals_sorted:
        for j,v in enumerate(evals):
            if v == i:
                evecs_sorted.append(evecs[j])
                break
    return np.array(evals_sorted), np.array(evecs_sorted, dtype='complex')

'''
Returns eigenvalues and eigenvectors of the matrix sorted in ascending order of energy

Note: Eigenvectors are returned along the rows of the output array in contrast to in columns
like in np.linalg.eig. This makes it easier to iterate over the set of excitons in subsequent
calculations. 
'''
def sorted_eig(M):
    evals, evecs = np.linalg.eig(M)
    return sort_evals_evecs(evals, evecs.T) # pass transpose of evecs as sort_evals_evecs expecting evectors defined along rows

'''
Returns Hamiltonian for a quantum harmonic oscillator
'''
def vibrational_hamiltonian(freq, basis_size):
        H_vib = np.zeros((basis_size, basis_size))    
        for i in range(basis_size):
            H_vib[i,i] = (i + 0.5)*freq
        return H_vib
    
    
def detailed_balance_backward_rate(forward_rate, omega, temperature):
    return np.exp(-omega / (KELVIN_TO_WAVENUMS*temperature)) * forward_rate