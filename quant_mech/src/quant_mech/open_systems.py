'''
Created on 31 Jul 2013

@author: rstones

Module containing methods for numerical study of open quantum systems.
1) create Liouvillian superoperator from Hamiltonian and Lindblad operators
2) Markovian dissipator
3) partial trace
4) etc

'''
import numpy as np
import quant_mech.utils as utils

# Function to construct Liouvillian super operator
# jump_operators should be list of tuples (first entry of tuple is lindblad operator, second is rate)
def super_operator(H, jump_operators):
    print "Constructing super-operator..."
    I = np.eye(H.shape[0], H.shape[1])
    L = -1.j * (np.kron(H, I) - np.kron(I, H))
    if jump_operators:
        L += incoherent_super_operator(jump_operators)
    return L

def incoherent_super_operator(jump_operators):
    jo_shape = jump_operators[0][0].shape
    L = np.zeros((jo_shape[0]**2, jo_shape[1]**2))
    I = np.eye(jo_shape[0])
    for tup in jump_operators:
        A = tup[0]
        A_dagger = A.conj().T
        A_dagger_A = np.dot(A_dagger, A)
        L += tup[1] * (np.kron(A, A) - 0.5 * np.kron(A_dagger_A, I) - 0.5 * np.kron(I, A_dagger_A))
    return L

def markovian_dissipator(density_matrix, lindblad_operator, dissipation_rate):
    lindblad_operator_dagger = lindblad_operator.conj().T
    return dissipation_rate * (np.dot(lindblad_operator, np.dot(density_matrix, lindblad_operator_dagger)) - 0.5 * utils.anticommutator(np.dot(lindblad_operator_dagger, lindblad_operator), density_matrix))

'''
Maybe setup HierarchySolver class which has attributes for various system parameters
(eg. vector containing frequencies(\nu), Hamiltonian parameters, spectral density etc)

'''

# solve differential equation using 4th order Runge-Kutta method (used here for hierarchy code but should probably move to more appropriate module)
# func is a function that must take 2 parameters, step and state
def RK4(func, init_state, interval, step_size):
    state_evolution = []
    state = init_state
    for step in range(int(interval/step_size)):
        k1 = func(step, state)
        k2 = func(step+0.5*step_size, state + (0.5*step_size*k1))
        k3 = func(step+0.5*step_size, state + (0.5*step_size*k2))
        k4 = func(step+step_size, state + step_size*k3)
  
        state = state + (1./6.) * step_size * (k1 + 2.*k2 + 2.*k3 + k4)
        state_evolution.append(state)
        
    return state_evolution

# top level function that returns system density matrix time evolution
def hierarchy_equations():
    # solve set of hierarchically coupled equations in matrix form (p' = M.p)
    # system density matrix and auxiliary density matrices in Liouville space are appended into one column vector (p)
    # construct init state with non-zero init system density matrix and auxiliary density matrices set to zero
    # construct matrix M from hierarchy operators expressed in Liouville space
    pass

# functional form of nth level in hierarchy
# time_step is time
# dm is density matrix of current level at given time
# dm_n_plus_1 is density matrix at next highest level at given time
# dm_n_minus_1 is density matrix at next lowest level at given time
# n is vector for current level
def nth_ode(time_step, dm, dm_n_plus_1, dm_n_minus_1, n):
    # 
    pass

# phonon-induced relaxation operator
def theta():
    pass

# phonon-induced relaxation operator
def phi():
    pass

# calculates order at which to terminate hierarchy
def terminator():
    pass

# Drude spectral_density for basic hierarchy expansion
def drude_spectral_density():
    pass