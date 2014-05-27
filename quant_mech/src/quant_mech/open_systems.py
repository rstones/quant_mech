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

'''
Brownian oscillator spectral density
'''    
def bo_spectral_density(E_reorg, w, w_c):
    return ((2. * E_reorg) / np.pi) * ((w * w_c) / (w**2 + w_c**2))

'''
Bath dependent relaxation rate between excitons
spectral_density should be a function taking parameters reorganisation energy, transition freq and cutoff freq (in that order, see
bo_spectral_density function for example)

Currently I set w_c = w for spectral density but I don't think this is a general thing, need to change it at some point....
'''
def gamma(w, w_c, spectral_density, E_reorg, temperature):
    return 2. * np.pi * spectral_density(E_reorg, np.abs(w), np.abs(w)) * np.abs(utils.planck_distribution(w, temperature))

'''
Total relaxation rate between excitons 
'''
def Gamma(w, w_c, spectral_density, E_reorg, temperature, ex1, ex2):
    return gamma(w, w_c, spectral_density, E_reorg, temperature) * np.dot(np.array([np.abs(i)**2 for i in ex1]), np.array([np.abs(i)**2 for i in ex2]))