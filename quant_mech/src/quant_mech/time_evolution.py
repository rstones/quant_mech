'''
Created on 31 Jul 2013

@author: rstones
'''
import numpy as np
import scipy.linalg as la
import quant_mech.utils as utils

######################################################################################################################
#
# Function which propagates a density matrix through time by stepwise application of time evolution operator
# @param hamiltonian 2D array
# @param initDM 2D array
# @param timestep float
# @param duration float Should be provided in picoseconds
# @param trace_basis list (default None) If it is more convenient to consider dynamics of a reduced system will trace 
#                    out irrelevant degrees of freedom at each time step
# @param wave_nums boolean (default True) If the hamilonian is provided in wavenumber units then it will automatically 
#                   be scaled to agree with the unit of the duration parameter which should be picoseconds.
# 
# @return dm_history list containing density_matrix at each timestep
#
#######################################################################################################################
def von_neumann_eqn(init_density_matrix, hamiltonian, duration, timestep, trace_basis=None, wave_nums=True):

    if wave_nums:
        hamiltonian = utils.hamiltonian_to_picosecs(hamiltonian)

    timestep_operator = la.expm(-1j*hamiltonian*timestep)
    timestep_operator_dagger = timestep_operator.conj().T
    density_matrix = init_density_matrix
    dm_history = []
    if trace_basis:
        dm_history.append(utils.partial_trace(density_matrix, trace_basis))
    else:
        dm_history.append(density_matrix) 

    for step in range(int(duration/timestep)):
        density_matrix = np.dot(np.dot(timestep_operator, density_matrix), timestep_operator_dagger)
        if trace_basis:
            dm_history.append(utils.partial_trace(density_matrix, trace_basis))
        else:
            dm_history.append(density_matrix)

    return dm_history
    
    
############################################################################################################
# Function to solve master equation optionally with a markovian dissipator using Runge-Kutta 4th order
# methods. Will return the final density matrix obtained and a history of the reduced density matrix for all
# time steps.
# @param initDensityMatrix 
# @param hamiltonian
# @param duration (should be provided in picoseconds)
# @param timestep 
# @param (optional) trace_basis (basis to trace over to get reduced density matrix)
# @param (optional) jump_operators list of tuples (jump operator, rate) 
# @param (optional) hamiltonInWaveNums 
# @param (optional) lindbladOperator 
# @param (optional) decayRateMatrix
#
############################################################################################################
def markovian_master_eqn_RK4(init_density_matrix, hamiltonian, duration, timestep, trace_basis=None, jump_operators=None, hamiltonianInWaveNums=True):

    if hamiltonianInWaveNums:
        hamiltonian = 0.06*np.pi*hamiltonian

    density_matrix = init_density_matrix
    dm_history = []
    if trace_basis:
        dm_history.append(utils.partial_trace(density_matrix, trace_basis))
    else:
        dm_history.append(density_matrix) 

    for step in range(0, int(duration/timestep)):
        if step % 100 == 0:
            print "Reached step " + str(step)
        
        k1 = timestep * master_equation(density_matrix, hamiltonian, jump_operators)
        k2 = timestep * master_equation(density_matrix + (0.5*k1), hamiltonian, jump_operators)
        k3 = timestep * master_equation(density_matrix + (0.5*k2), hamiltonian, jump_operators)
        k4 = timestep * master_equation(density_matrix + k3, hamiltonian, jump_operators)
  
        density_matrix = density_matrix + (1./6.) * (k1 + 2.*k2 + 2.*k3 + k4)
        if trace_basis:
            dm_history.append(utils.partial_trace(density_matrix, trace_basis))
        else:
            dm_history.append(density_matrix)
  
    return dm_history


# maybe should be in open_systems module
def lindblad_dissipator(density_matrix, lindblad_operator, rate):
    return rate * (np.dot(lindblad_operator, np.dot(density_matrix, lindblad_operator.conj().T)) - 0.5*utils.anticommutator(np.dot(lindblad_operator.conj().T, lindblad_operator), density_matrix))

# maybe should be in open_systems module
def master_equation(density_matrix, hamiltonian, jump_operators):
    result = -1.j * utils.commutator(hamiltonian, density_matrix)
    if jump_operators:
        for jo in jump_operators:
            result += lindblad_dissipator(density_matrix, jo[0], jo[1])
    return  result

#####################################################################################################
#
# Function for evolving an initial wavevector through time for the given hamiltonian and duration.
# Returns the final obtained state and the contributions from each basis state for every timestep.
#
#####################################################################################################
def schrodinger_eqn(hamiltonian, initState, timestep, duration):

    timestepOperator = la.expm(-1j*hamiltonian*timestep)
    finalState = initState

    dimH = hamiltonian.shape[0]
    stateContributions = np.zeros((dimH, int(np.floor(duration/timestep))))

    for step in range(int(duration/timestep)):
        finalState = np.dot(timestepOperator, finalState)

        for state in range(dimH):
            stateContributions[state, int(step)] = abs(finalState[state])

    return finalState, stateContributions