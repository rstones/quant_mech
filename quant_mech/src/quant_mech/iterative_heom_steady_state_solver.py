'''
Created on 3 Nov 2017

@author: richard
'''
import numpy as np
from hierarchy_solver import HierarchySolver
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator
import quant_mech.utils as utils

electronic_coupling = 20.
system_hamiltonian = np.array([[100., electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 100.
cutoff_freq = 53.08
temperature = 300. # Kelvin
beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)
mode_params = [] #[(200., 0.25, 10.)]

K =0
environment = []
if mode_params: # assuming that there is a single identical mode on each site 
    environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K)), \
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K))]
else:
    environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]
hs = HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=False)
hs.truncation_level = 6

epsilon = 500.

steady_state = np.zeros(hs.M_dimension(), dtype='complex128')
# act on init steady state guess with D + epsilon = L_s + \gamma + epsilon since we are solving for this
# and getting the steady state back later
# for the system density matrix the \gamma bit is zero so acting on steady state of the system with L_S has no effect
# just use steady state of Liouvillian as initial guess acted on by epsilon bit
steady_state[:hs.system_dimension**2] = np.dot(epsilon*np.eye(hs.system_dimension**2), utils.stationary_state_svd(hs.liouvillian(), np.array([1,0,0,1]))).flatten()

solver_matrix = hs.construct_hierarchy_matrix_steady_state_solver(epsilon)
heom_matrix = hs.construct_hierarchy_matrix_super_fast()

dm_per_tier = hs.dm_per_tier() # this array is num_tiers + 1
print dm_per_tier

error_tolerance = 1.e-3
error = 1.

while error > error_tolerance:
    for i in range(len(dm_per_tier)-1):
        slice_start = (np.sum(dm_per_tier[:i-1]) if i != 0 else 0) * hs.system_dimension**2
        slice_end = np.sum(dm_per_tier[:i+1]) * hs.system_dimension**2
        steady_state[slice_start:slice_end] = solver_matrix[slice_start:slice_end, :].dot(steady_state[slice_start:slice_end])
    
    # update error estimate
    # need to act on steady state with (D + epsilon)^-1 before checking global error, this might be computationally hard
    error = (1. / epsilon) * np.sqrt(np.sum())
    
    # or select some ADOs from lowest active tier, once they are converged, we stop updating that tier
    
    # convergence algorithm:
    # in lowest active tier (initially this is just the lowest tier), select 3ish (?) ADOs
    # act on them with (D + epsilon)^-1 to get steady state (in the Liouville space of the ADO not the hierarchy super-space)
    # maybe construct the (D _epsilon)^-1 for each tier before starting iteration since it is the same for
    # ADOs within a tier (unless we have an underdamped mode and Matsubara terms?)
    # compare with the previous iteration (if this is the first iteration to compare this active tier then continue after 
    # saving as previous ADOs)
    # once comparison is below the error tolerance, we can stop stepping to this tier in the iteration
    # set previous and current ADOs to zero
    # lowest active tier becomes the next tier up
    # repeat until the highest tier (system density matrix) is converged
    
    









