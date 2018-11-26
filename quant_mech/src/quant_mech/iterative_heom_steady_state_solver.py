'''
Created on 3 Nov 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from hierarchy_solver import HierarchySolver
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator
import quant_mech.utils as utils

np.set_printoptions(precision=6, linewidth=1000, suppress=True)

'''
First try the most explicit method directly using the inverse (D + \epsilon)^-1
This is probably not too memory intensive for a two or three level system
For this method we can use the global error to define convergence

Next try solving for the vector (D + \epsilon)^-1 \rho and doing the tier by tier convergence check...
in lowest active tier (initially this is just the lowest tier), select 3ish (?) ADOs
act on them with (D + epsilon)^-1 to get steady state (in the Liouville space of the ADO not the hierarchy super-space)
maybe construct the (D _epsilon)^-1 for each tier before starting iteration since it is the same for
ADOs within a tier (unless we have an underdamped mode and Matsubara terms?)
compare with the previous iteration (if this is the first iteration to compare this active tier then continue after 
saving as previous ADOs)
once comparison is below the error tolerance, we can stop stepping to this tier in the iteration
set previous and current ADOs to zero
lowest active tier becomes the next tier up
repeat until the highest tier (system density matrix) is converged
'''

electronic_coupling = 20.
system_hamiltonian = np.array([[100., electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 1200. # 100.
cutoff_freq = 53.08
temperature = 300. # Kelvin
beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)
mode_params = [] #[(200., 0.25, 10.)]

K = 0
environment = []
if mode_params: # assuming that there is a single identical mode on each site 
    environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K)), \
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K))]
else:
    environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]
hs = HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=True)
hs.truncation_level = 40

epsilon = 2000.

init_guess = np.zeros(hs.M_dimension(), dtype='complex128')
system_thermal_state = utils.general_thermal_state_beta(hs.system_hamiltonian, beta)
init_guess[:hs.system_dimension**2] = system_thermal_state.flatten()

print 'Init solution guess...'
print init_guess[:hs.system_dimension**2]

# direct steady_state
heom_matrix = hs.construct_hierarchy_matrix_super_fast()
pops = np.zeros(heom_matrix.shape[0])
pops[[0,3]] = 1
ss = spla.eigs(heom_matrix.tocsc(), k=1, sigma=None, which='SM', v0=pops/np.sum(pops))[1]
ss = ss / pops.dot(ss)
print 'Direct steady state calculation'
print ss[:4]

steady_state = hs.efficient_steady_state_solver(epsilon, init_guess)

print 'testing the steady state on the time propagtor...'
print heom_matrix.dot(steady_state)[:hs.system_dimension**2]

print 'the final steady state...'
print steady_state[:hs.system_dimension**2]
     
    
    









