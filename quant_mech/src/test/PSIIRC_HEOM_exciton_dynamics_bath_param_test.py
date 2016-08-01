'''
Created on 26 Jul 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.time_evolution as te
import quant_mech.time_utils as tutils
from quant_mech.hierarchy_solver import HierarchySolver

np.set_printoptions(precision=3, linewidth=150, suppress=True)

time_step = 0.001
duration = 5 # picoseconds

average_site_CT_energies = np.array([15260., 15190., 15000., 15100., 15030., 15020., 15992., 16132.])

# site-CT couplings
couplings = np.array([[0,150.,-42.,-55.,-6.,17.,0,0],
                     [0,0,-56.,-36.,20.,-2.,0,0],
                     [0,0,0,7.,46.,-4.,70.,0],
                     [0,0,0,0,-5.,37.,0,0],
                     [0,0,0,0,0,-3.,70.,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,40.],
                     [0,0,0,0,0,0,0,0]])
system_hamiltonian = np.diag(average_site_CT_energies) + couplings + couplings.T
system_hamiltonian = system_hamiltonian[:6,:6]

truncation_level = 8

# try different reorg energies
# reorg_energy_values = [35., 70., 100.]
# cutoff_freq = 40.
# temperature = 300.
# 
# for reorg_energy in reorg_energy_values:
#     print 'Started calculating HEOM dynamics for reorg_energy = ' + str(reorg_energy) + 'cm-1 at ' + str(tutils.getTime())
#     hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature)
#     init_state = np.zeros(system_hamiltonian.shape)
#     init_state[0,0] = 1. # exciton basis
#     init_state = np.dot(hs.system_evectors, np.dot(init_state, hs.system_evectors.T)) # transform to site basis for HEOM calculation
#     dm_history, time = hs.converged_time_evolution(init_state, truncation_level, truncation_level, time_step, duration)
#     exciton_dm_history = hs.transform_to_exciton_basis(dm_history)
#     np.savez('../../data/PSIIRC_HEOM_dynamics_reorg_energy_'+str(int(reorg_energy))+'_wavenums.npz', exciton_dm_history=exciton_dm_history, \
#                                             time=time, init_state=init_state, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
#                                             system_hamiltonian=system_hamiltonian)
    
# try different cutoff freqs
reorg_energy = 35.
cutoff_freq_values = [40., 70., 100.]
temperature = 300.

for cutoff_freq in cutoff_freq_values:
    print 'Started calculating HEOM dynamics for cutoff_freq = ' + str(cutoff_freq) + 'cm-1 at ' + str(tutils.getTime())
    hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature)
    init_state = np.zeros(system_hamiltonian.shape)
    init_state[0,0] = 1. # exciton basis
    init_state = np.dot(hs.system_evectors, np.dot(init_state, hs.system_evectors.T)) # transform to site basis for HEOM calculation
    dm_history, time = hs.converged_time_evolution(init_state, truncation_level, truncation_level, time_step, duration)
    exciton_dm_history = hs.transform_to_exciton_basis(dm_history)
    np.savez('../../data/PSIIRC_HEOM_dynamics_cutoff_freq_'+str(int(cutoff_freq))+'_wavenums.npz', exciton_dm_history=exciton_dm_history, \
                                            time=time, init_state=init_state, reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature, \
                                            system_hamiltonian=system_hamiltonian)

print 'Calculations finished'

