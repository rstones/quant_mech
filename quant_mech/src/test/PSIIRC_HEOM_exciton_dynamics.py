'''
Created on 1 Jun 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.time_evolution as te
import quant_mech.time_utils as tutils
from quant_mech.hierarchy_solver import HierarchySolver

np.set_printoptions(precision=3, linewidth=150, suppress=True)

print 'Calculating time evolution...'
start_time = tutils.getTime()

time_step = 0.01
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
reorg_energy = 35.
cutoff_freq = 40.
temperature = 300.
mode_params = [(342., 342.*0.4, 100.)]
hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, underdamped_mode_params=mode_params)

init_state = np.zeros(system_hamiltonian.shape)
init_state[0,0] = 1. # exciton basis
init_state = np.dot(hs.system_evectors, np.dot(init_state, hs.system_evectors.T)) # transform to site basis for HEOM calculation

#dm_history, time = hs.converged_time_evolution(init_state, 6, 6, time_step, duration)
hs.init_system_dm = init_state
hs.truncation_level = 6
dm_history, time = hs.calculate_time_evolution(time_step, duration)
exciton_dm_history = hs.transform_to_exciton_basis(dm_history)

end_time = tutils.getTime()
print 'Calculation took ' + str(tutils.duration(end_time, start_time))

# print 'Calculating steady state...'
# start_time = tutils.getTime()
#   
# steady_state = hs.calculate_steady_state(8,8)
# exciton_steady_state = np.dot(hs.system_evectors.T, np.dot(steady_state, hs.system_evectors))
# 
# end_time = tutils.getTime()
# print 'Calculation took ' + str(tutils.duration(end_time, start_time))
# print 'Exciton steady state trace: ' + str(np.trace(exciton_steady_state))
# 
# print steady_state
# print exciton_steady_state

print 'Saving data...'
np.savez('../../data/PSIIRC_mode_HEOM_6_tiers_data.npz', dm_history=dm_history, exciton_dm_history=exciton_dm_history, time=time, system_hamiltonian=system_hamiltonian, \
                                                reorg_energy=reorg_energy, cutoff_freq=cutoff_freq, temperature=temperature)

# colours = ['b', 'g', 'r', 'c', 'm', 'y']
# # for i in range(system_hamiltonian.shape[0]):
# #     plt.axhline(exciton_steady_state[i,i], ls='--', color=colours[i])
# for i in range(system_hamiltonian.shape[0]):
#     plt.plot(time, [dm[i,i] for dm in exciton_dm_history[:-1]], label=i, linewidth=2, color=colours[i])
# plt.legend().draggable()
# plt.show()