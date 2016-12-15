'''

@author Richard Stones
'''
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_evolution as te
import quant_mech.time_utils as tutils

np.set_printoptions(precision=6,linewidth=200, suppress=False)

print '[Executing script...]'

time_step = 0.005
duration = 1. # picoseconds

'''Testing new branch commiting'''

electronic_coupling = 100.
system_hamiltonian = np.array([[100., electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 100.
cutoff_freq = 53.08
temperature = 300.
mode_params = [] #[(200., 0.25, 10.)]
hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, underdamped_mode_params=mode_params, num_matsubara_freqs=0, temperature_correction=False)

print 'Calculating time evolution...'
start = tutils.getTime()

init_state = np.array([[1.,0],[0,0]])
# init_state = np.dot(hs.system_evectors, np.dot(init_state, hs.system_evectors.T))
# print init_state

hs.init_system_dm = init_state
hs.truncation_level = 11
dm_history, time = hs.calculate_time_evolution(time_step, duration)
#exciton_dm_history = hs.transform_to_exciton_basis(dm_history)
       
end = tutils.getTime()
print 'Calculation took ' + str(tutils.duration(end, start))

plt.subplot(121)
plt.plot(time, [dm[0,0] for dm in dm_history], linewidth=2)
#plt.plot(time, [dm[1,1] for dm in dm_history], linewidth=2)
plt.ylim(0.3, 1)
plt.subplot(122)
plt.plot(time, np.abs([dm[0,1] for dm in dm_history]), linewidth=2)
plt.show()

# print 'Calculating steady state...'
# start = tutils.getTime()
#    
# hs.init_system_dm = np.array([[1.,0],[0,0]])
# hs.truncation_level = 8
# 
# steady_state = hs.normalised_steady_state()
# print steady_state
#     
# end = tutils.getTime()
# print 'Calculation took ' + str(tutils.duration(end, start))

#np.savez('../../data/?.npz', a=a, b=b)

print '[Script execution complete]'