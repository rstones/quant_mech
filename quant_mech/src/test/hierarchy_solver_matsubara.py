'''
Created on 10 Jan 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_evolution as te
import quant_mech.time_utils as tutils

np.set_printoptions(precision=6, linewidth=200, suppress=False)

print '[Executing script...]'

time_step = 0.001
#duration = 1. # picoseconds
duration = 0.2 # inverse wavenums

electronic_coupling = 100.
system_hamiltonian = np.array([[100., electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 100.
cutoff_freq = 53.08
temperature = 300. # Kelvin
beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)
mode_params = [] #[(200., 0.25, 10.)]
init_state = np.array([[1.,0],[0,0]])

K_values = range(4)

dm_histories = np.zeros((len(K_values)+1, duration/time_step+1, 2, 2), dtype='complex128')

for i in K_values:
    hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, beta, underdamped_mode_params=mode_params, \
                        num_matsubara_freqs=i, temperature_correction=True)
    
    print 'Calculating for K = ' + str(i)
    start = tutils.getTime()
    
    hs.init_system_dm = init_state
    hs.truncation_level = 9
    dm_history, time = hs.calculate_time_evolution(time_step, duration)
    dm_histories[i] = dm_history
    
    end = tutils.getTime()
    print 'Calculation took ' + str(tutils.duration(end, start))
    
'''Now calculate for K = 0 but include temperature correction'''
hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, beta, underdamped_mode_params=mode_params, \
                        num_matsubara_freqs=0, temperature_correction=True)
    
print 'Calculating for K = 0, with temperature correction'
start = tutils.getTime()

hs.init_system_dm = init_state
hs.truncation_level = 9
dm_history, time = hs.calculate_time_evolution(time_step, duration)
dm_histories[-1] = dm_history

end = tutils.getTime()
print 'Calculation took ' + str(tutils.duration(end, start))


'''Plotting'''
# convert time in inverse wavenums to picoseconds
time /= utils.WAVENUMS_TO_INVERSE_PS

labels = ['K=0', 'K=1', 'K=2', 'K=3', 'temp. correction']
    
for i,dm_history in enumerate(dm_histories):
    plt.subplot(121)
    plt.plot(time, [dm[0,0] for dm in dm_history], linewidth=2, label=labels[i])
    #plt.plot(time, [dm[1,1] for dm in dm_history], linewidth=2)
    plt.ylim(0.3, 1)
    plt.subplot(122)
    plt.plot(time, np.abs([dm[0,1] for dm in dm_history]), linewidth=2, label=labels[i])

plt.subplot(121)
plt.legend().draggable()
plt.show()