'''
Created on 7 Feb 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_evolution as te
import quant_mech.time_utils as tutils
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator
from matplotlib.ticker import AutoMinorLocator

np.set_printoptions(precision=6, linewidth=200, suppress=False)

print '[Executing script...]'

time_step = 0.0005
#duration = 1. # picoseconds
duration = 0.2 # inverse wavenums

'''Ed's params'''
electronic_coupling = 92.
system_hamiltonian = np.array([[1042., electronic_coupling], [electronic_coupling, 0]])
reorg_energy = 110.
cutoff_freq = 100. #53.08
temperature = 300. # Kelvin
beta = 1. / (utils.KELVIN_TO_WAVENUMS * temperature)

fig,ax1 = plt.subplots()

mode_params = []
    
K = 0
environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
               (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]
hs = HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=True)
hs.truncation_level = 12

print 'Calculating time evolution...'
start = tutils.getTime()

init_state = np.array([[0,0],[0,1.]])
init_state = np.dot(hs.system_evectors.T, np.dot(init_state, hs.system_evectors))
hs.init_system_dm = init_state
dm_history, time = hs.calculate_time_evolution(time_step, duration)
exciton_dm_history = hs.transform_to_exciton_basis(dm_history)

end = tutils.getTime()
print 'Calculation took ' + str(tutils.duration(end, start))

# convert time in inverse wavenums to picoseconds
time /= utils.WAVENUMS_TO_INVERSE_PS

ax1.plot(time, [dm[0,0] for dm in exciton_dm_history], linewidth=1, color='k', label='no mode')


damping_values = [0.0001, 0.4, 2., 10., 50.]
colours = ['k', 'r', '#7b2fb6', '#279a08', '#f38f3f']
linestyles = ['--', '-', '-', '-', '-']
labels = [r'Qmode ($\gamma = 0$)', r'$\gamma = 0.2$', r'$\gamma = 1$', r'$\gamma = 5$', r'$\gamma = 25$']

colours.reverse()
linestyles.reverse()
labels.reverse()

for i,gamma in enumerate(reversed(damping_values)):

    mode_params = [(1111., 0.0578, gamma)]
    
    K = 0
    environment = []
    if mode_params: # assuming that there is a single identical mode on each site 
        environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K)), \
                       (OBOscillator(reorg_energy, cutoff_freq, beta, K=K), UBOscillator(mode_params[0][0], mode_params[0][1], mode_params[0][2], beta, K=K))]
    else:
        environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                       (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]
    hs = HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=True)
    hs.truncation_level = 12
    
    print 'Calculating time evolution...'
    start = tutils.getTime()
    
    init_state = np.array([[0,0],[0,1.]])
    init_state = np.dot(hs.system_evectors.T, np.dot(init_state, hs.system_evectors))
    hs.init_system_dm = init_state
    dm_history, time = hs.calculate_time_evolution(time_step, duration)
    exciton_dm_history = hs.transform_to_exciton_basis(dm_history)
    
    end = tutils.getTime()
    print 'Calculation took ' + str(tutils.duration(end, start))
    
    # convert time in inverse wavenums to picoseconds
    time /= utils.WAVENUMS_TO_INVERSE_PS
    
    ax1.plot(time, [dm[0,0] for dm in exciton_dm_history], linewidth=3, ls=linestyles[i], color=colours[i], label=labels[i])
    
ax1.set_ylim(0, 0.77)
ax1.set_xlim(0,1)
ax1.yaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('time (ps)')
plt.ylabel(r'population $\rho_{YY}$')
plt.legend().draggable()

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