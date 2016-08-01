'''
Created on 10 Jun 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

np.set_printoptions(precision=3, linewidth=150, suppress=False)

data = np.load('../../data/PSIIRC_mode_HEOM_data.npz')

exciton_dm_history = data['exciton_dm_history']
time = data['time']
system_hamiltonian = data['system_hamiltonian']
#steady_state = data['steady_state']

print time.shape
print exciton_dm_history.shape

ts_data = np.load('../../data/PSIIRC_6site_thermal_state_single_mode.npz')
thermal_state = ts_data['thermal_state']
print thermal_state
#plt.subplot(121)
thermal_state_colours = ['b', 'g', 'r', 'c', 'm', 'y']
for i in range(thermal_state.shape[0]):
    plt.axhline(thermal_state[i,i], ls='--', color=thermal_state_colours[i])

for i in range(system_hamiltonian.shape[0]):
    plt.plot(time, exciton_dm_history[:-1,i,i], linewidth=2, label=i)
plt.legend().draggable()
plt.xlabel('time (ps)')
plt.ylabel('population')
plt.xlim(0,3.5)
plt.ylim(0, 1)
 
# plt.subplot(122)
# for i in range(1,system_hamiltonian.shape[0]):
#     plt.plot(time[:-1], np.abs(exciton_dm_history[:,0,i]), linewidth=2, label=r'$|0\rangle\langle'+str(i)+'|$')
# plt.legend().draggable()
# plt.xlabel('time (ps)')
# plt.ylabel('|coherence|')
plt.show()