import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

reorg_energy_values = [35., 70., 100.]

for i,reorg_energy in enumerate(reorg_energy_values):
    data = np.load('../../data/PSIIRC_HEOM_dynamics_reorg_energy_'+str(int(reorg_energy))+'_wavenums.npz')
    exciton_dm_history = data['exciton_dm_history']
    time = data['time']
    plt.subplot(1,3,i+1)
    for j in range(6):
        plt.plot(time[:-1], [dm[j,j] for dm in exciton_dm_history], linewidth=2, label=j)
    plt.text(0.2, 0.92, r'$\lambda = '+str(int(reorg_energy))+'cm^{-1}$')
    plt.xlim(0,3.5)
    plt.xlabel('time (ps)')

plt.subplot(131)
plt.ylabel('population')

plt.legend().draggable()
plt.show()

cutoff_freq_values = [40., 70., 100.]

for i,cutoff_freq in enumerate(cutoff_freq_values):
    data = np.load('../../data/PSIIRC_HEOM_dynamics_cutoff_freq_'+str(int(cutoff_freq))+'_wavenums.npz')
    exciton_dm_history = data['exciton_dm_history']
    time = data['time']
    plt.subplot(1,3,i+1)
    for j in range(6):
        plt.plot(time[:-1], [dm[j,j] for dm in exciton_dm_history], linewidth=2, label=j)
    plt.text(0.2, 0.92, r'$\Omega_c = '+str(int(cutoff_freq))+'cm^{-1}$')
    plt.xlim(0,3.5)
    plt.xlabel('time (ps)')     
    
plt.subplot(131)
plt.ylabel('population')
 
plt.legend().draggable()        
plt.show()