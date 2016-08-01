import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'size':18}
matplotlib.rc('font', **font)

labels = ['g', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'CT1', 'CT2', 'empty']
system_dim = 10

# reorg_energy_values = [35., 70., 100.]
# 
# for i,reorg_energy in enumerate(reorg_energy_values):
#     data = np.load('../../data/PSIIRC_HEOM_incoherent_rates_reorg_energy_'+str(int(reorg_energy))+'_wavenums.npz')
#     exciton_dm_history = data['exciton_dm_history']
#     time = data['time']
#     plt.subplot(1,3,i+1)
#     for j in range(system_dim):
#         plt.plot(time[:-1], [dm[j,j] for dm in exciton_dm_history], linewidth=2, label=labels[j])
# plt.legend().draggable()
# plt.show()
# 
# cutoff_freq_values = [40., 70., 100.]
# 
# for i,cutoff_freq in enumerate(cutoff_freq_values):
#     data = np.load('../../data/PSIIRC_HEOM_incoherent_rates_cutoff_freq_'+str(int(cutoff_freq))+'_wavenums.npz')
#     exciton_dm_history = data['exciton_dm_history']
#     time = data['time']
#     plt.subplot(1,3,i+1)
#     for j in range(system_dim):
#         plt.plot(time[:-1], [dm[j,j] for dm in exciton_dm_history], linewidth=2, label=labels[j])
# plt.legend().draggable()
# plt.show()

data = np.load('../../data/PSIIRC_HEOM_incoherent_rates_slow_jump_rates_data.npz')
exciton_dm_history = data['exciton_dm_history']
time = data['time']
for j in range(system_dim):
    plt.plot(time[:-1], [dm[j,j] for dm in exciton_dm_history], linewidth=2, label=labels[j])
plt.legend().draggable()
plt.show()
