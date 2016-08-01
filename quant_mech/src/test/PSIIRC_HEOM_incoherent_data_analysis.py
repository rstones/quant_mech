'''
Created on 14 Jun 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
import quant_mech.utils as utils
import matplotlib

font = {'size':16}
matplotlib.rc('font', **font)

np.set_printoptions(precision=3, linewidth=150, suppress=True)

data = np.load('../../data/PSIIRC_incoherent_HEOM_full_system_data.npz')

dm_history = data['dm_history']
system_hamiltonian = data['system_hamiltonian']
time = data['time']
jump_rates = data['jump_rates']
jump_operators = data['jump_operators']

print jump_rates
print jump_operators

evalues, evectors = utils.sorted_eig(system_hamiltonian[1:7,1:7])
site_exciton_transform = np.eye(10)
site_exciton_transform[1:7,1:7] = evectors.T

exciton_dm_history = np.array([np.dot(site_exciton_transform.T, np.dot(dm, site_exciton_transform)) for dm in dm_history])

labels = ['g', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'CT1', 'CT2', 'empty']

for i in range(system_hamiltonian.shape[0]):
    plt.plot(time, exciton_dm_history[:,i,i], linewidth=2, label=labels[i])
plt.legend().draggable()
plt.xlabel('time (ps)')
plt.ylabel('population')
plt.show()