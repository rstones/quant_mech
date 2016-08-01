'''
Created on 11 Jun 2016

@author: rstones
'''
import numpy as np
import matplotlib.pyplot as plt
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_utils as tutils

np.set_printoptions(precision=3, linewidth=100)

delta_E = 100.
coupling = 100.
system_hamiltonian = np.array([[0, 0, 0, 0],
                               [0, 1000.+delta_E/2., coupling, 0],
                               [0, coupling, 1000.-delta_E/2., 0],
                               [0, 0, 0, 500.]])
reorg_energy = 100.
cutoff_freq = 53.
temperature = 300.

system_dimension = system_hamiltonian.shape[0]
jump_operators = np.zeros((7, system_dimension, system_dimension))
# excitation
jump_operators[0] = np.array([[0, 1., 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
# deexcitation
jump_operators[1] = np.array([[0, 0, 0, 0],
                              [1., 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
# CT1 backward
jump_operators[2] = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1.],
                              [0, 0, 0, 0]])
# CT1 forward
jump_operators[3] = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 1., 0]])
# CT2 backward
jump_operators[4] = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 1.],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
# CT2 forward
jump_operators[5] = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 1., 0, 0]])
# irreversible transfer to ground state
jump_operators[6] = np.array([[0, 0, 0, 1.],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
jump_rates = np.array([1., 1.1, 2., 2.1, 0.8, 0.82, 0.2])

hs = HierarchySolver(system_hamiltonian, reorg_energy, cutoff_freq, temperature, jump_operators=jump_operators, jump_rates=jump_rates)

# start_time = tutils.getTime()
# print 'Calculating steady state...'
# steady_state = hs.calculate_steady_state(6,6)
# print steady_state
#  
# end_time = tutils.getTime()
# print 'Calculation took ' + str(tutils.duration(end_time, start_time))

start_time = tutils.getTime()
print 'Calculating time evolution...'
dm_history,time = hs.hierarchy_time_evolution(np.array([[0, 0, 0, 0],[0, 1., 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]), 8, 8, 0.005, 15., sparse=True)
end_time = tutils.getTime()
print 'Calculation took ' + str(tutils.duration(end_time, start_time))
print time.shape
print dm_history.shape
for i in range(4):
    plt.plot(time[:-1], dm_history[:,i,i], label=i)
plt.legend()
plt.show()

