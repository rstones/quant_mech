'''
Created on 19 Apr 2016

Fit Pauli rate equation to population dynamics of two-level system

@author: rstones
'''
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_evolution as te
import numdifftools as ndt

time_step = 0.001
duration = 1. # picoseconds
# 
# hs = HierarchySolver()
# dm_history, time = hs.hierarchy_time_evolution(17, time_step, duration)
# 
# pops1 = np.array([dm[0,0] for dm in dm_history], dtype='float64')
# pops2 = np.array([dm[1,1] for dm in dm_history], dtype='float64')
# 
# np.savez('../../data/HEOM_dimer_population_dynamics.npz', pops1=pops1, pops2=pops2, time=time)

data = np.load('../../data/HEOM_dimer_population_dynamics.npz')
pops1 = data['pops1']
pops2 = data['pops2']
time = data['time']

y = np.array(utils.differentiate_function(pops1, time), dtype='float64')

def residuals(p, y, pops1, pops2):
    k12, k21 = p
    return y - (-k12*pops1 + k21*pops2)

rates0 = np.array([0.1 ,0.1])
result = leastsq(residuals, rates0, args=(y, pops1, pops2))
rates = result[0]

time_propagator = np.zeros((2,2))
time_propagator[0,0] = -rates[0]
time_propagator[0,1] = rates[1]
time_propagator[1,1] = -rates[1]
time_propagator[1,0] = rates[0]
print time_propagator

pauli_evolution = te.liouvillian_time_evolution(np.array([1,0]), time_propagator, duration, time_step)

plt.plot(time, pops1)
plt.plot(time, pops2)
plt.plot(time, [dv[0] for dv in pauli_evolution], ls='--')
plt.plot(time, [dv[1] for dv in pauli_evolution], ls='--')
plt.plot(time, y, color='k')
plt.show()


