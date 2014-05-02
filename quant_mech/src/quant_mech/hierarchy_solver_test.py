'''

@author Richard Stones
'''
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import quant_mech.utils as utils
from quant_mech.hierarchy_solver import HierarchySolver
import quant_mech.time_evolution as te

print '[Executing script...]'

time_step = 0.001
duration = 1. # picoseconds

hs = HierarchySolver()
dm_history, time = hs.hierarchy_time_evolution(17, time_step, duration)
plt.plot(time, [dm[0,0] for dm in dm_history])

dm_history2 = te.von_neumann_eqn(hs.init_system_dm, hs.system_hamiltonian, duration, time_step)
plt.plot(time, [dm[0,0] for dm in dm_history2])

plt.show()

#np.savez('../../data/?.npz', a=a, b=b)

print '[Script execution complete]'