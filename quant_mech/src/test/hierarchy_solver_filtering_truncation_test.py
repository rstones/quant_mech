'''
Created on 22 Mar 2017

@author: richard
'''
import numpy as np
import quant_mech.hierarchy_solver as hs
import quant_mech.hierarchy_solver_filtering_truncation as hsft
from quant_mech.OBOscillator import OBOscillator

time_step = 0.01
#duration = 1. # picoseconds
duration = 15.7 # inverse wavenums

'''Shi J. Chem. Phys. 130, 2009'''
beta = 1.
electronic_coupling = 0.1 / beta
system_hamiltonian = np.array([[0, electronic_coupling], [electronic_coupling, 0]])
# system_hamiltonian = np.array([[0, electronic_coupling, electronic_coupling],
#                                [electronic_coupling, 0, electronic_coupling],
#                                [electronic_coupling, electronic_coupling, 0]])
reorg_energy = 5. / beta
cutoff_freq = 1. / beta
K = 0
environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]

solver = hs.HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=False)
solver.truncation_level = 3

num_dms = solver.number_density_matrices()
num_indices = solver.num_aux_dm_indices
truncation_level = solver.truncation_level
dm_per_tier = solver.dm_per_tier()
coeffs = np.array([environment[0][0].coeffs[0], environment[1][0].coeffs[0]])
#coeffs = np.array([environment[0][0].coeffs[0], environment[1][0].coeffs[0], environment[2][0].coeffs[0]])
#coeffs = np.array([environment[0][0].coeffs[0], environment[0][0].coeffs[1], environment[1][0].coeffs[0], environment[1][0].coeffs[1]])

n_vectors, graph_elements, graph_row_indices, graph_col_indices, ados_to_delete = hsft.generate_hierarchy_graph(num_dms, num_indices, truncation_level, dm_per_tier, coeffs)

print dm_per_tier

import scipy.sparse as sp
graph = sp.coo_matrix((num_dms,num_dms))
for n in range(num_indices):
    graph += sp.coo_matrix((graph_elements[n],(graph_row_indices[n], graph_col_indices[n])), shape=(num_dms,num_dms))
    
np.set_printoptions(precision=3, linewidth=1000, suppress=True, threshold=1e6)
#print graph
print graph.todense()
#print graph_elements[0]
for i in range(truncation_level):
    print graph_elements[0,np.sum(dm_per_tier[:i]):np.sum(dm_per_tier[:i+1])]
print graph_row_indices[0]
print graph_col_indices[0]
# 
# ados_to_delete = []
# for n in range(num_indices):
#     for i in range(truncation_level-1):
#         tier = graph_elements[n,np.sum(dm_per_tier[:i]):np.sum(dm_per_tier[:i+1])]
#         max_coupling = np.amax(tier)
#         for j,el in enumerate(tier):
#             if el < max_coupling/10:
#                 ados_to_delete.append((graph_row_indices[n,np.sum(dm_per_tier[:i])+j],graph_col_indices[n,np.sum(dm_per_tier[:i])+j]))
# print ados_to_delete
# print len(ados_to_delete)
# print graph_elements.size

