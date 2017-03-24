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
reorg_energy = 5. / beta
cutoff_freq = 1. / beta
K = 0
environment = [(OBOscillator(reorg_energy, cutoff_freq, beta, K=K),),
                   (OBOscillator(reorg_energy, cutoff_freq, beta, K=K),)]

solver = hs.HierarchySolver(system_hamiltonian, environment, beta, num_matsubara_freqs=K, temperature_correction=False)
solver.truncation_level = 12

num_dms = solver.number_density_matrices()
num_indices = solver.num_aux_dm_indices
truncation_level = solver.truncation_level
dm_per_tier = solver.dm_per_tier()
coeffs = np.array([environment[0][0].coeffs[0], environment[1][0].coeffs[0]])

n_vectors, graph_elements, graph_row_indices, graph_col_indices \
        = hsft.generate_hierarchy_graph(num_dms, num_indices, truncation_level, dm_per_tier, coeffs)
    
import scipy.sparse as sp
graph = sp.coo_matrix((num_dms,num_dms))
for n in range(num_indices):
    graph += sp.coo_matrix((graph_elements[n],(graph_row_indices[n], graph_col_indices[n])), shape=(num_dms,num_dms))
    
np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=1e6)
print coeffs
#print graph.todense()
print graph_elements
print graph_row_indices
print graph_col_indices
