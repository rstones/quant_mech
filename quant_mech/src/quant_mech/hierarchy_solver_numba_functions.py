'''
Created on 19 Aug 2016

@author: rstones
'''
import numpy as np
import numba

@numba.jit(nopython=True)
def generate_hierarchy_and_tier_couplings(num_dms, num_indices, truncation_level, dm_per_tier, scaling_factors):
    hierarchy = np.zeros((num_dms, num_indices))
    
    tier_start_indices = np.zeros(truncation_level+1, dtype=np.int32)
    for i in range(dm_per_tier.size+1):
        tier_start_indices[i] = np.sum(dm_per_tier[:i])
    
    higher_coupling_matrices = np.zeros((num_indices, num_dms, num_dms), dtype=np.complex64)
    lower_coupling_matrices = np.zeros((num_indices, num_dms, num_dms), dtype=np.complex64)
    next_level = 1
    while next_level < truncation_level:
        hierarchy_next_level = hierarchy[tier_start_indices[next_level]:tier_start_indices[next_level+1]]
        previous_level = hierarchy[tier_start_indices[next_level-1]:tier_start_indices[next_level]]
        for i in range(previous_level.shape[0]):
            vec = previous_level[i]
            for n in range(vec.size):
                vec[n] += 1
                for j in range(hierarchy_next_level.shape[0]):
                    r = hierarchy_next_level[j]
                    in_array = True
                    zero_vec = True
                    for k in range(r.size):
                        if r[k] != 0:
                            zero_vec = False
                        if r[k] != vec[k]:
                            in_array = False
                    if zero_vec:
                        hierarchy_next_level[j] = vec
                        current_tier_vec_idx = j
                        break
                    elif in_array:
                        current_tier_vec_idx = j
                        break
                lower_coupling_matrices[n][tier_start_indices[next_level]+current_tier_vec_idx, tier_start_indices[next_level-1]+i] = np.sqrt(vec[n] / scaling_factors[n])
                vec[n] -= 1
                higher_coupling_matrices[n][tier_start_indices[next_level-1]+i, tier_start_indices[next_level]+current_tier_vec_idx] = np.sqrt((vec[n]+1)*scaling_factors[n])
        next_level += 1
        
    return hierarchy, higher_coupling_matrices, lower_coupling_matrices