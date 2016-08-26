'''
Created on 19 Aug 2016

@author: rstones
'''
import numpy as np
import numba
import scipy.sparse as sp

@numba.jit(nopython=True)
def row_index_in_array(row, array):
    array_length = array.shape[0]
    for i in range(array_length):
        r = array[i]
        in_list = True
        for j in range(r.size):
            if row[j] != r[j]:
                in_list = False
                break
        if in_list:
            return i, True
        else:
            in_list = True
    return 0, False

@numba.jit(nopython=True)
def row_in_array(row, array):
    #return np.any(np.all(array==row, axis=1))
    array_length = array.shape[0]
    for i in range(array_length):
        r = array[i]
        in_list = True
        for j in range(r.size):
            if row[j] != r[j]:
                in_list = False
                break
        if in_list:
            return True
        else:
            in_list = True
    return False

@numba.jit(nopython=True)
def higher_lower_tier_coupling(num_dms, num_aux_dm_indices, truncation_level, n_vectors, tier_indices, dm_per_tier, unit_vectors, scaling_factors, thetax_coeffs, thetao_coeffs):
    A1s = np.zeros((num_aux_dm_indices, num_dms, num_dms), dtype=numba.types.c8)
    A2s = np.zeros((num_aux_dm_indices, num_dms, num_dms), dtype=numba.types.c8)
    for n in range(num_aux_dm_indices):
        for k in range(truncation_level):
            if k < truncation_level-1:
                tier_vectors = n_vectors[tier_indices[k]:tier_indices[k+1]]
            else:
                tier_vectors = n_vectors[tier_indices[k]:]
            current_tier_offset = np.sum(dm_per_tier[:k])
            higher_tier_offset = np.sum(dm_per_tier[:k+1])
            lower_tier_offset = np.sum(dm_per_tier[:k-1])
            for i in range(tier_vectors.shape[0]):
                n_vec = tier_vectors[i]
                # coupling to higher tiers
                if k < truncation_level-1:
                    if k+2 < tier_indices.size:
                        upper_hierarchy = n_vectors[tier_indices[k+1]:tier_indices[k+2]]
                    else:
                        upper_hierarchy = n_vectors[tier_indices[k+1]:]
                    temp_dm = n_vec + unit_vectors[n]
                    idx, in_array = row_index_in_array(temp_dm, upper_hierarchy)
                    if in_array:
                        idx = idx + higher_tier_offset
                        A1s[n, current_tier_offset+i, idx] = 1.j * np.sqrt((n_vec[n]+1) * scaling_factors[n])
                        
                # coupling to lower tiers
                if k > 0:
                    lower_hierarchy = n_vectors[tier_indices[k-1]:tier_indices[k]]
                    temp_dm = n_vec - unit_vectors[n]
                    idx, in_array = row_index_in_array(temp_dm, lower_hierarchy)
                    if in_array:
                        idx = idx + lower_tier_offset
                        A1s[n, current_tier_offset+i,idx] = np.sqrt(n_vec[n] / scaling_factors[n]) * thetax_coeffs[n]
                        A2s[n, current_tier_offset+i,idx] = np.sqrt(n_vec[n] / scaling_factors[n]) * thetao_coeffs[n]

    return A1s, A2s

@numba.jit(nopython=True)
def off_diag_hierarchy_coupling(num_dms, num_indices, truncation_level, dm_per_tier, scaling_factors):
    hierarchy = np.zeros((num_dms, num_indices)) # {0: [np.zeros(num_indices, dtype=np.int64)]}
    n_vectors = [np.zeros(num_indices)] # probably won't need this now
    
    tier_start_indices = np.zeros(truncation_level)
    for i in range(dm_per_tier.size+1):
        tier_start_indices[i] = np.sum(dm_per_tier[:i])
    
    higher_coupling_matrices = np.zeros((num_indices, num_dms, num_dms), dtype=numba.types.c8) # using sparse matrices here was slower as you can only have 2D sparse
    lower_coupling_matrices = np.zeros((num_indices, num_dms, num_dms), dtype=numba.types.c8)
    next_level = 1
    while next_level < truncation_level:
        hierarchy_next_level = hierarchy[tier_start_indices[next_level]:tier_start_indices[next_level+1]]
        previous_level = hierarchy[tier_start_indices[next_level-1]:tier_start_indices[next_level]]
        for i,vec in enumerate(previous_level): # not sure if enumerate is included in numba
            for n in range(vec.size):
                vec[n] += 1
                if not row_in_array(vec, hierarchy_next_level): # add to next level
                    new_vec = np.copy(vec)
                    hierarchy[next_level].append(new_vec)
                    n_vectors.append(new_vec)
                    current_tier_vec_idx = len(hierarchy[next_level])-1
                else: # get index in current level
                    current_tier_vec_idx = row_index_in_array(vec, np.array(hierarchy[next_level]))#vec_index_in_list(vec, hierarchy[next_level])
                lower_coupling_matrices[n][tier_start_indices[next_level]+current_tier_vec_idx, tier_start_indices[next_level-1]+i] = np.sqrt(vec[n] / scaling_factors[n])
                vec[n] -= 1
                higher_coupling_matrices[n][tier_start_indices[next_level-1]+i, tier_start_indices[next_level]+current_tier_vec_idx] = np.sqrt((vec[n]+1)*scaling_factors[n])
        next_level += 1
        
    return n_vectors, higher_coupling_matrices, lower_coupling_matrices