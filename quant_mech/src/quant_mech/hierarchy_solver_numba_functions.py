'''
Created on 19 Aug 2016

@author: rstones
'''
import numpy as np
import numba

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
            return i
        else:
            in_list = True

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
                    if row_in_array(temp_dm, upper_hierarchy): # numbaize this function
                        idx = row_index_in_array(temp_dm, upper_hierarchy) + higher_tier_offset
                        A1s[n, current_tier_offset+i, idx] = 1.j * np.sqrt((n_vec[n]+1) * scaling_factors[n])
                        
                # coupling to lower tiers
                if k > 0:
                    lower_hierarchy = n_vectors[tier_indices[k-1]:tier_indices[k]]
                    temp_dm = n_vec - unit_vectors[n]
                    if row_in_array(temp_dm, lower_hierarchy):
                        idx = row_index_in_array(temp_dm, lower_hierarchy) + lower_tier_offset
                        A1s[n, current_tier_offset+i,idx] = np.sqrt(n_vec[n] / scaling_factors[n]) * thetax_coeffs[n]
                        A2s[n, current_tier_offset+i,idx] = np.sqrt(n_vec[n] / scaling_factors[n]) * thetao_coeffs[n]

    return A1s, A2s