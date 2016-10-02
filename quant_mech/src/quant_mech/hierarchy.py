'''
Created on 21 Aug 2016

Class to generate and provide functions to access the hierarchically structured vectors indexing the
equations of motion.

@author: richard
'''
import numpy as np
import scipy.linalg as la
import numba

@numba.jitclass
class Hierarchy():
    
    def __init__(self, num_indices, truncation_level):
        self.num_indices = num_indices
        self.truncation_level = truncation_level
        
        self.pt = self.pascals_triangle()
        
        # calculate num vectors per tier and create appropriately sized array
        # and fill in tier_indices 
        self.n_vectors = np.empty()
        self.tier_indices = []
        
        self.current_tier = 0
        
    def pascals_triangle(self):
        return la.pascal(self.num_indices if self.num_indices > self.truncation_level else self.truncation_level)
    
    def dm_per_tier(self):
        return self.pt[self.num_indices-1][:self.truncation_level]
    
        '''
        Construct Pascal's triangle in matrix form
        Then take nth row corresponding to system dimension
        Then sum over first N elements corresponding to truncation level
        '''
    def number_density_matrices(self):
        return np.sum(self.pt[self.num_indices-1][:self.truncation_level])
    
    def generate_hierarchy(self):
        n_vectors = np.zeros((self.number_density_matrices(), self.num_indices))
        tier_indices = np.zeros(self.truncation_level)
        
        next_level = 1
        while next_level < self.truncation_level:
            tier_indices[next_level] = np.sum(self.pt[:next_level])
            # loop through previous tier previous tier
            current_vector_in_tier = 0
            
            # loop through each index
            
            # add 1 to each index, if already in tier move on otherwise add to tier
        
        
        
    
    def generate_hierarchy_old(self):
        n_hierarchy = {0:[np.zeros(self.num_aux_dm_indices)]}
        n_vectors = [np.zeros(self.num_aux_dm_indices)]
        tier_indices = [0]
        next_level = 1
        tier_idx = 1
        while next_level < self.truncation_level:
            n_hierarchy[next_level] = []
            tier_indices.append(tier_idx)
            for j in n_hierarchy[next_level-1]:
                for k in range(j.size):
                    j[k] += 1.
                    if not self.array_in_list(j, n_hierarchy[next_level]):
                        n_vec = np.copy(j)
                        n_hierarchy[next_level].append(n_vec)
                        n_vectors.append(n_vec)
                        tier_idx += 1
                    j[k] -= 1.
            next_level += 1
            
        return np.array(n_vectors), n_hierarchy, np.array(tier_indices)
    
    def index_in_tier(self):
        pass
    
    def get_next_tier(self):
        pass
    
    def get_previous_tier(self):
        pass
    
    def get_tier(self, tier_num):
        return self.n_vectors[self.tier_indices[tier_num]:self.tier_indices[tier_num+1]]
        
        
        