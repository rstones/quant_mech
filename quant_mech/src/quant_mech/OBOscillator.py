'''
Created on 17 Jan 2017

@author: richard
'''
import numpy as np

class OBOscillator(object):
    
    def __init__(self, reorg_energy, cutoff_freq, beta, K=0):
        '''Constructor
        
        K is number of Matsubara frequencies'''
        self.reorg_energy = reorg_energy
        self.cutoff_freq =cutoff_freq
        self.beta = beta
        
        self.coeffs = np.array([self.expansion_coeffs(k) for k in range(K+1)])
        
    def matsubara_freq(self, k):
        return 2. * np.pi * k / self.beta
        
    def expansion_coeffs(self, k):
#         if not (k>=0 and isinstance(k, (int, long))):
#             raise ValueError("k should be non-negative integer")
        if k == 0: # leading coefficient
            return self.cutoff_freq*self.reorg_energy * (1./np.tan(self.beta*self.cutoff_freq/2.) - 1.j) 
        else: # Matsubara coefficients
            mf = self.matsubara_freq(k)
            return (4.*self.reorg_energy*self.cutoff_freq / self.beta) * (mf / (mf**2 - self.cutoff_freq**2))
        
    def temp_correction_sum(self):
        return (self.beta*self.cutoff_freq)**-1 * (2.*self.reorg_energy - \
                                self.beta*self.cutoff_freq*self.reorg_energy*(1./np.tan(self.beta*self.cutoff_freq/2.)))
        
    def temp_correction_sum_kth_term(self, k):
        #return (4.*self.reorg_energy*self.cutoff_freq) / (self.beta*(self.matsubara_freq(k)**2 - self.cutoff_freq**2))
        return self.coeffs[k] / self.matsubara_freq(k)
    
