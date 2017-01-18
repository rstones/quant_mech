'''
Created on 17 Jan 2017

@author: richard
'''
import numpy as np

class UBOscillator(object):
    
    def __init__(self, freq, hr_factor, damping, beta, K=0):
        self.freq = freq
        self.hr_factor = hr_factor
        self.damping = damping
        self.beta = beta
        
        self.reorg_energy = self.freq * self.hr_factor
        self.zeta = np.sqrt(self.freq**2 - (self.damping**2/4.))
        
        self.coeffs = np.append([self.expansion_coeffs(0, pm=1), self.expansion_coeffs(0, pm=-1)], self.expansion_coeffs(range(1,K+1)))
    
    def matsubara_freq(self, k):
        return 2. * np.pi * k / self.beta
    
    def nu_plus_minus(self, pm=1):
        return self.self.damping/2. + pm*1.j*self.zeta
    
    def expansion_coeffs(self, k, pm=1):
        '''k is an integer 
        pm must be either +1 or -1'''
        if not k>=0 and isinstance(k, (int, long)):
            raise ValueError("k should be non-negative integer")
        elif pm not in [1,-1]:
            raise ValueError("plus_or_minus should be either +1 or -1")
        if k == 0: # positive leading coefficient
            return pm*1.j*(self.reorg_energy*self.freq**2/(2.*self.zeta)) * (1./np.tan(self.nu_plus_minus(pm)*self.beta/2.) - 1.j)
        else: # Matsubara coefficients
            mf = self.matsubara_freq(k)
            return - (4. * self.reorg_energy * self.damping * self.freq**2 / self.beta) \
                        * (mf / ((self.freq**2 + mf**2)**2 - self.damping**2 * mf**2))
                        
    def temp_correction_sum(self):
        return (self.reorg_energy/(2.*self.zeta)) * ((np.sin(self.beta*self.damping/2.) + self.damping*np.sinh(self.beta*self.zeta)) / \
                                               (np.cos(self.beta*self.damping/2.) - np.cosh(self.beta*self.zeta))) \
                                               + (2.*self.damping / (self.beta*self.freq**2))
                                               
    def temp_correction_sum_kth_term(self, k):
        return (4.*self.reorg_energy*self.damping*self.freq**2) / (self.beta*((self.freq**2 + self.matsubara_freq(k)**2)**2 - (self.damping * self.matsubara_freq(k))**2))