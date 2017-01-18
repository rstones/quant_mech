'''
Created on 21 Mar 2014

@author: rstones
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import ode
import quant_mech.utils as utils
from quant_mech.hierarchy_solver_numba_functions import generate_hierarchy_and_tier_couplings
from quant_mech import UBOscillator, OBOscillator

class HierarchySolver(object):
    '''
    classdocs
    '''
    
    def __init__(self, hamiltonian, environment, beta, jump_operators=None, jump_rates=None, \
                 num_matsubara_freqs=0, temperature_correction=False):
        '''
        Constructor
        '''
        self.hamiltonian = hamiltonian
        self.environment = environment
        self.beta = beta
        self.jump_operators = jump_operators
        self.jump_rates = jump_rates
        self.num_matsubara_freqs = num_matsubara_freqs
        self.temperature_correction = temperature_correction
        
        self.system_evalues, self.system_evectors = utils.sorted_eig(self.system_hamiltonian)
        self.system_evectors = self.system_evectors.T
        self.system_dimension = self.system_hamiltonian.shape[0]
        
        self.heom_sys_dim = 0
        if self.num_matsubara_freqs>0 or self.temperature_correction:
            self.matsubara_freqs = self.calculate_matsubara_freqs()
        self.Vx_operators, self.Vo_operators = [],[]#self.construct_commutator_operators()
        
        # if tuple or UBOscillator or OBOscillator put into list of tuples
        if type(self.environment) is tuple: # multiple oscillators, identical on each site
            self.environment = [self.environment] * self.system_dimension
        elif isinstance(environment, (OBOscillator, UBOscillator)): # single oscillator identical on each site
            self.environment = [(self.environment)] * self.system_dimension
        
        if type(self.environment) is list:# environment defined per site
            self.diag_coeffs = []
            self.phix_coeffs = []
            self.thetax_coeffs = []
            self.thetao_coeffs = []
            self.tc_terms = []
            for i,site in enumerate(self.environment):
                if site:
                    self.heom_sys_dim += 1
                    site_coupling_operator = np.zeros(self.system_dimension)
                    site_coupling_operator[i,i] = 1.
                    site_Vx_operator = self.commutator_to_superoperator(site_coupling_operator, type='-')
                    site_Vo_operator = self.commutator_to_superoperator(site_coupling_operator, type='+')
                    self.temp_correction_Vx_ops.append(site_Vx_operator)
                    tc_term = 0
                    for osc in site:
                        if isinstance(osc, OBOscillator):
                            self.diag_coeffs.append(osc.cutoff_freq)
                            self.phix_coeffs.append(self.phix_coeff_OBO(osc))
                            site_coupling_operator = np.zeros(self.system_dimension)
                            site_coupling_operator[i,i] = 1.
                            self.Vx_operators.append(self.commutator_to_superoperator(site_coupling_operator, type='-'))
                            self.thetax_coeffs.append(self.thetax_coeff_OBO(osc))
                            self.thetao_coeffs.append(self.thetao_coeff_OBO(osc))
                            self.Vx_operators.append(site_Vx_operator)
                            self.Vo_operators.append(site_Vo_operator)
                        elif isinstance(osc, UBOscillator):
                            self.diag_coeffs.append(0.5*osc.damping)
                            self.diag_coeffs.append(-1.j*osc.zeta)
                            self.phix_coeffs.append(self.phix_coeff_UBO(osc, pm=1))
                            self.phix_coeffs.append(self.phix_coeff_UBO(osc, pm=-1))
                            self.thetax_coeffs.append(self.thetax_coeff_UBO(osc, pm=1))
                            self.thetax_coeffs.append(self.thetax_coeff_UBO(osc, pm=-1))
                            self.thetao_coeffs.append(self.thetao_coeff_UBO(osc, pm=1))
                            self.thetao_coeffs.append(self.thetao_coeff_UBO(osc, pm=-1))
                            self.Vx_operators.append(site_Vx_operator)
                            self.Vx_operators.append(site_Vx_operator)
                            self.Vo_operators.append(site_Vo_operator)
                            self.Vo_operators.append(site_Vo_operator)
                        tc_term += osc.temp_correction_sum() - np.sum(osc.temp_correction_sum_kth_term(range(1,self.num_matsubara_freqs+1)))
                    self.tc_terms.append(tc_term * np.dot(site_coupling_operator, site_coupling_operator))
                    for k in range(1,self.num_matsubara_freqs+1):
                        self.diag_coeffs.append(self.matsubara_freqs(k))
                        self.phix_coeffs.append(self.phix_coeff_MF(site, k))
                        self.thetax_coeffs.append(self.thetax_coeff_MF(site, k+1))
                        self.Vx_operators.append(site_Vx_operator)
        else:
            raise ValueError("Environment defined in invalid format!")

        self.num_aux_dm_indices = len(self.diag_coeffs)
        self.diag_coeffs = np.array(self.diag_coeffs)
        self.phix_coeffs = np.array(self.phix_coeffs)
        self.thetax_coeffs = np.array(self.thetax_coeffs)
        self.thetao_coeffs = np.array(self.thetao_coeffs)
        self.Vx_operators = np.array(self.Vx_operators)
        self.Vo_operators = np.array(self.Vo_operators)
        self.temp_correction_Vx_ops = np.array(self.temp_correction_Vx_ops)

    @classmethod
    def old_constructor(self, hamiltonian, drude_reorg_energy, drude_cutoff, beta, \
                 jump_operators=None, jump_rates=None, underdamped_mode_params=[], \
                 num_matsubara_freqs=0, temperature_correction=False, \
                 sites_to_couple=[]):
        pass
        
#         self.init_system_dm = None
#         self.drude_reorg_energy = drude_reorg_energy
#         self.drude_cutoff = drude_cutoff
#         self.beta = beta
#         self.system_hamiltonian = hamiltonian
#         self.system_evalues, self.system_evectors = utils.sorted_eig(self.system_hamiltonian)
#         self.system_evectors = self.system_evectors.T
#         
#         self.jump_operators = jump_operators
#         self.jump_rates = jump_rates
#         
#         self.system_dimension = self.system_hamiltonian.shape[0]
#         if np.any(sites_to_couple):
#             if sites_to_couple.ndim != 1 or len(sites_to_couple) != self.system_hamiltonian.shape[0] or np.all(sites_to_couple == 0):
#                 raise ValueError("sites_to_couple should be a 1d array containing 1s and 0s indicating the index of sites" + \
#                                  " to couple to a bath described by hierarchical equations of motion.")
#             self.sites_to_couple = sites_to_couple
#             self.heom_sys_dim = np.count_nonzero(self.sites_to_couple)
#         else:
#             self.heom_sys_dim = self.system_dimension
#             self.sites_to_couple = np.ones(self.system_dimension)
#         
#         # commutator operators for each site coupling to a bath
#         self.Vx_operators, self.Vo_operators = self.construct_commutator_operators()
#         
#         # zeroth order freqs for identical Drude spectral density on each site are just cutoff freq
#         self.drude_zeroth_order_freqs = np.zeros(self.heom_sys_dim, dtype='complex64')
#         self.drude_zeroth_order_freqs.fill(self.drude_cutoff)
#         
#         # underdamped_mode_params should be a list of tuples
#         # each tuple having frequency, Huang-Rhys factor and damping for each mode in that order
#         self.underdamped_mode_params = underdamped_mode_params
#         self.num_modes = len(self.underdamped_mode_params)
#         if np.any(self.underdamped_mode_params):
#             self.BO_zeroth_order_freqs = np.zeros((self.num_modes, self.heom_sys_dim))
#             self.BO_zetas = np.zeros((self.num_modes, self.heom_sys_dim))
#             for i,mode in enumerate(self.underdamped_mode_params):
#                 self.BO_zeroth_order_freqs[i].fill(mode[2])
#                 self.BO_zetas[i].fill(np.sqrt(mode[0]**2 - mode[2]**2/4.))
#         
#         self.num_matsubara_freqs = num_matsubara_freqs
#         self.temperature_correction = temperature_correction
#         if self.num_matsubara_freqs > 0 or self.temperature_correction:
#             self.matsubara_freqs = self.calculate_matsubara_freqs()
#         
#         '''Currently assumes there will be one Drude spectral density,
#         with optional number of modes. Need to change to allow arbitrary combination of 
#         over and underdamped Brownian oscillators.'''
#         self.num_aux_dm_indices = (1 + 2*self.num_modes + self.num_matsubara_freqs)*self.heom_sys_dim
#         
#         '''Also assumes that spectral density is same on each site.
#         Change to have different spectral densities on each site.'''
#         # calculate coefficients of operators coupling to lower tiers and fill in vectors
#         self.phix_coeffs = np.zeros(self.num_aux_dm_indices, dtype='complex64')
#         self.thetax_coeffs = np.zeros(self.num_aux_dm_indices, dtype='complex64')
#         self.thetao_coeffs = np.zeros(self.num_aux_dm_indices, dtype='complex64')
#         self.phix_coeffs[:self.heom_sys_dim].fill(self.drude_phix_coeffs(0))
#         self.thetax_coeffs[:self.heom_sys_dim].fill(self.drude_thetax_coeff(0))
#         self.thetao_coeffs[:self.heom_sys_dim].fill(self.drude_thetao_coeff(0))
#         for i in range(self.num_modes):
#             freq = self.underdamped_mode_params[i][0]
#             reorg_energy = self.underdamped_mode_params[i][0]*self.underdamped_mode_params[i][1]
#             damping = self.underdamped_mode_params[i][2]
#             self.phix_coeffs[self.heom_sys_dim*(1+2*i):self.heom_sys_dim*(2+2*i)].fill(self.mode_phix_coeff(freq, reorg_energy, damping, 0, 1.))
#             self.phix_coeffs[self.heom_sys_dim*(2+2*i):self.heom_sys_dim*(3+2*i)].fill(self.mode_phix_coeff(freq, reorg_energy, damping, 0, -1.))
#             self.thetax_coeffs[self.heom_sys_dim*(1+2*i):self.heom_sys_dim*(2+2*i)].fill(self.mode_thetax_coeff(freq, reorg_energy, damping, 0, -1.))
#             self.thetax_coeffs[self.heom_sys_dim*(2+2*i):self.heom_sys_dim*(3+2*i)].fill(self.mode_thetax_coeff(freq, reorg_energy, damping, 0, 1.))
#             self.thetao_coeffs[self.heom_sys_dim*(1+2*i):self.heom_sys_dim*(2+2*i)].fill(self.mode_thetao_coeff(freq, reorg_energy, damping, 0, -1.))
#             self.thetao_coeffs[self.heom_sys_dim*(2+2*i):self.heom_sys_dim*(3+2*i)].fill(self.mode_thetao_coeff(freq, reorg_energy, damping, 0, 1.))            
#             
#         mf_start_idx = (1 + 2*self.num_modes)*self.heom_sys_dim
#         for k in range(1, self.num_matsubara_freqs+1):
#             self.phix_coeffs[mf_start_idx+((k-1)*self.heom_sys_dim):mf_start_idx+(k*self.heom_sys_dim)].fill(self.mf_phix_coeff(self.underdamped_mode_params, k))
#             self.thetax_coeffs[mf_start_idx+((k-1)*self.heom_sys_dim):mf_start_idx+(k*self.heom_sys_dim)].fill(self.mf_thetax_coeff(self.underdamped_mode_params, k))
#             # thetao coeffs are 0 for Matsubara freqs
        
    def calculate_matsubara_freqs(self):
        return np.array([2.*np.pi*k / self.beta for k in range(1,self.num_matsubara_freqs+1)])
    
#     def drude_expansion_coeffs(self, k):
#         '''k is an integer running from 0 to self.num_matsubara_freqs'''
#         if not (k>=0 and isinstance(k, (int, long))) or k > self.num_matsubara_freqs:
#             raise ValueError("k should be non-negative integer running from 0 to self.num_matsubara_freqs")
#         if k == 0: # leading coefficient
#             return self.drude_cutoff*self.drude_reorg_energy * (1./np.tan(self.beta*self.drude_cutoff/2.) - 1.j) 
#         else: # Matsubara coefficients
#             return (4.*self.drude_reorg_energy*self.drude_cutoff / self.beta) \
#                                 * (self.matsubara_freqs[k-1] / (self.matsubara_freqs[k-1]**2 - self.drude_cutoff**2))
#     
#     def drude_phix_coeffs(self, k):
#         return 1.j * np.sqrt(np.abs(self.drude_expansion_coeffs(k)))
#         
#     def drude_thetax_coeff(self, k):
#         return 1.j * (1. / np.sqrt(np.abs(self.drude_expansion_coeffs(k)))) \
#                             * self.drude_reorg_energy*self.drude_cutoff / np.tan(self.beta*self.drude_cutoff/2.)
#     
#     def drude_thetao_coeff(self, k):
#         return (1. / np.sqrt(np.abs(self.drude_expansion_coeffs(k)))) \
#                             * self.drude_reorg_energy*self.drude_cutoff
                            
    def phix_coeff_OBO(self, osc):
        return 1.j * np.sqrt(np.abs(osc.coeffs[0]))
        
    def thetax_coeff_OBO(self, osc):
        return 1.j * (1. / np.sqrt(np.abs(osc.coeffs[0]))) \
                            * osc.reorg_energy*osc.cutoff_freq / np.tan(osc.beta*osc.cutoff_freq/2.)
    
    def thetao_coeff_OBO(self, osc):
        return (1. / np.sqrt(np.abs(osc.coeffs[0]))) \
                            * osc.reorg_energy*osc.cutoff_freq 
    
#     def drude_scaling_factor(self):
#         '''Absolute value of the leading coefficient from exponential expansion of the Drude spectral density'''
#         # not sure this + 1.j is correct?
#         return np.abs(self.drude_cutoff*self.drude_reorg_energy * (1./np.tan(self.beta*self.drude_cutoff/2.) + 1.j))
    
#     def mode_expansion_coeffs(self, freq, reorg_energy, damping, k, plus_or_minus=1.):
#         '''k is an integer running from 0 to self.num_matsubara_freqs
#         plus_or_minus must be either +1 or -1'''
#         if not (k>=0 and isinstance(k, (int, long))) or k > self.num_matsubara_freqs:
#             raise ValueError("k should be non-negative integer running from 0 to self.num_matsubara_freqs")
#         elif plus_or_minus not in [1,-1]:
#             raise ValueError("plus_or_minus should be either +1 or -1")
#         if k == 0: # positive leading coefficient
#             zeta = np.sqrt(freq**2 - (damping**2/4.))
#             nu = damping/2. + plus_or_minus*1.j*zeta
#             return plus_or_minus*1.j*(reorg_energy*freq**2/(2.*zeta)) * (1./np.tan(nu*self.beta/2.) - 1.j)
#         else: # Matsubara coefficients
#             return - (4. * reorg_energy * damping * freq**2 / self.beta) \
#                         * (self.matsubara_freqs[k] / ((freq**2 + self.matsubara_freqs[k]**2)**2 - damping**2 * self.matsubara_freqs[k]**2))
#     
#     def mode_phix_coeff(self, freq, reorg_energy, damping, k, plus_or_minus):
#         return 1.j * np.sqrt(np.sqrt(np.abs( \
#                     self.mode_expansion_coeffs(freq, reorg_energy, damping, k, plus_or_minus) \
#                         * self.mode_expansion_coeffs(freq, reorg_energy, damping, k, -plus_or_minus).conj() )))
#     
#     def mode_thetax_coeff(self, freq, reorg_energy, damping, k, plus_or_minus):
#         zeta = np.sqrt(freq**2 - (damping**2/4.))
#         return (-plus_or_minus*1.j / np.sqrt(np.sqrt(np.abs( \
#                     self.mode_expansion_coeffs(freq, reorg_energy, damping, k, plus_or_minus) \
#                         * self.mode_expansion_coeffs(freq, reorg_energy, damping, k, -plus_or_minus).conj() )))) \
#                         *(reorg_energy*freq**2/(2.*zeta)) / np.tanh((1.j*self.beta/4.)*(damping + plus_or_minus*2.j*zeta))
#     
#     def mode_thetao_coeff(self, freq, reorg_energy, damping, k, plus_or_minus):
#         zeta = np.sqrt(freq**2 - (damping**2/4.))
#         return (plus_or_minus*1.j  / np.sqrt(np.sqrt(np.abs( \
#                     self.mode_expansion_coeffs(freq, reorg_energy, damping, k, plus_or_minus) \
#                         * self.mode_expansion_coeffs(freq, reorg_energy, damping, k, -plus_or_minus).conj() )))) \
#                         *(reorg_energy*freq**2 / (2.*zeta))
                        
    def phix_coeff_UBO(self, osc, pm=1):
        return 1.j * np.sqrt(np.sqrt(np.abs( osc.coeffs[0 if pm>0 else 1] * osc.expansion_coeffs(0, pm=-pm).conj() )))
        
    def thetax_coeff_UBO(self, osc, pm=1):
        return (-pm*1.j / np.sqrt(np.sqrt(np.abs( \
                    osc.coeffs[0 if pm>0 else 1] * osc.expansion_coeffs(0, pm=-pm).conj() )))) \
                        *(osc.reorg_energy*osc.freq**2/(2.*osc.zeta)) / np.tanh((1.j*self.beta/4.)*(osc.damping + pm*2.j*osc.zeta))
                        
    def thetao_coeff_UBO(self, osc, pm=1):
        return (pm*1.j  / np.sqrt(np.sqrt(np.abs( \
                    osc.coeffs[0 if pm>0 else 1] * osc.expansion_coeffs(0, pm=-pm).conj() )))) \
                        *(osc.reorg_energy*osc.freq**2 / (2.*osc.zeta))

#     def mf_phix_coeff(self, mode_params, k):
#         '''Sum of Drude and mode coeffs
#         k should run from 1 to self.num_matsubara_freqs '''
#         result = np.sqrt(np.abs(self.drude_expansion_coeffs(k)))
#         for mode in mode_params:
#             freq = mode[0]
#             reorg_energy = mode[0]*mode[1]
#             damping = mode[2]
#             result += np.sqrt(np.abs(self.mode_expansion_coeffs(freq, reorg_energy, damping, k)))
#         return 1.j * result
#     
#     def mf_thetax_coeff(self, mode_params, k):
#         '''Sum of Drude and mode coeffs
#         k should run from 1 to self.num_matsubara_freqs '''
#         c_drude = self.drude_expansion_coeffs(k)
#         result = c_drude / np.sqrt(np.abs(c_drude))
#         for mode in mode_params:
#             freq = mode[0]
#             reorg_energy = mode[0]*mode[1]
#             damping = mode[2]
#             c_mode = self.mode_expansion_coeffs(freq, reorg_energy, damping, k)
#             result += 2. * c_mode / np.sqrt(np.abs(c_mode))
#         return 1.j * result
    
    def phix_coeff_MF(self, oscs, k):
        coeff = 0
        for osc in oscs:
            if isinstance(osc, OBOscillator):
                coeff += np.sqrt(np.abs(osc.coeffs[k]))
            elif isinstance(osc, UBOscillator):
                coeff += np.sqrt(np.abs(osc.coeffs[k+1]))
        return 1.j * coeff
    
    def thetax_coeff_MF(self, oscs, k):
        coeff = 0
        for osc in oscs:
            if isinstance(osc, OBOscillator):
                coeff += osc.coeffs[k] / np.sqrt(np.abs(osc.coeffs[k]))
            elif isinstance(osc, UBOscillator):
                coeff += 2.* osc.coeffs[k+1] / np.sqrt(np.abs(osc.coeffs[k+1]))
        return 1.j * coeff

    def pascals_triangle(self):
        return la.pascal(self.num_aux_dm_indices+1 if self.num_aux_dm_indices > self.truncation_level else self.truncation_level+1)
    
    '''
    Get dm for one extra tier to make indexing of hierarchy easier when keeping them in an array and not a dict
    '''
    def dm_per_tier(self):
        return self.pascals_triangle()[self.num_aux_dm_indices-1][:self.truncation_level+1]
    
    '''
    Construct Pascal's triangle in matrix form
    Then take nth row corresponding to system dimension
    Then sum over first N elements corresponding to truncation level
    '''
    def number_density_matrices(self):
        return int(np.sum(self.pascals_triangle()[self.num_aux_dm_indices-1][:self.truncation_level]))
    
    def M_dimension(self):
        return self.number_density_matrices() * self.system_dimension**2
        
    # constructs initial vector from initial system density matrix
    def construct_init_vector(self):
        init_vector = np.zeros(self.M_dimension(), dtype='complex64')
        init_vector[:self.system_dimension**2] = self.init_system_dm.flatten()
        return init_vector

    def liouvillian(self):
        L_incoherent = self.incoherent_superoperator() if np.any(self.jump_operators) and np.any(self.jump_rates) else 0
        return sp.lil_matrix(-1.j*self.commutator_to_superoperator(self.system_hamiltonian) + L_incoherent, dtype='complex64')
    
    def drude_temperature_correction(self):
        # analytic form of sum from k=1 -> k=infty of c_j/nu_j
        full_sum = (self.beta*self.drude_cutoff)**-1 * (2.*self.drude_reorg_energy - self.beta*self.drude_cutoff*self.drude_reorg_energy*(1./np.tan(self.beta*self.drude_cutoff/2.)))

        def kth_term(k):
            return (4.*self.drude_reorg_energy*self.drude_cutoff) / (self.beta*(self.matsubara_freqs[k]**2 - self.drude_cutoff**2))
        
        return full_sum - np.sum(kth_term(range(self.num_matsubara_freqs)))
    
    def mode_temperature_correction(self, mode_idx):
        freq = self.underdamped_mode_params[mode_idx][0]
        reorg_energy = freq * self.underdamped_mode_params[mode_idx][1]
        damping = self.underdamped_mode_params[mode_idx][2]
        zeta = np.sqrt(freq**2 - (damping**2/4.))
        
        full_sum = (reorg_energy/(2.*zeta)) * ((np.sin(self.beta*damping/2.) + damping*np.sinh(self.beta*zeta)) / \
                                               (np.cos(self.beta*damping/2.) - np.cosh(self.beta*zeta))) \
                                               + (2.*damping / (self.beta*freq**2))
                                               
        def kth_term(k):
            return (4.*reorg_energy*damping*freq**2) / (self.beta*((freq**2 + self.matsubara_freqs[k]**2)**2 - (damping * self.matsubara_freqs[k])**2))
        
        return full_sum - np.sum(kth_term(range(self.num_matsubara_freqs)))
        
    def construct_commutator_operators(self):
        Vx_operators = np.zeros((self.heom_sys_dim, self.system_dimension**2, self.system_dimension**2))
        Vo_operators = np.zeros((self.heom_sys_dim, self.system_dimension**2, self.system_dimension**2))
        for i,j in enumerate(np.nonzero(self.sites_to_couple)[0]):
            coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
            coupling_operator[j,j] = 1.
            Vx_operators[i] = self.commutator_to_superoperator(coupling_operator)
            Vo_operators[i] = self.commutator_to_superoperator(coupling_operator, type='+')
        return Vx_operators, Vo_operators
    
    def commutator_to_superoperator(self, operator, type='-'):
        if type == '-':
            z = -1.
        elif type == '+':
            z  = 1.
        else:
            raise 'Invalid commutator type defined in commutator_to_superoperator function'
        
        return  np.kron(operator, np.eye(self.system_dimension)) + z * np.kron(np.eye(self.system_dimension), operator)
    
    def incoherent_superoperator(self):
        L = np.zeros((self.system_dimension**2, self.system_dimension**2), dtype='complex64')
        I = np.eye(self.system_dimension)
        for i in range(self.jump_rates.size):
            A = self.jump_operators[i]
            A_dagger = A.conj().T
            A_dagger_A = np.dot(A_dagger, A)
            L += self.jump_rates[i] * (np.kron(A, A) - 0.5 * np.kron(A_dagger_A, I) - 0.5 * np.kron(I, A_dagger_A))
        return L
    
    def construct_hierarchy_matrix_super_fast(self):
        num_dms = self.number_density_matrices()
        n_vectors, higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
                lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices = generate_hierarchy_and_tier_couplings(num_dms, self.num_aux_dm_indices, self.truncation_level, \
                                                                                       self.dm_per_tier())
        
        # now build the hierarchy matrix
        # diag bits
        hm = sp.kron(sp.eye(num_dms, dtype='complex64'), self.liouvillian())
        
        # n.gamma bit on diagonal
        #n_vectors = np.array(n_vectors)
#         diag_stuff = np.zeros(n_vectors.shape)
#         diag_stuff[:,:self.heom_sys_dim] = n_vectors[:,:self.heom_sys_dim]
#         coeff_vector = self.drude_zeroth_order_freqs
#         if self.num_modes:
#             for i in range(self.num_modes):
#                 neg_mode_start_idx = self.heom_sys_dim*(1+2*i)
#                 pos_mode_start_idx = self.heom_sys_dim*(2+2*i)
#                 diag_stuff[:,neg_mode_start_idx:pos_mode_start_idx] = n_vectors[:,neg_mode_start_idx:pos_mode_start_idx] + n_vectors[:,pos_mode_start_idx:pos_mode_start_idx+self.heom_sys_dim]
#                 diag_stuff[:,pos_mode_start_idx:pos_mode_start_idx+self.heom_sys_dim] = n_vectors[:,neg_mode_start_idx:pos_mode_start_idx] - n_vectors[:,pos_mode_start_idx:pos_mode_start_idx+self.heom_sys_dim]
#                 # build vector of drude_zeroth_order_freqs + -0.5*mode_dampings + 1j*mode_zetas
#                 coeff_vector = np.append(coeff_vector, 0.5*self.BO_zeroth_order_freqs[i])
#                 coeff_vector = np.append(coeff_vector, -1.j*self.BO_zetas[i])
#                 
#         if self.num_matsubara_freqs:
#             diag_stuff[:,self.heom_sys_dim*(1+2*self.num_modes):] = n_vectors[:,self.heom_sys_dim*(1+2*self.num_modes):]
#             for k in range(self.num_matsubara_freqs):
#                 mfk = np.zeros(self.heom_sys_dim)
#                 mfk.fill(self.matsubara_freqs[k])
#                 coeff_vector = np.append(coeff_vector, mfk)
        
        diag_vectors = n_vectors
        aux_dm_idx = 0
        for site in self.environment:
            if site:
                for osc in site:
                    if isinstance(osc, OBOscillator):
                        aux_dm_idx += 1
                    elif isinstance(osc, UBOscillator):
                        diag_vectors[:,aux_dm_idx] = n_vectors[:,aux_dm_idx] + n_vectors[:,aux_dm_idx+1]
                        diag_vectors[:,aux_dm_idx+1] = n_vectors[:,aux_dm_idx+1] - n_vectors[:,aux_dm_idx] 
                        aux_dm_idx += 2
                aux_dm_idx += self.num_matsubara_freqs

        hm -= sp.kron(sp.diags(np.dot(diag_vectors, self.diag_coeffs), dtype='complex64'), sp.eye(self.system_dimension**2, dtype='complex64'))
        # include temperature correction / Markovian truncation term for Matsubara frequencies
        if self.temperature_correction:
            hm -= sp.kron(sp.eye(self.number_density_matrices(), dtype='complex64'), np.sum(self.tc_terms, axis=0)).astype('complex64')
#             tc_term = self.drude_temperature_correction()
#             for i in range(self.num_modes):
#                 tc_term += self.mode_temperature_correction(i)
#             Vx_squared = np.sum(np.array([np.dot(Vx,Vx) for Vx in self.temp_correction_Vx_ops]), axis=0)
#             hm -= sp.kron(sp.eye(self.number_density_matrices(), dtype='complex64').multiply(tc_term), Vx_squared).astype('complex64')
        
        # off diag bits
        for n in range(self.num_aux_dm_indices):
            higher_coupling_matrix = sp.coo_matrix((higher_coupling_elements[n], (higher_coupling_row_indices[n], higher_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            lower_coupling_matrix = sp.coo_matrix((lower_coupling_elements[n], (lower_coupling_row_indices[n], lower_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            hm += sp.kron(higher_coupling_matrix.multiply(self.phix_coeffs[n]) + lower_coupling_matrix.multiply(self.thetax_coeffs[n]), self.Vx_operators[n]) \
                            + sp.kron(lower_coupling_matrix.multiply(self.thetao_coeffs[n]), self.Vo_operators[n])
        
        return hm.astype('complex64')
    
    def extract_system_density_matrix(self, hierarchy_vector):
        sys_dm = hierarchy_vector[:self.system_dimension**2]
        sys_dm.shape = self.system_dimension, self.system_dimension
        return sys_dm
    
    def calculate_time_evolution(self, time_step, duration):
        
        hierarchy_matrix = self.construct_hierarchy_matrix_super_fast()
            
        time = np.arange(0,duration+time_step,time_step)
        init_state, t0 = self.construct_init_vector(), 0
        
        def f(t, rho):
            return hierarchy_matrix.dot(rho)

        r = ode(f).set_integrator('zvode', method='bdf')
        
        dm_history = []
        dm_history.append(self.extract_system_density_matrix(init_state))
        
        r.set_initial_value(init_state, t0)
        while r.successful() and r.t < duration:
            dm_history.append(self.extract_system_density_matrix(r.integrate(r.t+time_step)))
        return np.array(dm_history), time
    
    # wrapper for time evolution which includes convergence testing
    def converged_time_evolution(self, init_state, init_trunc, max_trunc, time_step, duration, accuracy=0.01): #, sparse=False):
        if init_trunc > max_trunc:
            raise Exception("init_trunc must be less than or equal to max_trunc")
        #accuracy = 0.01 #need to think about proper accuracy here
        
        self.init_system_dm = init_state
        
        # first check previous level in case hierarchy has converged first time
        self.truncation_level = init_trunc
        current_history, time = self.calculate_time_evolution(time_step, duration)
        self.truncation_level = init_trunc - 1
        previous_history, time = self.calculate_time_evolution(time_step, duration)
        
        print current_history.shape
        print previous_history.shape
        
        # if not already converged re-calculate time evolution at incrementally higher orders of truncation until convergence is reached    
        if self.check_dm_history_convergence(current_history, previous_history, accuracy):
            print "Hierarchy converged first time!"
        elif init_trunc < max_trunc:
            converged = False
            current_level = init_trunc
            while not converged and self.truncation_level < max_trunc:
                self.truncation_level = current_level + 1
                next_history, time = self.calculate_time_evolution(time_step, duration)
                converged = self.check_dm_history_convergence(current_history, next_history, accuracy)
                current_level += 1
                current_history = next_history
            if converged:
                print "Hierarchy converged at N = " + str(current_level)
            else:
                print "Hierarchy did not converge but reached max truncation level"
        else:
            print "Hierarchy did not converge but reached max truncation level"
        return current_history, time
    
    def transform_to_exciton_basis(self, dm_history):
        return np.array([np.dot(self.system_evectors.T, np.dot(dm_history[i], self.system_evectors)) for i in range(dm_history.shape[0])])
            
    def check_dm_history_convergence(self, history1, history2, accuracy):
        if history1.size != history2.size:
            raise('Density matrix histories in check_dm_history_convergence are not of same size!')
        
        converged = True
        history1_pops = [dm[0,0] for dm in history1]
        history2_pops = [dm[0,0] for dm in history2]
        for i in range(history1.shape[0]):
            if np.abs(np.real(history1_pops[i] - history2_pops[i])) > accuracy:
                converged = False
                break
        return converged
    
    # diagonalise hierarchy matrix
    # find hierarchy vector corresponding to zero eigenvalue
    # extract system density matrix from hierarchy vector
    # test for convergence...
    def calculate_steady_state(self, init_trunc, max_trunc, exciton_basis=False):
        
        accuracy = 0.0001
        
        if init_trunc > max_trunc:
            raise Exception("init_trunc must be less than or equal to max_trunc")        
        
        self.truncation_level = init_trunc
        current_steady_state = self.normalise_steady_state(self.extract_system_density_matrix(self.hierarchy_steady_state()))
        self.truncation_level = init_trunc - 1
        previous_steady_state = self.normalise_steady_state(self.extract_system_density_matrix(self.hierarchy_steady_state()))
        
        if self.check_steady_state_convergence(current_steady_state, previous_steady_state, accuracy):
            print "Steady state converged first time!"
        elif init_trunc < max_trunc:
            converged = False
            current_level = init_trunc
            while not converged and self.truncation_level < max_trunc:
                self.truncation_level = current_level + 1
                next_steady_state = self.normalise_steady_state(self.extract_system_density_matrix(self.hierarchy_steady_state()))
                converged = self.check_steady_state_convergence(current_steady_state, next_steady_state, accuracy)
                current_level += 1
                current_steady_state = next_steady_state
            if converged:
                print "Steady state converged at N = " + str(current_level)
            else:
                print "Steady state did not converge but reached max truncation level"
        else:
            print "Steady state did not converge but max truncation level was reached"
            
        return current_steady_state
    
    def init_steady_state_vector(self, init_state, time_step, duration):
        
        hierarchy_matrix = self.construct_hierarchy_matrix_fast()
            
        time = np.arange(0,duration+time_step,time_step)
        init_vector = np.zeros(self.M_dimension(), dtype='complex64')
        init_vector[:self.system_dimension**2] = init_state.flatten()
        t0 = 0
        
        def f(t, rho):
            return hierarchy_matrix.dot(rho)
        
        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(init_vector, t0)
        
        current_state = 0
        while r.successful() and r.t < duration:
            current_state = r.integrate(r.t+time_step)

        return current_state
        
    def hierarchy_steady_state(self):
        hierarchy_matrix = self.construct_hierarchy_matrix_fast()
        init_state = self.construct_init_vector()
        evalue, steady_state = spla.eigs(hierarchy_matrix.tocsc(), k=1, sigma=0, which='LM', v0=init_state) # using eigs in shift-invert mode by setting sigma=0 and which='LM'
        print 'calculated steady state'
        return steady_state
    
    def normalised_steady_state(self):
        return self.normalise_steady_state(self.extract_system_density_matrix(self.hierarchy_steady_state()))
    
    def normalise_steady_state(self, density_matrix):
        try:
            return density_matrix / np.trace(density_matrix)
        except ValueError:
            raise Exception("Could not normalise density matrix, check it is 2D")
    
    def find_nearest_index(self, array, value):
        return np.abs(array-value).argmin()
        
    def check_steady_state_convergence(self, current_steady_state, previous_steady_state, accuracy):
        return np.all(np.abs(current_steady_state-previous_steady_state) < accuracy)
