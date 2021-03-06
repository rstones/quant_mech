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
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator

class HierarchySolver(object):
    '''
    classdocs
    '''
    
    def __init__(self, hamiltonian, environment, beta, jump_operators=None, jump_rates=None, N=1, \
                 num_matsubara_freqs=0, temperature_correction=False, dissipator_test=False):
        '''
        Constructor
        '''
        self.system_hamiltonian = hamiltonian
        self.environment = environment
        self.beta = beta
        self.jump_operators = jump_operators
        self.jump_rates = jump_rates
        self.truncation_level = N
        self.num_matsubara_freqs = num_matsubara_freqs
        self.temperature_correction = temperature_correction
        
        '''
        dissipator_test is a flag to include the Lindblad dissipator in the top level of 
        the hierarchy only
        It is used in self.liouvillian() and self.construct_hierarchy_matrix_super_fast()
        '''
        self.dissipator_test = dissipator_test
        
        self.system_evalues, self.system_evectors = utils.sorted_eig(self.system_hamiltonian)
        self.system_evectors = self.system_evectors.T
        self.system_dimension = self.system_hamiltonian.shape[0]
        
        if self.num_matsubara_freqs>0 or self.temperature_correction:
            self.matsubara_freqs = self.calculate_matsubara_freqs()
        self.Vx_operators, self.Vo_operators = [],[]
        
        # if tuple or UBOscillator or OBOscillator put into list of tuples
        if type(self.environment) is tuple: # multiple oscillators, identical on each site
            self.environment = [self.environment] * self.system_dimension
        elif isinstance(environment, (OBOscillator, UBOscillator)): # single oscillator identical on each site
            self.environment = [(self.environment,)] * self.system_dimension
        
        if type(self.environment) is list:# environment defined per site
            self.diag_coeffs = []
            self.phix_coeffs = []
            self.thetax_coeffs = []
            self.thetao_coeffs = []
            self.tc_terms = []
            for i,site in enumerate(self.environment):
                if site:
                    site_coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
                    site_coupling_operator[i,i] = 1.
                    site_Vx_operator = self.commutator_to_superoperator(site_coupling_operator, type='-')
                    site_Vo_operator = self.commutator_to_superoperator(site_coupling_operator, type='+')
                    tc_term = 0
                    for osc in site:
                        if isinstance(osc, OBOscillator):
                            self.diag_coeffs.append(osc.cutoff_freq)
                            self.phix_coeffs.append(self.phix_coeff_OBO(osc))
                            #site_coupling_operator = np.zeros(self.system_dimension)
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
                        tc_term += osc.temp_correction_sum() \
                                        - np.sum([osc.temp_correction_sum_kth_term(k) for k in range(1,self.num_matsubara_freqs+1)])
                    self.tc_terms.append(tc_term * np.dot(site_Vx_operator, site_Vx_operator))
                    for k in range(1,self.num_matsubara_freqs+1):
                        self.diag_coeffs.append(self.matsubara_freqs[k-1])
                        self.phix_coeffs.append(self.phix_coeff_MF(site, k))
                        self.thetax_coeffs.append(self.thetax_coeff_MF(site, k))
                        self.thetao_coeffs.append(0)
                        self.Vx_operators.append(site_Vx_operator)
                        self.Vo_operators.append(site_Vo_operator)
        else:
            raise ValueError("Environment defined in invalid format!")

        self.num_aux_dm_indices = len(self.diag_coeffs)
        self.diag_coeffs = np.array(self.diag_coeffs)
        self.phix_coeffs = np.array(self.phix_coeffs)
        self.thetax_coeffs = np.array(self.thetax_coeffs)
        self.thetao_coeffs = np.array(self.thetao_coeffs)
        self.Vx_operators = np.array(self.Vx_operators)
        self.Vo_operators = np.array(self.Vo_operators)

    def calculate_matsubara_freqs(self):
        return np.array([2.*np.pi*k / self.beta for k in range(1,self.num_matsubara_freqs+1)])
                            
    def phix_coeff_OBO(self, osc):
        return 1.j * np.sqrt(np.sqrt(np.abs(osc.coeffs[0] * osc.coeffs[0].conj())))
        
    def thetax_coeff_OBO(self, osc):
        return 1.j * (1. / np.sqrt(np.sqrt(np.abs(osc.coeffs[0] * osc.coeffs[0].conj())))) \
                            * osc.reorg_energy*osc.cutoff_freq / np.tan(osc.beta*osc.cutoff_freq/2.)
    
    def thetao_coeff_OBO(self, osc):
        return (1. / np.sqrt(np.sqrt(np.abs(osc.coeffs[0] * osc.coeffs[0].conj())))) \
                            * osc.reorg_energy*osc.cutoff_freq
                        
    def phix_coeff_UBO(self, osc, pm=1):
        return 1.j * np.sqrt(np.sqrt(np.abs( osc.coeffs[0 if pm>0 else 1] * osc.coeffs[1 if pm>0 else 0].conj() )))
        
    def thetax_coeff_UBO(self, osc, pm=1):
        return (-pm*1.j / np.sqrt(np.sqrt(np.abs( \
                    osc.coeffs[0 if pm>0 else 1] * osc.coeffs[1 if pm>0 else 0].conj() )))) \
                        *(osc.reorg_energy*osc.freq**2/(2.*osc.zeta)) / np.tanh((1.j*self.beta/4.)*(osc.damping + pm*2.j*osc.zeta))
                        
    def thetao_coeff_UBO(self, osc, pm=1):
        return (pm*1.j  / np.sqrt(np.sqrt(np.abs( \
                    osc.coeffs[0 if pm>0 else 1] * osc.coeffs[1 if pm>0 else 0].conj() )))) \
                        *(osc.reorg_energy*osc.freq**2 / (2.*osc.zeta))
    
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
        return self.pascals_triangle()[self.num_aux_dm_indices-1][:self.truncation_level+1].astype('int64')
    
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
        if not self.dissipator_test:
            L_incoherent = self.incoherent_superoperator() if np.any(self.jump_operators) and np.any(self.jump_rates) else 0
        else:
            L_incoherent = 0
        return sp.lil_matrix(-1.j*self.commutator_to_superoperator(self.system_hamiltonian) + L_incoherent, dtype='complex128')
    
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
        
        return  np.kron(operator, np.eye(self.system_dimension)) + z * np.kron(np.eye(self.system_dimension), operator.conj())
    
    def incoherent_superoperator(self):
        L = np.zeros((self.system_dimension**2, self.system_dimension**2), dtype='complex128')
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
        
        dtype = 'complex128'
        
        # now build the hierarchy matrix
        # diag bits
        hm = sp.kron(sp.eye(num_dms, dtype=dtype), self.liouvillian())
        
        if self.dissipator_test: # need to have jump_operators and rates in conjunction with dissipator_test = True
            top_level = sp.lil_matrix((num_dms, num_dms), dtype=dtype)
            top_level[0,0] = 1.
            hm += sp.kron(top_level, self.incoherent_superoperator())
        
        diag_vectors = np.copy(n_vectors)
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

        hm -= sp.kron(sp.diags(np.dot(diag_vectors, self.diag_coeffs), dtype=dtype), sp.eye(self.system_dimension**2, dtype=dtype))
        # include temperature correction / Markovian truncation term for Matsubara frequencies
        if self.temperature_correction:
            hm -= sp.kron(sp.eye(self.number_density_matrices(), dtype=dtype), np.sum(self.tc_terms, axis=0)).astype(dtype)
        
        # off diag bits        
        for n in range(self.num_aux_dm_indices):
            higher_coupling_matrix = sp.coo_matrix((higher_coupling_elements[n], (higher_coupling_row_indices[n], higher_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            lower_coupling_matrix = sp.coo_matrix((lower_coupling_elements[n], (lower_coupling_row_indices[n], lower_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            hm -= sp.kron(higher_coupling_matrix.multiply(self.phix_coeffs[n]) + lower_coupling_matrix.multiply(self.thetax_coeffs[n]), self.Vx_operators[n]) \
                            + sp.kron(lower_coupling_matrix.multiply(self.thetao_coeffs[n]), self.Vo_operators[n])
        
        return hm.astype(dtype)
    
    def construct_efficient_steady_state_solver_matrices(self, epsilon):
        num_dms = self.number_density_matrices()
        n_vectors, higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
                lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices = generate_hierarchy_and_tier_couplings(num_dms, self.num_aux_dm_indices, self.truncation_level, \
                                                                                       self.dm_per_tier())
        
        dtype = 'complex128'
        
        # first construct inverse operator
        op_to_invert = sp.kron(sp.eye(num_dms, dtype=dtype), -self.liouvillian() + epsilon*np.eye(self.system_dimension**2))
         
        diag_vectors = np.copy(n_vectors)
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
 
        op_to_invert += sp.kron(sp.diags(np.dot(diag_vectors, self.diag_coeffs), dtype=dtype), sp.eye(self.system_dimension**2, dtype=dtype))
        inverse_op = spla.inv(op_to_invert)
        
        # now construct hierarchy matrix starting with relaxation parameter
        hm = sp.eye(self.M_dimension(), dtype=dtype).multiply(epsilon) #sp.kron(epsilon*sp.eye(num_dms, dtype=dtype), sp.eye(self.system_dimension**2, dtype=dtype))

        # include temperature correction / Markovian truncation term for Matsubara frequencies
        if self.temperature_correction:
            hm -= sp.kron(sp.eye(self.number_density_matrices(), dtype=dtype), np.sum(self.tc_terms, axis=0)).astype(dtype)
        
        # off diag bits
        for n in range(self.num_aux_dm_indices):
            higher_coupling_matrix = sp.coo_matrix((higher_coupling_elements[n], (higher_coupling_row_indices[n], higher_coupling_column_indices[n])), shape=(num_dms, num_dms), dtype=dtype).tocsr()
            lower_coupling_matrix = sp.coo_matrix((lower_coupling_elements[n], (lower_coupling_row_indices[n], lower_coupling_column_indices[n])), shape=(num_dms, num_dms), dtype=dtype).tocsr()
            hm -= sp.kron(higher_coupling_matrix.multiply(self.phix_coeffs[n]) + lower_coupling_matrix.multiply(self.thetax_coeffs[n]), self.Vx_operators[n]) \
                            + sp.kron(lower_coupling_matrix.multiply(self.thetao_coeffs[n]), self.Vo_operators[n])

        return hm.tocsc().astype(dtype), inverse_op.tocsr()
    
    def efficient_steady_state_solver(self, epsilon, init_guess):
        solver_matrix, inverse_op = self.construct_efficient_steady_state_solver_matrices(epsilon)
        error_tolerance = 1.e-3
        error = 1.
        steady_state = init_guess.copy()
        dm_per_tier = self.dm_per_tier()
        
        prev_steady_state = init_guess[:self.system_dimension**2].copy()
        count = 0
        
        while error > error_tolerance:
             
            for i in range(len(dm_per_tier)-1):
                tier_start = np.sum(dm_per_tier[:i]) * self.system_dimension**2
                tier_end = np.sum(dm_per_tier[:i+1]) * self.system_dimension**2
                slice_start = (np.sum(dm_per_tier[:i-1]) if i != 0 else 0) * self.system_dimension**2
                slice_end = np.sum(dm_per_tier[:i+2]) * self.system_dimension**2
                if slice_end > self.M_dimension(): # fix if the slice is extending beyond the size of the solver matrix
                    slice_end = self.M_dimension()
                steady_state[tier_start:tier_end] = solver_matrix[tier_start:tier_end, slice_start:slice_end].dot(steady_state[slice_start:slice_end])
                steady_state[tier_start:tier_end] = inverse_op[tier_start:tier_end, tier_start:tier_end].dot(steady_state[tier_start:tier_end])
            
#             error = np.sum(np.abs(steady_state[:self.system_dimension**2] - prev_steady_state))
#             prev_steady_state = steady_state[:self.system_dimension**2].copy()
            count += 1
            if not count % 100:
                print(steady_state[:self.system_dimension**2])
        
        
        return steady_state
    
    def construct_hierarchy_matrix_no_numba(self, n_vectors, \
                                            higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
                                            lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices):
        num_dms = self.number_density_matrices()
#         n_vectors, higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
#                 lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices = generate_hierarchy_and_tier_couplings(num_dms, self.num_aux_dm_indices, self.truncation_level, \
#                                                                                        self.dm_per_tier())
        
        dtype = 'complex128'
        
        # now build the hierarchy matrix
        # diag bits
        hm = sp.kron(sp.eye(num_dms, dtype=dtype), self.liouvillian())
        diag_vectors = np.copy(n_vectors)
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

        hm -= sp.kron(sp.diags(np.dot(diag_vectors, self.diag_coeffs), dtype=dtype), sp.eye(self.system_dimension**2, dtype=dtype))
        # include temperature correction / Markovian truncation term for Matsubara frequencies
        if self.temperature_correction:
            hm -= sp.kron(sp.eye(self.number_density_matrices(), dtype=dtype), np.sum(self.tc_terms, axis=0)).astype(dtype)
        
        # off diag bits        
        for n in range(self.num_aux_dm_indices):
            higher_coupling_matrix = sp.coo_matrix((higher_coupling_elements[n], (higher_coupling_row_indices[n], higher_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            lower_coupling_matrix = sp.coo_matrix((lower_coupling_elements[n], (lower_coupling_row_indices[n], lower_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            hm -= sp.kron(higher_coupling_matrix.multiply(self.phix_coeffs[n]) + lower_coupling_matrix.multiply(self.thetax_coeffs[n]), self.Vx_operators[n]) \
                            + sp.kron(lower_coupling_matrix.multiply(self.thetao_coeffs[n]), self.Vo_operators[n])
        
        return hm.astype(dtype)
    
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
        
        print(current_history.shape)
        print(previous_history.shape)
        
        # if not already converged re-calculate time evolution at incrementally higher orders of truncation until convergence is reached    
        if self.check_dm_history_convergence(current_history, previous_history, accuracy):
            print("Hierarchy converged first time!")
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
                print("Hierarchy converged at N = " + str(current_level))
            else:
                print("Hierarchy did not converge but reached max truncation level")
        else:
            print("Hierarchy did not converge but reached max truncation level")
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
            print("Steady state converged first time!")
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
                print("Steady state converged at N = " + str(current_level))
            else:
                print("Steady state did not converge but reached max truncation level")
        else:
            print("Steady state did not converge but max truncation level was reached")
            
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
        print('calculated steady state')
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
