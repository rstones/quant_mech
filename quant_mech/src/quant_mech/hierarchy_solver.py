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

class HierarchySolver(object):
    '''
    classdocs
    '''

    def __init__(self, hamiltonian, drude_reorg_energy, drude_cutoff, beta, jump_operators=None, jump_rates=None, underdamped_mode_params=[], num_matsubara_freqs=0, temperature_correction=False):
        '''
        Constructor
        
        Currently assumes identical Drude spectral density on each site
        
        '''
        self.init_system_dm = None
        self.drude_reorg_energy = drude_reorg_energy
        self.drude_cutoff = drude_cutoff
        #self.temperature = temperature
        '''
        get user to provide beta in appropriate units
        '''
        self.beta = beta #1. / (utils.KELVIN_TO_WAVENUMS * self.temperature)
        self.system_hamiltonian = hamiltonian
        self.system_evalues, self.system_evectors = utils.sorted_eig(self.system_hamiltonian)
        self.system_evectors = self.system_evectors.T
        
        self.system_dimension = self.system_hamiltonian.shape[0]
        
        # zeroth order freqs for identical Drude spectral density on each site are just cutoff freq
        self.drude_zeroth_order_freqs = np.zeros(self.system_dimension)
        self.drude_zeroth_order_freqs.fill(self.drude_cutoff)
        
        # underdamped_mode_params should be a list of tuples
        # each tuple having frequency, Huang-Rhys factor and damping for each mode in that order
        self.underdamped_mode_params = underdamped_mode_params
        self.num_modes = len(self.underdamped_mode_params)
        if np.any(self.underdamped_mode_params):
            self.BO_zeroth_order_freqs = np.zeros((self.num_modes, self.system_dimension))
            self.BO_zetas = np.zeros((self.num_modes, self.system_dimension))
            for i,mode in enumerate(self.underdamped_mode_params):
                self.BO_zeroth_order_freqs[i].fill(mode[2])
                self.BO_zetas[i].fill(np.sqrt(mode[0]**2 - mode[2]**2/4.))
        
        self.num_matsubara_freqs = num_matsubara_freqs
        self.matsubara_freqs = self.calculate_matsubara_freqs()
        
        self.temperature_correction = temperature_correction
        
        self.num_aux_dm_indices = (1 + 2*self.num_modes)*self.system_dimension + self.num_matsubara_freqs
        
        # calculate coefficients of operators coupling to lower tiers and fill in vectors
        self.thetax_coeffs = np.zeros(self.num_aux_dm_indices, dtype='complex64')
        self.thetao_coeffs = np.zeros(self.num_aux_dm_indices, dtype='complex64')
        self.scaling_factors = np.zeros(self.num_aux_dm_indices, dtype='complex64')
        self.thetax_coeffs[:self.system_dimension].fill(self.drude_Vx_coeff())
        self.thetao_coeffs[:self.system_dimension].fill(self.drude_Vo_coeff())
        self.scaling_factors[:self.system_dimension].fill(self.drude_scaling_factor())
        for i in range(self.num_modes):
            freq = self.underdamped_mode_params[i][0]
            reorg_energy = self.underdamped_mode_params[i][0]*self.underdamped_mode_params[i][1]
            damping = self.underdamped_mode_params[i][2]
            self.thetax_coeffs[self.system_dimension*(1+2*i):self.system_dimension*(2+2*i)].fill(self.mode_Vx_coeff(freq, reorg_energy, damping, -1.))
            self.thetax_coeffs[self.system_dimension*(2+2*i):self.system_dimension*(3+2*i)].fill(self.mode_Vx_coeff(freq, reorg_energy, damping, 1.))
            self.thetao_coeffs[self.system_dimension*(1+2*i):self.system_dimension*(2+2*i)].fill(self.mode_Vo_coeff(freq, reorg_energy, damping, -1.))
            self.thetao_coeffs[self.system_dimension*(2+2*i):self.system_dimension*(3+2*i)].fill(self.mode_Vo_coeff(freq, reorg_energy, damping, 1.))
            self.scaling_factors[self.system_dimension*(1+2*i):self.system_dimension*(2+2*i)].fill(self.mode_scaling_factor(freq, reorg_energy, damping, -1.))
            self.scaling_factors[self.system_dimension*(2+2*i):self.system_dimension*(3+2*i)].fill(self.mode_scaling_factor(freq, reorg_energy, damping, 1.))
        # still need to do Matsubara freqs for this
        
        self.jump_operators = jump_operators
        self.jump_rates = jump_rates
        
        self.Vx_operators, self.Vo_operators = self.construct_commutator_operators() # commutator operators for each site
        #self.theta_drude_operators = self.construct_theta_drude_operators() # theta operator for each site
        
    def calculate_matsubara_freqs(self):
        return np.array([2.*np.pi*k / self.beta for k in range(1,self.num_matsubara_freqs+1)])
        
    def drude_Vx_coeff(self):
        return 1.j*self.drude_reorg_energy*self.drude_cutoff / np.tan(self.beta*self.drude_cutoff/2.)
    
    def drude_Vo_coeff(self):
        return self.drude_reorg_energy*self.drude_cutoff
    
    '''
    Absolute value of the leading coefficient from exponential expansion of the Drude spectral density
    '''
    def drude_scaling_factor(self):
        return np.abs(self.drude_cutoff*self.drude_reorg_energy * (1./np.tan(self.beta*self.drude_cutoff/2.) + 1.j))
    
    def mode_Vx_coeff(self, freq, reorg_energy, damping, plus_or_minus):
        zeta = np.sqrt(freq**2 - (damping**2/4.))
        return -plus_or_minus*1.j*(reorg_energy*freq**2/(2.*zeta)) / np.tanh((1.j*self.beta/4.)*(damping + plus_or_minus*2.j*zeta))
    
    def mode_Vo_coeff(self, freq, reorg_energy, damping, plus_or_minus):
        zeta = np.sqrt(freq**2 - (damping**2/4.))
        return plus_or_minus*1.j*(reorg_energy*freq**2/(2.*zeta))
    
    '''
    Absolute value of the leading coefficient from exponential expansion of Brownian oscillator spectral density
    '''
    def mode_scaling_factor(self, freq, reorg_energy, damping, plus_or_minus):
        zeta = np.sqrt(freq**2 - (damping**2/4.))
        nu = damping/2. + plus_or_minus*1.j*zeta
        return np.abs(plus_or_minus*1.j*(reorg_energy*freq**2/(2.*zeta)) * (1./np.tan(nu*self.beta/2.) - 1.j))

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
        return np.sum(self.pascals_triangle()[self.num_aux_dm_indices-1][:self.truncation_level])
    
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
        Vx_operators = np.zeros((self.system_dimension, self.system_dimension**2, self.system_dimension**2))
        Vo_operators = np.zeros((self.system_dimension, self.system_dimension**2, self.system_dimension**2))
        for i in range(self.system_dimension):
            coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
            coupling_operator[i,i] = 1.
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
#         n_vectors, higher_coupling_matrices, lower_coupling_matrices = generate_hierarchy_and_tier_couplings(num_dms, self.num_aux_dm_indices, self.truncation_level, \
#                                                                                                    self.dm_per_tier(), self.scaling_factors)
        n_vectors, higher_coupling_elements, higher_coupling_row_indices, higher_coupling_column_indices, \
                            lower_coupling_elements, lower_coupling_row_indices, lower_coupling_column_indices = generate_hierarchy_and_tier_couplings(num_dms, self.num_aux_dm_indices, self.truncation_level, \
                                                                                                   self.dm_per_tier(), self.scaling_factors)
        
        # now build the hierarchy matrix
        # diag bits
        hm = sp.kron(sp.eye(num_dms), self.liouvillian())
        
        # n.gamma bit on diagonal
        #n_vectors = np.array(n_vectors)
        diag_stuff = np.zeros(n_vectors.shape)
        diag_stuff[:,:self.system_dimension] = n_vectors[:,:self.system_dimension]
        coeff_vector = self.drude_zeroth_order_freqs
        if self.num_matsubara_freqs:
            pass
        if self.num_modes:
            for i in range(self.num_modes):
                neg_mode_start_idx = self.system_dimension*(1+2*i)
                pos_mode_start_idx = self.system_dimension*(2+2*i)
                diag_stuff[:,neg_mode_start_idx:pos_mode_start_idx] = n_vectors[:,neg_mode_start_idx:pos_mode_start_idx] + n_vectors[:,pos_mode_start_idx:pos_mode_start_idx+self.system_dimension]
                diag_stuff[:,pos_mode_start_idx:pos_mode_start_idx+self.system_dimension] = n_vectors[:,neg_mode_start_idx:pos_mode_start_idx] - n_vectors[:,pos_mode_start_idx:pos_mode_start_idx+self.system_dimension]
                # build vector of drude_zeroth_order_freqs + -0.5*mode_dampings + 1j*mode_zetas
                coeff_vector = np.append(coeff_vector, 0.5*self.BO_zeroth_order_freqs[i])
                coeff_vector = np.append(coeff_vector, -1.j*self.BO_zetas[i])
        
        hm -= sp.kron(sp.diags(np.dot(diag_stuff, coeff_vector)), sp.eye(self.system_dimension**2))
        
        # include temperature correction / Markovian truncation term for Matsubara frequencies
        if self.temperature_correction:
            tc_term = self.drude_temperature_correction()
            for i in range(self.num_modes):
                tc_term += self.mode_temperature_correction(i)
            Vx_squared = np.sum(np.array([np.dot(Vx,Vx) for Vx in self.Vx_operators]), axis=0)
            hm -= sp.kron(tc_term*sp.eye(self.number_density_matrices()), Vx_squared)
        
        # off diag bits
        for n in range(self.num_aux_dm_indices):
#             hm += sp.kron((higher_coupling_matrices[n] * 1.j) + (lower_coupling_matrices[n] * self.thetax_coeffs[n]), self.Vx_operators[n%self.system_dimension]) \
#                             + sp.kron(lower_coupling_matrices[n] * self.thetao_coeffs[n], self.Vo_operators[n%self.system_dimension])
            higher_coupling_matrix = sp.coo_matrix((higher_coupling_elements[n], (higher_coupling_row_indices[n], higher_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            lower_coupling_matrix = sp.coo_matrix((lower_coupling_elements[n], (lower_coupling_row_indices[n], lower_coupling_column_indices[n])), shape=(num_dms, num_dms)).tocsr()
            hm += sp.kron(higher_coupling_matrix.multiply(1.j) + lower_coupling_matrix.multiply(self.thetax_coeffs[n]), self.Vx_operators[n%self.system_dimension]) \
                            + sp.kron(lower_coupling_matrix.multiply(self.thetao_coeffs[n]), self.Vo_operators[n%self.system_dimension])
        
        return hm
    
    def extract_system_density_matrix(self, hierarchy_vector):
        sys_dm = hierarchy_vector[:self.system_dimension**2]
        sys_dm.shape = self.system_dimension, self.system_dimension
        return sys_dm
    
    def calculate_time_evolution(self, time_step, duration):
        
        hierarchy_matrix = self.construct_hierarchy_matrix_super_fast()
        
#         if params_in_wavenums:
#             hierarchy_matrix = hierarchy_matrix.multiply(utils.WAVENUMS_TO_INVERSE_PS)
            
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
    def converged_time_evolution(self, init_state, init_trunc, max_trunc, time_step, duration, accuracy=0.01, params_in_wavenums=True): #, sparse=False):
        if init_trunc > max_trunc:
            raise Exception("init_trunc must be less than or equal to max_trunc")
        #accuracy = 0.01 #need to think about proper accuracy here
        
        self.init_system_dm = init_state
        
        # first check previous level in case hierarchy has converged first time
        self.truncation_level = init_trunc
        current_history, time = self.calculate_time_evolution(time_step, duration, params_in_wavenums)# if not sparse else self.calculate_time_evolution_sparse(time_step, duration, params_in_wavenums)
        self.truncation_level = init_trunc - 1
        previous_history, time = self.calculate_time_evolution(time_step, duration, params_in_wavenums)# if not sparse else self.calculate_time_evolution_sparse(time_step, duration, params_in_wavenums)[0]
        
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
                next_history, time = self.calculate_time_evolution(time_step, duration, params_in_wavenums)# if not sparse else self.calculate_time_evolution_sparse(time_step, duration, params_in_wavenums)[0]
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
    
    def init_steady_state_vector(self, init_state, time_step, duration, params_in_wavenums=True):
        
        hierarchy_matrix = self.construct_hierarchy_matrix_fast()
#         print "Memory usage of hierarchy matrix: " \
#                     + str((hierarchy_matrix.data.nbytes+hierarchy_matrix.indptr.nbytes+hierarchy_matrix.indices.nbytes) / 1.e9) \
#                     + "Gb"
        
        if params_in_wavenums:
            hierarchy_matrix = hierarchy_matrix.multiply(utils.WAVENUMS_TO_INVERSE_PS)
            
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
