'''
Created on 21 Mar 2014

@author: rstones
'''
import numpy as np
import scipy.linalg as la
import quant_mech.open_systems as os
import quant_mech.utils as utils

class HierarchySolver(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.init_system_dm = np.array([[1.,0],[0,0]])
        self.E_reorg = 500.
        self.gamma = 53.08 # find out exactly what gamma represents in Drude spectral density (something about broadening)
        self.temperature = 300.
        self.omega_0 = 0
        self.system_hamiltonian = np.array([[100.,100.],[100.,0]])
        self.matsubara_freqs = np.array([[self.gamma, self.gamma]]).T
        self.system_coupling_operator = np.array([[1.,0],[0,-1.]])
        
        self.truncation_level = 15
        self.system_dimension = self.system_hamiltonian.shape[0]
        #self.M_dimension = self.number_density_matrices() * self.system_dimension**2
        
        
    def pascals_triangle(self):
        return la.pascal(self.system_dimension if self.system_dimension > self.truncation_level else self.truncation_level)
        
        '''
        Construct Pascal's triangle in matrix form
        Then take nth row corresponding to system dimension
        Then sum over first N elements corresponding to truncation level
        '''
    def number_density_matrices(self):
        return np.sum(self.pascals_triangle()[self.system_dimension-1][:self.truncation_level])
    
    def M_dimension(self):
        return self.number_density_matrices() * self.system_dimension**2
        
    # constructs initial vector from initial system density matrix
    def construct_init_vector(self):
        print 'Constructing initial hierarchy vector'
        init_vector = np.zeros(self.M_dimension())
        init_vector[:self.system_dimension**2] = self.init_system_dm.flatten()
        return init_vector
    
    # construct matrix to solve coupled equations
    def construct_hierarchy_matrix(self):
        M_dim = self.M_dimension() 
        print 'Constructing hierarchy matrix'
        M = np.zeros((M_dim, M_dim), dtype=np.complex64)
        n_vectors = self.generate_n_vectors()
        #matsubara_freqs = self.calculate_matsubara_freqs()
        print M.shape
        sub_matrix_size = self.system_dimension ** 2
        
        for i in range(self.number_density_matrices()):
            nth_ode = self.nth_ode(i, n_vectors, self.matsubara_freqs)
            for j,v in enumerate(nth_ode):
                if v.any():
                    M[i*sub_matrix_size:(i+1)*sub_matrix_size, j*sub_matrix_size:(j+1)*sub_matrix_size] = v
                   
        return M
    
    def extract_system_density_matrix(self, hierarchy_vector):
        sys_dm = hierarchy_vector[:self.system_dimension**2]
        sys_dm.shape = self.system_dimension, self.system_dimension
        return sys_dm
    
    # get hierarchy matrix
    # exponentiate to get time step operator
    # propagate initial state through time
    # extract system density matrix from hierarchy vector at each time step
    def calculate_time_evolution(self, time_step, duration, params_in_wavenums=True):
        
        hierarchy_matrix = self.construct_hierarchy_matrix()
        
        if params_in_wavenums:
            hierarchy_matrix = utils.WAVENUMS_TO_PS * hierarchy_matrix
            
        time = np.arange(0,duration+time_step,time_step)
        
        dm_history = np.zeros(time.size, dtype=np.ndarray)
        time_step_operator = la.expm(hierarchy_matrix * time_step)
        current_hierarchy_vector = self.construct_init_vector()
        dm_history[0] = self.extract_system_density_matrix(current_hierarchy_vector)
        for i in range(1,time.size):
            current_hierarchy_vector = np.dot(time_step_operator, current_hierarchy_vector)
            dm_history[i] = self.extract_system_density_matrix(current_hierarchy_vector)
            
        return dm_history, time
    
    # public wrapper for time evolution which includes convergence testing
    def hierarchy_time_evolution(self, init_trunc, time_step, duration, params_in_wavenums=True):
        accuracy = 0.005 #need to think about proper accuracy here
        
        # first check previous level in case hierarchy has converged first time
        self.truncation_level = init_trunc
        current_history, time = self.calculate_time_evolution(time_step, duration, params_in_wavenums)
        self.truncation_level = init_trunc - 1 
        previous_history = self.calculate_time_evolution(time_step, duration, params_in_wavenums)[0]
        
        # if not already converged re-calculate time evolution at incrementally higher orders of truncation until convergence is reached    
        if self.check_dm_history_convergence(current_history, previous_history, accuracy):
            print "Hierarchy converged first time!"
            return current_history, time
        else:
            converged = False
            current_level = init_trunc
            while not converged:
                self.truncation_level = current_level + 1 
                next_history = self.calculate_time_evolution(time_step, duration, params_in_wavenums)[0]
                converged = self.check_dm_history_convergence(current_history, next_history, accuracy)
                current_level += 1
                current_history = next_history
            print "Hierarchy converged at N = " + str(current_level)
            return current_history, time
            
    def check_dm_history_convergence(self, history1, history2, accuracy):
        if history1.size != history2.size:
            raise('Density matrix histories in check_dm_history_convergence are not of same size!')
        
        converged = True
        history1_pops = [dm[0,0] for dm in history1]
        history2_pops = [dm[0,0] for dm in history2]
        for i in range(history1.size):
            if np.abs(np.real(history1_pops[i] - history2_pops[i])) > accuracy:
                converged = False
                break
        return converged
    
    # diagonalise hierarchy matrix
    # find hierarchy vector corresponding to zero eigenvalue
    # extract system density matrix from hierarchy vector
    # test for convergence...
    def hierarchy_steady_state(self):
        pass
    
    '''
    Tests for correct convergence of hierarchy, compares 2 consecutive orders of hierarchy and will rerun calculation at higher level if not matched
    For time_evolution will need to compare populations for all times
    For steady state will need to compare steady state populations    
    '''
    def test_hierarchy_convergence(self, func):
        pass
    
    # calculates matsubara frequencies from environment correlation function
    # there are a couple of ways to do this so need to check which is best
    # for instance: I'm not sure if omega_0 is generally applicable
    # maybe use nu_jm = (2\pi m) / (\beta \hbar)
    def calculate_matsubara_freqs(self):
        matsubara_freqs = np.zeros(self.system_dimension)
        for i in range(self.system_dimension):
            matsubara_freqs[i] = self.gamma + (-1.)**i * 1.j * self.omega_0
        return matsubara_freqs
    
    # return list of operators for specified n vector with index in list corresponding to auxiliary dm the operator acts on in ode
    def nth_ode(self, n_index, n_vectors, matsubara_freqs):
        operators = [np.zeros((self.system_dimension**2, self.system_dimension**2))] * self.number_density_matrices()
        
        # diagonal elements of M
        operators[n_index] = -(1.j*self.commutator_to_superoperator(self.system_hamiltonian) + np.eye(self.system_dimension**2)*np.dot(n_vectors[n_index].T,matsubara_freqs))
        
        unit_vectors = utils.orthog_basis_set(self.system_dimension)
        
        # non-diagonal elements
        for i in range(self.system_dimension):
            temp_dm = n_vectors[n_index] + unit_vectors[i]
            if self.array_in_list(temp_dm, n_vectors):
                operators[self.index_of_array_in_list(temp_dm, n_vectors)] = self.phi2(i)
            temp_dm = n_vectors[n_index] - unit_vectors[i]
            if self.array_in_list(temp_dm, n_vectors):
                operators[self.index_of_array_in_list(temp_dm, n_vectors)] = self.theta2(i, n_vectors[n_index])
        
        return operators
    
    def generate_n_vectors(self):
        n_hierarchy = {0:[np.zeros((self.system_dimension,1))]}
        next_level = 1
        while next_level < self.truncation_level:
            n_hierarchy[next_level] = []
            for j in n_hierarchy[next_level-1]:
                for k in range(j.size):
                    j[k,0] += 1.
                    if not self.array_in_list(j, n_hierarchy[next_level]):
                        n_hierarchy[next_level].append(np.copy(j))
                    j[k,0] -= 1.
            next_level += 1
            
        n_vectors = []    
        for i in n_hierarchy.keys():
            for j in n_hierarchy[i]:
                n_vectors.append(j)
                
        return n_vectors
    
    # utility function for generate_n_vectors function
    def array_in_list(self, array, array_list):
        if not array_list:
            return False
        for l in array_list:
            if np.array_equal(array, l):
                return True
        return False
    
    def index_of_array_in_list(self, array, array_list):
        if not self.array_in_list(array, array_list):
            raise 'Array not in list!'
        for i,v in enumerate(array_list):
            if np.array_equal(array, v):
                return i

        
    # returns theta operator in Liouville space
    def theta(self, k, n):
        coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
        coupling_operator[k,k] = 1.
        return -0.5j * self.E_reorg * n[k] * (self.commutator_to_superoperator(coupling_operator) + (-1)**k * self.commutator_to_superoperator(coupling_operator, type='+'))
    
    def theta2(self, k, n):
        coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
        coupling_operator[k,k] = 1.
        return 1.j * self.E_reorg * n[k] * ((2.*0.695*self.temperature)*self.commutator_to_superoperator(coupling_operator) - 1.j * self.matsubara_freqs[k] * self.commutator_to_superoperator(coupling_operator, type='+'))
    
    # returns phi operator in Liouville space
    def phi(self, k):
        coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
        coupling_operator[k,k] = 1.
        return -1j * self.commutator_to_superoperator(coupling_operator)#self.system_coupling_operator)
    
    def phi2(self, k):
        coupling_operator = np.zeros((self.system_dimension, self.system_dimension))
        coupling_operator[k,k] = 1.
        return 1.j * self.commutator_to_superoperator(coupling_operator)#self.system_coupling_operator)
    
    def commutator_to_superoperator(self, operator, type='-'):
        if type == '-':
            z = -1.
        elif type == '+':
            z  = 1.
        else:
            raise 'Invalid commutator type defined in commutator_to_superoperator function'
        
        return  np.kron(operator, np.eye(self.system_dimension)) + z * np.kron(np.eye(self.system_dimension), operator)
    
    
    
    
    