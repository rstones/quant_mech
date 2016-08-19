'''
Created on 31 Jul 2013

@author: rstones

Module containing methods for numerical study of open quantum systems.
1) create Liouvillian superoperator from Hamiltonian and Lindblad operators
2) Markovian dissipator
3) partial trace
4) etc

'''
import numpy as np
import numpy.fft as fft
import scipy.integrate as integrate
import quant_mech.utils as utils
from datetime import datetime

# Function to construct Liouvillian super operator
# jump_operators should be list of tuples (first entry of tuple is lindblad operator, second is rate)
def super_operator(H, jump_operators):
    #print "Constructing super-operator..."
    I = np.eye(H.shape[0], H.shape[1], dtype='complex')
    L = -1.j * (np.kron(H, I) - np.kron(I, H))
    if jump_operators:
        L += incoherent_super_operator(jump_operators)
    return L

def incoherent_super_operator(jump_operators):
    jo_shape = jump_operators[0][0].shape
    L = np.zeros((jo_shape[0]**2, jo_shape[1]**2), dtype=complex)
    I = np.eye(jo_shape[0])
    for tup in jump_operators:
        A = tup[0]
        A_dagger = A.conj().T
        A_dagger_A = np.dot(A_dagger, A)
        L += tup[1] * (np.kron(A, A) - 0.5 * np.kron(A_dagger_A, I) - 0.5 * np.kron(I, A_dagger_A))
    return L

def markovian_dissipator(density_matrix, lindblad_operator, dissipation_rate):
    lindblad_operator_dagger = lindblad_operator.conj().T
    return dissipation_rate * (np.dot(lindblad_operator, np.dot(density_matrix, lindblad_operator_dagger)) - 0.5 * utils.anticommutator(np.dot(lindblad_operator_dagger, lindblad_operator), density_matrix))

# solve differential equation using 4th order Runge-Kutta method (used here for hierarchy code but should probably move to more appropriate module)
# func is a function that must take 2 parameters, step and state
def RK4(func, init_state, interval, step_size):
    state_evolution = []
    state = init_state
    for step in range(int(interval/step_size)):
        k1 = func(step, state)
        k2 = func(step+0.5*step_size, state + (0.5*step_size*k1))
        k3 = func(step+0.5*step_size, state + (0.5*step_size*k2))
        k4 = func(step+step_size, state + step_size*k3)
  
        state = state + (1./6.) * step_size * (k1 + 2.*k2 + 2.*k3 + k4)
        state_evolution.append(state)
        
    return np.array(state_evolution)

def overdamped_BO_spectral_density(freq, E_reorg, cutoff_freq):
    return (2. * E_reorg) * ((freq * cutoff_freq) / (freq**2 + cutoff_freq**2))

def underdamped_BO_spectral_density(freq, high_energy_mode_params):
    J = 0
    if high_energy_mode_params is not None and high_energy_mode_params.any():
        for tup in high_energy_mode_params:
            w_j = tup[0]
            S_j = tup[1]
            gamma_j = tup[2]
            lambda_j = S_j * w_j
            J += 2. * lambda_j * w_j**2 * ((freq * gamma_j) / ((w_j**2 - freq**2)**2 + freq**2 * gamma_j**2))
    return J

'''
Brownian oscillator spectral density
E_reorg and cutoff_freq and params for low energy part of spectral density
high_energy_params should be a list of tuples containing (freq(cm-1), huang-rhys factor, damping constant(cm-1)) 
'''    
def bo_spectral_density(freq, E_reorg, cutoff_freq, high_energy_params=None):
    return overdamped_BO_spectral_density(freq, E_reorg, cutoff_freq) + underdamped_BO_spectral_density(freq, high_energy_params)

'''
Bath dependent relaxation rate between excitons
spectral_density should be a function taking parameters reorganisation energy, transition freq and cutoff freq (in that order, see
bo_spectral_density function for example)
'''
def gamma(freq, cutoff_freq, spectral_density, E_reorg, temperature, high_energy_params=None):
    return 2. * spectral_density(np.abs(freq), E_reorg, cutoff_freq, high_energy_params) * np.abs(utils.planck_distribution(freq, temperature))

'''
Total relaxation rate between excitons 
'''
def Gamma(freq, cutoff_freq, spectral_density, E_reorg, temperature, ex1, ex2, high_energy_params=None):
    ex1_square = np.square(np.abs(ex1))
    ex2_square = np.square(np.abs(ex2))
    try:
        E_reorg_it = iter(E_reorg)
        cutoff_freq_it = iter(cutoff_freq)
        # E_reorg is a list of reorganisation energies for each site
        return np.sum(np.array([ex1_square[i]*ex2_square[i]*gamma(freq, cutoff_freq[i], spectral_density, E_reorg[i], temperature, high_energy_params) for i in range(ex1.size)]))
    except TypeError:
        # E_reorg is same reorganisation energy for each site
        return gamma(freq, cutoff_freq, spectral_density, E_reorg, temperature, high_energy_params) * np.dot(np.array([np.abs(i)**2 for i in ex1]), np.array([np.abs(i)**2 for i in ex2]))

'''
Vectorized function for Redfield population transfer rate calculation. exciton1, exciton2, site_reorg_energies and site_cutoff_freqs
should be provided as numpy arrays. Though if site_reorg_energies and site_cutoff_freqs are same for each site they can be scalars.
Tested against Chapter 5 of Ed's thesis

Currently uses only Drude spectral density, should probably write an OpenQuantumSystem class where you override hamiltonian and
spectral density function with which you can then calculate different quantities from that. 

Positive exciton_splitting for uphill rate, negative for downhill rate
'''
def redfield_population_transfer_rate(exciton_splitting, exciton1, exciton2, site_reorg_energies, site_cutoff_freqs, temperature):
    return 2. * np.sum(exciton1**2 * exciton2**2 * overdamped_BO_spectral_density(np.abs(exciton_splitting), site_reorg_energies, site_cutoff_freqs)) \
                    * np.abs(utils.planck_distribution(exciton_splitting, temperature))

'''
Calculates relaxation rates between exciton populations using Redfield theory

All these relaxation rate functions currently assume identical reorganisation energies for all sites, so need to generalise this.
'''
def exciton_relaxation_rates(site_hamiltonian, E_reorg, cutoff_freq, spectral_density, temperature, high_energy_params=None):
    # calculate sorted exciton energies and eigenvectors
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    rates = np.zeros((site_hamiltonian.shape[0], site_hamiltonian.shape[1]))
    
    for i in range(evals.size):
        for j in range(evals.size):
            E_i = evals[i]
            E_j = evals[j]
            w = E_j - E_i
            rates[i,j] = Gamma(w, cutoff_freq, spectral_density, E_reorg, temperature, evecs[i], evecs[j], high_energy_params) if w != 0 else 0
            
    return rates

'''
Calculates the redfield tensor for an excitonic system with identical Drude-Lorentz spectral density on each site.
Returns tensor in the flattened basis {rho_11, rho_12, rho_13,.... }
'''
def redfield_tensor_identical_drude(site_hamiltonian, reorg_energy, cutoff_freq, temperature):
    
    system_dimension = site_hamiltonian.shape[0]
    evalues,evectors = utils.sorted_eig(site_hamiltonian) # may need to get bare exciton energies here
    #evalues,evectors = np.linalg.eig(site_hamiltonian)
    
    # site_correlation function in frequency space for Drude-Lorentz spectral density
    def site_correlation_function(freq):
        if freq != 0 :
            return overdamped_BO_spectral_density(np.abs(freq), reorg_energy, cutoff_freq) * np.abs(utils.planck_distribution(freq, temperature))
        else:
            return 0
    
    # construct system part of system-bath coupling operators (single diagonal elements for this simple case)
    site_bath_coupling_matrices = np.zeros((system_dimension, system_dimension, system_dimension))
    for i in range(system_dimension):
        site_bath_coupling_matrices[i,i,i] = 1.
    
    # construct pairs of indices to iterate over when constructing the tensor
    from itertools import product
    indices = [(i,j) for i,j in product(range(system_dimension), range(system_dimension))]
        
    def Gamma(a, b, c, d):
        return np.real(np.sum([np.dot(evectors[a], np.dot(site_bath_coupling_matrices[n], evectors[b])) \
                                * np.dot(evectors[c], np.dot(site_bath_coupling_matrices[n], evectors[d])) for n in range(system_dimension)]) \
                                * site_correlation_function(evalues[c]-evalues[d]))
        
    def tensor_element(a, b, c, d):
        element = 0
        if a == c:
            for e in range(system_dimension):
                element += Gamma(b, e, e, d)
        if b == d:
            for e in range(system_dimension):
                element += Gamma(a, e, e ,c)
        element -= Gamma(c, a, b, d) + Gamma(d, b, a, c)
        return element
    
    result = np.zeros((system_dimension**2, system_dimension**2))
    for i,ab in enumerate(indices):
        for j,cd in enumerate(indices):
            result[i,j] = tensor_element(ab[0], ab[1], cd[0], cd[1])
    
    return result

# '''
# calculates a rate between two excitons using redfield theory
# '''
# def redfield_rate(freq, exciton_states, site_reorg_energies, site_cutoff_freqs, temperature, high_energy_mode_params=None):
#     pass
# 
# '''
# Calculates population to population Redfield rates for the case of a different spectral density on each site
# If high_energy_mode_params defined then spectral density will consist of overdamped part plus sum of underdamped modes
# otherwise just an overdamped part will be used.
# '''
# def redfield_rates(exciton_energies, exciton_states, site_reorg_energies, site_cutoff_freqs, temperature, high_energy_mode_params=None):
#     system_dim = exciton_energies.size
#     rates = np.zeros((system_dim, system_dim))
#     for i in range(system_dim):
#         for j in range(system_dim):
#             rates[i,j] = 


'''
Takes array for amplitudes of sites contributing to exciton and an array
of reorganisation energies for each site in same order as appears in exciton.
'''
def exciton_reorg_energy(exciton, E_reorgs):
    result = 0
    for i,v in enumerate(exciton):
        result += np.abs(v)**4 * E_reorgs[i]
    return result

'''
Calculates exciton overlap at a given site

excitons should be an iterable containing the vectors for 2 excitons
'''
def exciton_overlap_at_site(excitons, site_index):
    return excitons[0][site_index].conj() * excitons[1][site_index]

# excitons should be array containing 4 excitons (each exciton occupying a row)
def generalised_exciton_reorg_energy(excitons, site_reorg_energies):
    # check there are only 4 excitons
    if excitons.shape[0] != 4:
        raise Exception('More (or less) than 4 excitons have been passed to the generalised_exciton_reorg_energy function!')
     
    result = 0
    for i,v in enumerate(site_reorg_energies):
        result += exciton_overlap_at_site(excitons[:2], i) * exciton_overlap_at_site(excitons[2:], i) * v
    return result

'''
Leading coefficient of expansion of bath correlation function for Drude (over-damped Brownian oscillator) spectral density
(see section 1.1.3 of Ed's thesis)
'''
def OBO_correlation_function_leading_coeff(E_reorg, cutoff_freq, temp):
    return E_reorg*cutoff_freq * ((1. / (np.tan(cutoff_freq / (2.*utils.KELVIN_TO_WAVENUMS*temp)))) - 1.j)

'''
Calculates Matsubara freqs for exponential expansion of correlations functions
'''
def matsubara_freq(k, temperature):
    beta = 1./(utils.KELVIN_TO_WAVENUMS*temperature)
    return 2. * np.pi * (k / beta)

'''
Non-leading coefficients of exponential correlation function expansion for Drude (over-damped Brownian oscillator) spectral density
'''
def OBO_correlation_function_coeff(E_reorg, cutoff_freq, temperature, nu_k):
    beta = 1./(utils.KELVIN_TO_WAVENUMS*temperature)
    return (4.*E_reorg*cutoff_freq / beta) * (nu_k / (nu_k**2 - cutoff_freq**2))

'''
Term of line broadening function expansion for both simple Drude and high frequency spectral densities
'''
def line_broadening_function_term(coeff, freq, time):
    return (coeff / freq**2) * (np.exp(-freq*time) + (freq*time) - 1.)

'''
This function assumes a Drude spectral density of form:
2 \lambda \Omega * \omega / (\omega^2 + \Omega^2) 
where lambda is reorganisation energy and Omega is cutoff freq.
For num_expansion_terms = 0, the high temperature approximation is assumed and so only the leading
coefficient of the correlation function expanded in terms of exponentials is used.
The accuracy can be improved at lower temperatures by using more terms of the expansion.
'''
def site_line_broadening_function(time, E_reorg, cutoff_freq, temperature, num_expansion_terms=0):
    c0 = OBO_correlation_function_leading_coeff(E_reorg, cutoff_freq, temperature)
    result = (c0 / (cutoff_freq**2)) * (np.exp(-cutoff_freq*time) + (cutoff_freq*time) - 1.)
    if num_expansion_terms:
        for n in range(num_expansion_terms):
            nu_k = matsubara_freq(n+1, temperature)
            c_k = OBO_correlation_function_coeff(E_reorg, cutoff_freq, temperature, nu_k)
            result += line_broadening_function_term(c_k, nu_k, time)
    return result

'''
Calculates the leading term of the exponential correlation function expansion for
a high energy mode
Need to pass plus or minus one for the term argument to indicate whether you want to calculate the positive 
or negative term
lambda_j is reorganisation energy of mode
omega_j is frequency of mode
gamma_j is damping constant of mode
'''
def UBO_correlation_function_leading_coeff(omega_j, lambda_j, gamma_j, zeta_j, nu, temperature, term=1.):
#     zeta_j = np.sqrt(omega_j**2 - (gamma_j**2 / 4.))
#     nu = gamma_j/2. + term*1.j*zeta_j
    beta = 1./(utils.KELVIN_TO_WAVENUMS*temperature)
    return term*1.j* lambda_j * (omega_j**2/(2.*zeta_j)) * ((1./np.tan((nu*beta)/2)) - 1.j)

'''
Calculates coefficents of exponential correlation function expansion for high energy mode
lambda_j is reorganisation energy of mode
omega_j is frequency of mode
gamma_j is damping constant of mode
k is term in expansion
'''
def UBO_correlation_function_coeffs(omega_j, lambda_j, gamma_j, nu_k, temperature):
    beta = 1./(utils.KELVIN_TO_WAVENUMS*temperature)
    return -((4.*lambda_j*gamma_j*omega_j**2)/beta) * (nu_k / ((omega_j**2 + nu_k**2)**2 - (gamma_j**2 * nu_k**2)))


'''
Computes coefficients of exponential expansion of the correlation function for a site with a high energy structured spectral density
E_reorg and cutoff_freq are for the Drude low energy part of the spectral density
high_energy_params should be a list of tuples of form (omega_j, S_j (Huang-Rhys factor), gamma_j) 

Pass the resulting array to 
'''
def lbf_coeffs(E_reorg, cutoff_freq, temperature, high_energy_params, num_expansion_terms=0):
    coeffs = []
    #just_coeffs = []
    #matsubara_freqs = []
    
    # put Drude coeffs into list of expansion coeffs with corresponding matsubara freqs
    for n in range(num_expansion_terms):
        nu_k = matsubara_freq(n+1, temperature)
        coeff = OBO_correlation_function_coeff(E_reorg, cutoff_freq, temperature, nu_k)
        coeffs.append([coeff, nu_k])
        #just_coeffs.append(coeff)
        #matsubara_freqs.append(nu_k)
    if high_energy_params is not None:
        if isinstance(high_energy_params, list):
            high_energy_params = np.array(high_energy_params)
        if high_energy_params.any():
            for mode in high_energy_params:
                omega_j = mode[0]
                S_j = mode[1]
                gamma_j = mode[2]
                lambda_j = omega_j * S_j
                # add two leading terms of correlation function expansion
                zeta_j = np.sqrt(omega_j**2 - (gamma_j**2/4.))
                nu_plus = gamma_j/2. + 1.j*zeta_j
                coeff = UBO_correlation_function_leading_coeff(omega_j, lambda_j, gamma_j, zeta_j, nu_plus, temperature, term=1.)
                coeffs.append([coeff, nu_plus])
                #just_coeffs.append(coeff)
                #matsubara_freqs.append(nu_plus)
                nu_minus = gamma_j/2. - 1.j*zeta_j
                coeff = UBO_correlation_function_leading_coeff(omega_j, lambda_j, gamma_j, zeta_j, nu_minus, temperature, term=-1.)
                coeffs.append([coeff, nu_minus])
                #just_coeffs.append(coeff)
                #matsubara_freqs.append(nu_minus)
                
                # add further expansion terms to already saved expansion terms
                for n in range(num_expansion_terms):
                    nu_k = coeffs[n][1]
                    coeff = coeffs[n][0] + UBO_correlation_function_coeffs(omega_j, lambda_j, gamma_j, nu_k, temperature)
                    coeffs[n] = [coeff, nu_k]
                    #just_coeffs[n] = coeff
            
    # add leading term of Drude spectral density and corresponding cutoff freq
    coeff = OBO_correlation_function_leading_coeff(E_reorg, cutoff_freq, temperature)
    coeffs.append([coeff, cutoff_freq])
    #just_coeffs.append(coeff)
    #matsubara_freqs.append(cutoff_freq)
    
    return np.array(coeffs) # np.array(just_coeffs, dtype='complex'), np.array(matsubara_freqs, dtype='complex') #

'''
Calculate site line broadening function at every time point
time is array of times to calculate lbf at
coeffs is array of expansion coefficients for given site
'''
def site_lbf(time, coeffs):
    return np.array([np.sum([line_broadening_function_term(c[0], c[1], t) for c in coeffs], dtype='complex') for t in time])

def site_lbf2(t, coeffs):
    return np.sum([line_broadening_function_term(c[0], c[1], t) for c in coeffs])

'''
Calculates line broadening function for an exciton given the line broadening 
'''
def exciton_lbf(exciton, site_lbfs):
    num_excitons = site_lbfs.shape[0]
    num_time_pts = site_lbfs[0].shape[0]
    result = np.zeros(num_time_pts, dtype='complex')
    for i in range(num_excitons):
        result += np.abs(exciton[i])**4 * site_lbfs[i]
#     for i in range(num_time_pts):
#         result[i] = np.sum([np.abs(exciton[j])**4 * site_lbfs[j][i] for j in range(num_excitons)])
    #return np.sum([(np.abs(exciton[i])**4) * site_lbfs[i] for i in range(exciton.shape[0])])
    return result

'''
Calculates the line broadening function for mixing between excitons
'''
def generalised_exciton_lbf(excitons, site_lbfs):
    num_excitons = site_lbfs.shape[0]
    num_time_pts = site_lbfs[0].shape[0]
    result = np.zeros(num_time_pts, dtype='complex')
    for i in range(num_time_pts):
        #result[i] = np.sum([np.prod([np.abs(excitons[k][j]) for k in range(excitons.shape[0])]) * site_lbfs[j][i] for j in range(num_excitons)])
        result[i] = np.sum([exciton_overlap_at_site(excitons[:2], j) * exciton_overlap_at_site(excitons[2:], j) * site_lbfs[j][i] for j in range(num_excitons)])
    return result

def generalised_exciton_lbf2(t, excitons, lbf_coeffs):
    num_excitons = lbf_coeffs.shape[0]
    return np.sum([exciton_overlap_at_site(excitons[:2], j) * exciton_overlap_at_site(excitons[2:], j) * site_lbf2(t, lbf_coeffs[j]) for j in range(num_excitons)])

'''
lbf is already defined at each time step before passing to FFT function
'''
def absorption_line_shape_FFT(time, state_freq, lbf, lifetime=None):
    N =  time.shape[0]
    #integrand = np.array([np.exp(-1.j*state_freq*t -lbf[i] - ((t/lifetime) if lifetime else 0)) for i,t in enumerate(time)])
    integrand = np.exp((-1.j*state_freq - ((1./lifetime) if lifetime else 0)) * time - lbf)
    
    lineshape = time[-1] * fft.ifft(integrand, N)
    lineshape = np.append(lineshape[N/2:], lineshape[:N/2])
    
    return 2*np.real(lineshape)

'''
Returns absorption line shape as a function of time
'''
def absorption_line_shape(time, state_freq, lbf):
    return np.array([np.exp(-1.j * state_freq * t  - lbf[i]) for i,t in enumerate(time)])

def absorption_line_shape2(t, state_freq, excitons, lbf_coeffs):
    return np.exp(-1.j * state_freq * t  - generalised_exciton_lbf2(t, excitons, lbf_coeffs))

def fluorescence_line_shape_FFT(time, state_freq, E_reorg, lbf, lifetime=None):
    N =  time.shape[0]
    #integrand = np.array([np.exp((-1.j*state_freq*t) + (2.j*E_reorg*t) + (-lbf[i].conj()) - ((t/lifetime) if lifetime else 0)) for i,t in enumerate(time)])
    integrand = np.exp((-1.j*state_freq + 2.j*E_reorg - ((1./lifetime) if lifetime else 0)) * time - lbf.conj())
    
    lineshape = time[-1] * fft.ifft(integrand, N)
    lineshape = np.append(lineshape[N/2:], lineshape[:N/2])
    
    return 2*np.real(lineshape)

def FFT_freq(time):
    dt = time[1] - time[0]
    N =  time.shape[0]
    freq = 2.*np.pi*fft.fftfreq(N, dt)
    return np.append(freq[N/2:], freq[:N/2])

'''
Calculates fluoresence line shape as function of time
'''
def fluorescence_line_shape(time, state_freq, reorg_energy, lbf):
    return np.array([np.exp((-1.j*state_freq*t) + (2.j*reorg_energy*t) - lbf[i].conj()) for i,t in enumerate(time)])

def fluorescence_line_shape2(t, state_freq, reorg_energy, excitons, lbf_coeffs):
    return np.exp((-1.j*state_freq*t) + (2.j*reorg_energy*t) - generalised_exciton_lbf2(t, excitons, lbf_coeffs).conj())

'''
Calculates exciton mixing function used in modified Redfield theory
'''
def modified_redfield_mixing_function(line_broadening_functions, reorg_energies, time):    
    lbfs = line_broadening_functions
    lbf0 = utils.differentiate_function(utils.differentiate_function(lbfs[0], time)[:-5], time)
    lbf1 = utils.differentiate_function(lbfs[1], time)[:-5]
    lbf2 = utils.differentiate_function(lbfs[2], time)[:-5]
    lbf3 = line_broadening_functions[3][:-5]
    
    return np.array([(lbf0[i] - (lbf1[i] - lbf2[i] + 2.*1.j*reorg_energies[0]) ** 2) * np.exp(2. * (lbf3[i] + 1.j*reorg_energies[1]*t)) for i,t in enumerate(time[:-5])])



'''
Calculates line broadening function, its derivatives and total site reorganisation energy for modfied Redfield calculations
'''
def modified_redfield_params(time, reorg_energy, cutoff_freq, temperature, mode_params, num_expansion_terms=0):
    coeffs = lbf_coeffs(reorg_energy, cutoff_freq, temperature, mode_params, num_expansion_terms)
    total_site_reorg_energy = reorg_energy + (np.sum([mode[0]*mode[1] for mode in mode_params]) if mode_params is not None and mode_params.any() else 0)
    return site_lbf_ed(time, coeffs), site_lbf_dot_ed(time, coeffs), site_lbf_dot_dot_ed(time, coeffs), total_site_reorg_energy

'''
The definitive modified Redfield population transfer rate calculator. Assuming the spectral density on each site is identical...
'''
def modified_redfield_rates(evals, evecs, g_site, g_site_dot, g_site_dot_dot, total_site_reorg_energy, temperature, time):
    system_dim = evals.size
    rates = np.zeros((system_dim, system_dim))
    
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if j > i: # should only loop over rates for downhill energy transfer
                # get energy gap
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_j - E_i #E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = evecs[i]
                c_betas = evecs[j]
                # calculate sums of combined eigenvector amplitudes 
                A = np.sum(c_alphas**4) + np.sum(c_betas**4)
                B = np.sum(c_alphas**2 * c_betas**2)
                C = np.sum(c_alphas * c_betas**3)
                D = np.sum(c_alphas**3 * c_betas)
                # calculate integrand                
                integrand =  np.exp(1.j*omega_ij*time - (A - 2.*B) * (1.j*total_site_reorg_energy*time + g_site)) * \
                                    (B*g_site_dot_dot - ((C - D)*g_site_dot + 2.j*C*total_site_reorg_energy)**2)
                # perform integration
                rates[i,j] = 2. * integrate.simps(np.real(integrand), time)
                # calculate uphill rate using detailed balance
                rates[j,i] = np.exp(-omega_ij/(utils.KELVIN_TO_WAVENUMS*temperature)) * rates[i,j]

    return rates

'''
Modified Redfield rate calculator which can cope with different spectral densities on each site.
'''
def modified_redfield_rates_general(evals, evecs, g_sites, g_sites_dot, g_sites_dot_dot, site_reorg_energies, temperature, time):    
    system_dim = evals.size
    rates = np.zeros((system_dim, system_dim))
    
    # check evals are in order lowest to highest in energy
    previous = -np.inf
    reorder = False
    for E in evals:
        if previous > E:
            reorder = True
            break
        previous = E
    
    # if not in order then reorder and reassign evals and evecs
    if reorder:
        print 'Reordering system eigenvalues and eigenvectors'
        evals, evecs = utils.sort_evals_evecs(evals, evecs)
        
    site_reorg_energies = np.array([site_reorg_energies]).T
        
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if j > i: # should only loop over rates for downhill energy transfer
                # get energy gap
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_j - E_i #E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = np.array([evecs[i]]) # add a second dimension (of 1) to evecs so we can do transpose below
                c_betas = np.array([evecs[j]])
                A = c_alphas.T**4
                B = c_betas.T**4
                C = c_alphas.T**2 * c_betas.T**2
                D = c_alphas.T * c_betas.T**3
                E = c_alphas.T**3 * c_betas.T
                
                # calculate reorg energies and line broadening functions
                lambda_aaaa = np.sum(A * site_reorg_energies)
                lambda_bbbb = np.sum(B * site_reorg_energies)
                g_aaaa = np.sum(A * g_sites, axis=0)
                g_bbbb = np.sum(B * g_sites, axis=0)
                g_bbaa = np.sum(C * g_sites, axis=0)
                lambda_bbaa = np.sum(C * site_reorg_energies)
                g_dot_dot_baba = np.sum(C * g_sites_dot_dot, axis=0)
                g_dot_babb = np.sum(D * g_sites_dot, axis=0)
                g_dot_baaa = np.sum(E * g_sites_dot, axis=0)
                lambda_babb = np.sum(D * site_reorg_energies)
                # calculate integrand
                integrand = np.exp(1.j*omega_ij*time - 1.j*(lambda_aaaa + lambda_bbbb - 2.*lambda_bbaa)*time - g_aaaa - g_bbbb + 2.*g_bbaa) \
                                    * (g_dot_dot_baba - (g_dot_babb - g_dot_baaa + 2.j*lambda_babb)**2)
                # perform integration
                rates[i,j] = 2. * integrate.simps(np.real(integrand), time)
                # calculate uphill rate using detailed balance
                rates[j,i] = np.exp(-omega_ij/(utils.KELVIN_TO_WAVENUMS*temperature)) * rates[i,j]

    return rates, evals, evecs # return sorted evals and evecs so we know the labelling of the rates array

'''
Modified Redfield rate calculator which can cope with different spectral densities on each site and arbitrary ordering of exciton energies.
'''
def modified_redfield_rates_general_unordered(evals, evecs, g_sites, g_sites_dot, g_sites_dot_dot, site_reorg_energies, temperature, time):    
    system_dim = evals.size
    rates = np.zeros((system_dim, system_dim))
    
    '''
    This line is commented out for a test
    '''
    site_reorg_energies = np.array([site_reorg_energies]).T
        
    for i in range(system_dim):
        for j in range(system_dim):
            if i != j:
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_j - E_i
                if omega_ij > 0:
                    # calculate overlaps (c_alpha and c_beta's)
                    c_alphas = np.array([evecs[i]]) # add a second dimension (of 1) to evecs so we can do transpose below
                    c_betas = np.array([evecs[j]])
                    
                    '''
                    Temporarily including this line here to test code in PSIIRCPhotcellModel 
                    '''
                    #site_reorg_energies = np.array([site_reorg_energies]).T
                    
                    A = c_alphas.T**4
                    B = c_betas.T**4
                    C = c_alphas.T**2 * c_betas.T**2
                    D = c_alphas.T * c_betas.T**3
                    E = c_alphas.T**3 * c_betas.T
                    # calculate reorg energies and line broadening functions
                    lambda_aaaa = np.sum(A * site_reorg_energies)
                    lambda_bbbb = np.sum(B * site_reorg_energies)
                    g_aaaa = np.sum(A * g_sites, axis=0)
                    g_bbbb = np.sum(B * g_sites, axis=0)
                    g_bbaa = np.sum(C * g_sites, axis=0)
                    lambda_bbaa = np.sum(C * site_reorg_energies)
                    g_dot_dot_baba = np.sum(C * g_sites_dot_dot, axis=0)
                    g_dot_babb = np.sum(D * g_sites_dot, axis=0)
                    g_dot_baaa = np.sum(E * g_sites_dot, axis=0)
                    lambda_babb = np.sum(D * site_reorg_energies)
                    # calculate integrand
                    integrand = np.exp(1.j*omega_ij*time - 1.j*(lambda_aaaa + lambda_bbbb - 2.*lambda_bbaa)*time - g_aaaa - g_bbbb + 2.*g_bbaa) \
                                        * (g_dot_dot_baba - (g_dot_babb - g_dot_baaa + 2.j*lambda_babb)**2)
                    # perform integration
                    rates[i,j] = 2. * integrate.simps(np.real(integrand), time)
                    # calculate uphill rate using detailed balance
                    rates[j,i] = np.exp(-omega_ij/(utils.KELVIN_TO_WAVENUMS*temperature)) * rates[i,j]

    return rates

# site line broadening function, check against other function
# def site_lbf_ed(time, coeffs):
#     return np.array([np.sum([(coeff[0] / coeff[1]**2) * (np.exp(-coeff[1]*t) + (coeff[1]*t) - 1.) for coeff in coeffs]) for t in time], dtype='complex')
    #return (coeffs / matsubara_freqs**2) * (np.exp(-matsubara_freqs*time) + (matsubara_freqs*time) - 1.)
    
# uses vectorization with time array    
def site_lbf_ed(time, coeffs):
    return np.squeeze(np.array([np.sum([(coeff[0] / coeff[1]**2) * (np.exp(-coeff[1]*time) + (coeff[1]*time) - 1.) for coeff in coeffs], axis=0)], dtype='complex'))
    
# first differential of site line broadening function, check against numerical differentiation
# def site_lbf_dot_ed(time, coeffs):
#     return np.array([np.sum([(coeff[0] / coeff[1]) * (1. - np.exp(-coeff[1]*t)) for coeff in coeffs]) for t in time], dtype='complex')
    #return (coeffs / matsubara_freqs) * (1. - np.exp(-matsubara_freqs*time))
    
def site_lbf_dot_ed(time, coeffs):
    return np.array(np.sum([(coeff[0] / coeff[1]) * (1. - np.exp(-coeff[1]*time)) for coeff in coeffs], axis=0), dtype='complex')

# second differential of site line broadening function, check against numerical differentiation
# def site_lbf_dot_dot_ed(time, coeffs):
#     return np.array([np.sum([coeff[0] * np.exp(-coeff[1]*t) for coeff in coeffs]) for t in time], dtype='complex')
    #return coeffs * np.exp(-matsubara_freqs*time)
    
def site_lbf_dot_dot_ed(time, coeffs):
    return np.array(np.sum([coeff[0] * np.exp(-coeff[1]*time) for coeff in coeffs], axis=0), dtype='complex')


'''
Calculation of modified Redfield theory rates in same way as Ed has done in Mathematica for simple case of no structured environment and identical 
spectral density for each site.
'''
def MRT_rate_ed(site_hamiltonian, site_reorg_energy, cutoff_freq, temperature, high_energy_mode_params, num_expansion_terms=0, time_interval=0.5):
    time = np.linspace(0,time_interval, int(time_interval*2000.))
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    
    coeffs = lbf_coeffs(site_reorg_energy, cutoff_freq, temperature, high_energy_mode_params, num_expansion_terms)
    g_site = site_lbf_ed(time, coeffs)
    g_site_dot = site_lbf_dot_ed(time, coeffs)
    g_site_dot_dot = site_lbf_dot_dot_ed(time, coeffs)
    
    if high_energy_mode_params is not None and high_energy_mode_params.any():
        site_reorg_energy += np.sum([mode[0]*mode[1] for mode in high_energy_mode_params])
    
    system_dim = site_hamiltonian.shape[0]
    rates = np.zeros((system_dim, system_dim))
    integrands = np.zeros((system_dim, system_dim, time.size), dtype='complex')
    
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if i != j:
                # get energy gap
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = evecs[i]
                c_betas = evecs[j]
                # calculate integrand                
                integrand = np.array([np.exp(1.j*omega_ij*t - (np.sum(c_alphas**4) + np.sum(c_betas**4))*(1.j*site_reorg_energy*t + g_site[k]) + 
                                      2. * np.sum(c_alphas**2 * c_betas**2) * (g_site[k] + 1.j*site_reorg_energy*t)) *
                                      ((np.sum(c_alphas**2 * c_betas**2)*g_site_dot_dot[k]) - 
                                       ((np.sum(c_alphas * c_betas**3) - np.sum(c_alphas**3 * c_betas))*g_site_dot[k] + 2.j*np.sum(c_betas**3 * c_alphas)*site_reorg_energy)**2) for k,t in enumerate(time)])
                # perform integration
                rates[i,j] = 2.* integrate.simps(np.real(integrand), time)
                integrands[i,j] = integrand

    return rates#, integrands, time

'''
Calculates modified Redfield rates for PE545 complex which requires inclusion of 2 over-damped Brownian oscillator spectral densities
'''
def MRT_rate_PE545(site_hamiltonian, site_reorg_energy1, cutoff_freq1, site_reorg_energy2, cutoff_freq2, temperature, high_energy_mode_params, num_expansion_terms=0, time_interval=0.5):
    time = np.linspace(0,time_interval, int(time_interval*16000.))
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    
    # correlation function coefficients for first over-damped oscillator and high energy modes
    coeffs = lbf_coeffs(site_reorg_energy1, cutoff_freq1, temperature, high_energy_mode_params, num_expansion_terms)
    # add correlation function coefficients for second over-damped oscillator
    coeffs = np.concatenate((coeffs, lbf_coeffs(site_reorg_energy2, cutoff_freq2, temperature, None, num_expansion_terms)))
    
    g_site = site_lbf_ed(time, coeffs)
    g_site_dot = site_lbf_dot_ed(time, coeffs)
    g_site_dot_dot = site_lbf_dot_dot_ed(time, coeffs)
    
    total_site_reorg_energy = site_reorg_energy1 + site_reorg_energy2
    if high_energy_mode_params is not None and high_energy_mode_params.any():
        total_site_reorg_energy += np.sum([mode[0]*mode[1] for mode in high_energy_mode_params])
    
    system_dim = site_hamiltonian.shape[0]
    rates = np.zeros((system_dim, system_dim))
    integrands = np.zeros((system_dim, system_dim, time.size), dtype='complex')
    
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if i != j:
                # get energy gap
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_j - E_i #E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = evecs[i]
                c_betas = evecs[j]
                # calculate integrand                
                integrand = np.array([np.exp(1.j*omega_ij*t - (np.sum(c_alphas**4) + np.sum(c_betas**4))*(1.j*total_site_reorg_energy*t + g_site[k]) + 
                                      2. * np.sum(c_alphas**2 * c_betas**2) * (g_site[k] + 1.j*total_site_reorg_energy*t)) *
                                      ((np.sum(c_alphas**2 * c_betas**2)*g_site_dot_dot[k]) - 
                                       ((np.sum(c_alphas * c_betas**3) - np.sum(c_alphas**3 * c_betas))*g_site_dot[k] + 2.j*np.sum(c_betas**3 * c_alphas)*total_site_reorg_energy)**2) for k,t in enumerate(time)])
                # perform integration
                rates[i,j] = 2.* integrate.simps(np.real(integrand), time)
                integrands[i,j] = integrand

    return rates#, integrands, time

def MRT_rate_PE545_quick(exciton_energies, eigenvectors, g_site, g_site_dot, g_site_dot_dot, total_site_reorg_energy, temperature, time):
    system_dim = exciton_energies.size
    rates = np.zeros((system_dim, system_dim))
    
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if j > i: # should only loop over rates for downhill energy transfer
                # get energy gap
                E_i = exciton_energies[i]
                E_j = exciton_energies[j]
                omega_ij = E_j - E_i #E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = eigenvectors[i]
                c_betas = eigenvectors[j]
                # calculate sums of combined eigenvector amplitudes 
                A = np.sum(c_alphas**4) + np.sum(c_betas**4)
                B = np.sum(c_alphas**2 * c_betas**2)
                C = np.sum(c_alphas * c_betas**3)
                D = np.sum(c_alphas**3 * c_betas)
                
                # calculate integrand             
#                 integrand = np.array([np.exp(1.j*omega_ij*t - (np.sum(c_alphas**4) + np.sum(c_betas**4))*(1.j*total_site_reorg_energy*t + g_site[k]) + 
#                                       2. * np.sum(c_alphas**2 * c_betas**2) * (g_site[k] + 1.j*total_site_reorg_energy*t)) *
#                                       ((np.sum(c_alphas**2 * c_betas**2)*g_site_dot_dot[k]) - 
#                                        ((np.sum(c_alphas * c_betas**3) - np.sum(c_alphas**3 * c_betas))*g_site_dot[k] + 2.j*np.sum(c_betas**3 * c_alphas)*total_site_reorg_energy)**2) for k,t in enumerate(time)])

#                 integrand = np.zeros(time.size, dtype='complex')
#                 for k,t in enumerate(time):
#                     integrand[k] =  np.exp(1.j*omega_ij*t - (A - 2.*B) * (1.j*total_site_reorg_energy*t + g_site[k])) * \
#                                     (B*g_site_dot_dot[k] - ((C - D)*g_site_dot[k] + 2.j*C*total_site_reorg_energy)**2)

                integrand =  np.exp(1.j*omega_ij*time - (A - 2.*B) * (1.j*total_site_reorg_energy*time + g_site)) * \
                                    (B*g_site_dot_dot - ((C - D)*g_site_dot + 2.j*C*total_site_reorg_energy)**2)
                
                # perform integration
                rates[i,j] = 2. * integrate.simps(np.real(integrand), time)
                # calculate uphill rate using detailed balance
                rates[j,i] = np.exp(-omega_ij/(utils.KELVIN_TO_WAVENUMS*temperature)) * rates[i,j]

    return rates

'''
Calculates modified Redfield rates with different site reorganisation energies of the underdamped Brownian oscillator at each site
but with identical cutoff frequencies and high energy mode parameters
'''
def MRT_rates(site_hamiltonian, site_reorg_energies, cutoff_freq, temperature, high_energy_mode_params, num_expansion_terms=0, time_interval=0.5):
    time = np.linspace(0,time_interval, int(time_interval*2000.))
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    system_dim = site_hamiltonian.shape[0]
    
    coeffs = np.array([lbf_coeffs(site_reorg_energies[i], cutoff_freq, temperature, high_energy_mode_params, num_expansion_terms) for i in range(system_dim)])
    g_site = np.array([site_lbf_ed(time, coeffs[i]) for i in range(system_dim)])
    g_site_dot = np.array([site_lbf_dot_ed(time, coeffs[i]) for i in range(system_dim)])
    g_site_dot_dot = np.array([site_lbf_dot_dot_ed(time, coeffs[i]) for i in range(system_dim)])
    
    if high_energy_mode_params is not None and high_energy_mode_params.any():
        site_reorg_energies += np.sum([mode[0]*mode[1] for mode in high_energy_mode_params])
    
    rates = np.zeros((system_dim, system_dim))
    integrands = np.zeros((system_dim, system_dim, time.size), dtype='complex')
    
    # excitons are labelled from lowest in energy to highest
    for i in range(system_dim):
        for j in range(system_dim):
            if i != j:
                # get energy gap
                E_i = evals[i]
                E_j = evals[j]
                omega_ij = E_i - E_j if E_i > E_j else E_j - E_i
                # calculate overlaps (c_alpha and c_beta's)
                c_alphas = evecs[i]
                c_betas = evecs[j]
                # calculate integrand
                integrand = np.array([np.exp(1.j*omega_ij*t - np.sum((c_alphas**4 + c_betas**4)*(1.j*site_reorg_energies*t + g_site.T[k])) + 
                                      2. * np.sum((c_alphas**2 * c_betas**2) * (g_site.T[k] + 1.j*site_reorg_energies*t))) *
                                      (np.sum(c_alphas**2 * c_betas**2 * g_site_dot_dot.T[k]) - 
                                       (np.sum((c_alphas * c_betas**3 - c_alphas**3 * c_betas)*g_site_dot.T[k]) + 2.j*np.sum(c_betas**3 * c_alphas * site_reorg_energies))**2) for k,t in enumerate(time)])
                # perform integration
                rates[i,j] = 2.* integrate.simps(np.real(integrand), time)
                integrands[i,j] = integrand

    return rates#, integrands, time

###########################################################################
#
# Functions to calculate quantities related to Forster theory
#
###########################################################################

'''
State energies E1 and E2 should not include reorganisation shift (need to clarify whether reorganisation energy should be removed before or
after diagonalisation of Hamiltonian)

Whether state 1 or 2 refers to absorbing or fluorescing state depends on whether forward or backward rate is being calculated 
'''
def forster_rate(E1, E2, E_reorg1, E_reorg2, line_broadening1, line_broadening2, lifetime1, lifetime2, state1, state2, hamiltonian, time):
    if lifetime1 and lifetime2 != 0:
        integrand = np.exp(1.j*(E1-E2)*time - 1.j*(E_reorg1+E_reorg2)*time - line_broadening1 - line_broadening2 - time*(1./lifetime1 + 1./lifetime2))
        #integrand = np.array([np.exp(1.j*(E1-E2)*t - 1.j*(E_reorg1+E_reorg2)*t - line_broadening1[i] - line_broadening2[i] - t*(1./lifetime1 + 1./lifetime2)) for i,t in enumerate(time)], dtype='complex')
    else:
        integrand = np.exp(1.j*(E1-E2)*time - 1.j*(E_reorg1+E_reorg2)*time - line_broadening1 - line_broadening2)
        #integrand = np.array([np.exp(1.j*(E1-E2)*t - 1.j*(E_reorg1+E_reorg2)*t - line_broadening1[i] - line_broadening2[i]) for i,t in enumerate(time)], dtype='complex')
    overlap = 2. * integrate.simps(np.real(integrand), time)
    #print overlap
    transition_matrix_element = (np.abs(np.dot(state2, np.dot(hamiltonian, state1)))**2)
    #print transition_matrix_element
    return transition_matrix_element * overlap

def exciton_lifetimes(hamiltonian, site_reorg_energies, site_cutoff_freqs, temperature, high_energy_modes=None):
    redfield_rates = exciton_relaxation_rates(hamiltonian, site_reorg_energies, site_cutoff_freqs, bo_spectral_density, temperature, high_energy_modes)
    return np.array([2./(np.sum(redfield_rates[i])) for i in range(hamiltonian.shape[0])]) 

'''
Calculates rate for transfer between two clusters of chromophores. The chromophores within each cluster are strongly coupled forming excitons
while coupling between the clusters is weak relative to the environmental coupling.
The Hamiltonian must be for all chromophores with the matrix elements for each cluster grouped together. It is assumed that the site energies
will include the reorganisation energy shift. The reorg energy is removed as the forster_rate function expects bare exciton energies.
'''
def generalised_forster_rate(hamiltonian, cluster1_dim, cluster2_dim, total_site_reorg_energies, site_cutoff_freqs, site_lbfs, time, temperature, high_energy_modes=None):
    sys_dim = cluster1_dim+cluster2_dim
    cluster1_hamiltonian = hamiltonian[:cluster1_dim, :cluster1_dim]# - np.diag(site_reorg_energies[:cluster1_dim])
    cluster2_hamiltonian = hamiltonian[cluster1_dim:, cluster1_dim:]# - np.diag(site_reorg_energies[cluster1_dim:])
    
    # diagonalise individual clusters
    cluster1_evals, cluster1_evecs = utils.sorted_eig(cluster1_hamiltonian)
    cluster2_evals, cluster2_evecs = utils.sorted_eig(cluster2_hamiltonian)
    
    # calculate inter-cluster exciton couplings
    couplings = hamiltonian[:cluster1_dim,cluster1_dim:]
    couplings_matrix = np.dot(cluster1_evecs.conj(), np.dot(couplings, cluster2_evecs.T))
    
    # construct partially diagonalised Hamiltonian
    exciton_hamiltonian = np.asarray(np.bmat([[np.diag(cluster1_evals), couplings_matrix], [couplings_matrix.conj().T, np.diag(cluster2_evals)]]))
    
    # calculate exciton reorg energies
    cluster1_exciton_reorg_energies = np.array([exciton_reorg_energy(cluster1_evecs[i], total_site_reorg_energies[:cluster1_dim]) for i in range(cluster1_dim)])
    cluster2_exciton_reorg_energies = np.array([exciton_reorg_energy(cluster2_evecs[i], total_site_reorg_energies[cluster1_dim:]) for i in range(cluster2_dim)])
    
    # calculate exciton line broadening functions
    cluster1_lbfs = np.array([exciton_lbf(cluster1_evecs[i], site_lbfs[:cluster1_dim]) for i in range(cluster1_dim)])
    cluster2_lbfs = np.array([exciton_lbf(cluster2_evecs[i], site_lbfs[cluster1_dim:]) for i in range(cluster2_dim)])
    
    # calculate lifetimes
    OBO_reorg_energies = total_site_reorg_energies - np.sum([s[0]*s[1] for s in high_energy_modes])
    cluster1_lifetimes = exciton_lifetimes(cluster1_hamiltonian, OBO_reorg_energies[:cluster1_dim], site_cutoff_freqs[:cluster1_dim], temperature, high_energy_modes=high_energy_modes)
    cluster2_lifetimes = exciton_lifetimes(cluster2_hamiltonian, OBO_reorg_energies[cluster1_dim:], site_cutoff_freqs[cluster1_dim:], temperature, high_energy_modes=high_energy_modes)
    
    bare_cluster1_evals = cluster1_evals - cluster1_exciton_reorg_energies
    bare_cluster2_evals = cluster2_evals - cluster2_exciton_reorg_energies
    
    # calculate individual Forster rates
    forster_rates = np.zeros((cluster1_dim, cluster2_dim))
    for i in range(cluster1_dim):
        for j in range(cluster2_dim):
            cluster1_state = np.zeros(sys_dim)
            cluster1_state[i] = 1.
            cluster2_state = np.zeros(sys_dim)
            cluster2_state[cluster1_dim+j] = 1.
            forster_rates[i,j] = forster_rate(bare_cluster1_evals[i], bare_cluster2_evals[j], cluster1_exciton_reorg_energies[i], \
                                                  cluster2_exciton_reorg_energies[j], cluster1_lbfs[i], cluster2_lbfs[j], cluster1_lifetimes[i], cluster2_lifetimes[j], \
                                                  cluster1_state, cluster2_state, exciton_hamiltonian, time)
    
    # calculate generalised Forster rate
    B800_thermal_state = utils.general_thermal_state(np.diag(cluster1_evals), temperature)
    result = np.dot(np.diag(B800_thermal_state), np.sum(forster_rates, axis=1))
    return result

def marcus_rate(coupling, temperature, reorg_energy, driving_force):
    k_BT_wavenums = utils.KELVIN_TO_WAVENUMS * temperature
    return np.abs(coupling)**2 * np.sqrt(np.pi/(k_BT_wavenums*reorg_energy)) \
                    * np.exp(-(driving_force - reorg_energy)**2/(4.*reorg_energy*k_BT_wavenums))
                    
def check_detailed_balance(forward_rate, backward_rate, energy_gap, temperature, accuracy=0.001):
    return np.abs(forward_rate - np.exp(energy_gap/(utils.KELVIN_TO_WAVENUMS*temperature))*backward_rate) < accuracy

    