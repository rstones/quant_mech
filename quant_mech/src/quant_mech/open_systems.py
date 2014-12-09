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

# Function to construct Liouvillian super operator
# jump_operators should be list of tuples (first entry of tuple is lindblad operator, second is rate)
def super_operator(H, jump_operators):
    print "Constructing super-operator..."
    I = np.eye(H.shape[0], H.shape[1])
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
        
    return state_evolution

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
        iterator = iter(E_reorg)
        # E_reorg is a list of reorganisation energies for each site
        return np.sum(np.array([ex1_square[i]*ex2_square[i]*gamma(freq, cutoff_freq, spectral_density, E_reorg[i], temperature, high_energy_params) for i in range(ex1.size)]))
    except TypeError:
        # E_reorg is same reorganisation energy for each site
        return gamma(freq, cutoff_freq, spectral_density, E_reorg, temperature, high_energy_params) * np.dot(np.array([np.abs(i)**2 for i in ex1]), np.array([np.abs(i)**2 for i in ex2]))

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
    
    # put Drude coeffs into list of expansion coeffs with corresponding matsubara freqs
    for n in range(num_expansion_terms):
        nu_k = matsubara_freq(n+1, temperature)
        coeffs.append((OBO_correlation_function_coeff(E_reorg, cutoff_freq, temperature, nu_k), nu_k))
    
    if high_energy_params is not None and high_energy_params.any():
        for mode in high_energy_params:
            omega_j = mode[0]
            S_j = mode[1]
            gamma_j = mode[2]
            lambda_j = omega_j * S_j
            # add two leading terms of correlation function expansion
            zeta_j = np.sqrt(omega_j**2 - (gamma_j**2/4.))
            nu_plus = gamma_j/2. + 1.j*zeta_j
            coeffs.append((UBO_correlation_function_leading_coeff(omega_j, lambda_j, gamma_j, zeta_j, nu_plus, temperature, term=1.), nu_plus))
            nu_minus = gamma_j/2. - 1.j*zeta_j
            coeffs.append((UBO_correlation_function_leading_coeff(omega_j, lambda_j, gamma_j, zeta_j, nu_minus, temperature, term=-1.), nu_minus))
            
            # add further expansion terms to already saved expansion terms
            for n in range(num_expansion_terms):
                nu_k = coeffs[n][1]
                coeff = coeffs[n][0] + UBO_correlation_function_coeffs(omega_j, lambda_j, gamma_j, nu_k, temperature)
                coeffs[n] = (coeff, nu_k)
            
    # add leading term of Drude spectral density and corresponding cutoff freq
    coeffs.append((OBO_correlation_function_leading_coeff(E_reorg, cutoff_freq, temperature), cutoff_freq))
    return np.array(coeffs)

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
    for i in range(num_time_pts):
        result[i] = np.sum([np.abs(exciton[j])**4 * site_lbfs[j][i] for j in range(num_excitons)])
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
    integrand = np.array([np.exp(-1.j*state_freq*t -lbf[i] - (t/lifetime) if lifetime else 0) for i,t in enumerate(time)])
    
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
    integrand = np.array([np.exp((-1.j*state_freq*t) + (2.j*E_reorg*t) + (-lbf[i].conj()) - (t/lifetime) if lifetime else 0) for i,t in enumerate(time)])
    #integrand = np.array([np.exp((-1.j*state_freq*t) + (2.j*E_reorg*t) + (-lbf(t, *lbf_args)) + (-t/lifetime)) for t in time])
    
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
Uses quad function (this is not that reliable as it is not made to integrate a function sampled at given time intervals, use
modified_redfield_integration_simps instead)
'''
def modified_redfield_integration(abs_line_shape, fl_line_shape, mixing_function, time):
    sample_gap = time[1] - time[0]

    def integrand(t):
        time_index = 0
        for i,v in enumerate(time):
            if (t < v + sample_gap) and (t > v - sample_gap):
                time_index = i
                break
        return np.real(abs_line_shape[time_index] * fl_line_shape[time_index] * mixing_function[time_index])
     
    return 2. * integrate.quad(integrand, 0, time[-1])[0]

def modified_redfield_integration_simps(abs_line_shape, fl_line_shape, mixing_function, time):
    integrand = abs_line_shape * fl_line_shape * mixing_function
    return 2. * integrate.simps(np.real(integrand), time)

def modified_redfield_integration_trapz(abs_line_shape, fl_line_shape, mixing_function, time):
    return 2. * np.trapz(np.real(abs_line_shape * fl_line_shape * mixing_function), time)

def modified_redfield_integration_freq_domain(abs_line_shape, fl_line_shape, mixing_function, time):
    import matplotlib.pyplot as plt
#     plt.loglog(time, np.abs(abs_line_shape), label='abs')
#     plt.loglog(time, np.abs(fl_line_shape), label='fl')
#     plt.loglog(time, np.abs(mixing_function), label='mix')
#     plt.legend()
#     plt.show()
    
    N =  time.shape[0]
    abs_freq = time[-1] * fft.ifft(abs_line_shape, N)
    abs_freq = np.append(abs_freq[N/2:], abs_freq[:N/2])
    fl_freq = time[-1] * fft.ifft(fl_line_shape, N)
    fl_freq = np.append(fl_freq[N/2:], fl_freq[:N/2])
    mix_freq = time[-1] * fft.ifft(mixing_function, N)
    mix_freq = np.append(mix_freq[N/2:], mix_freq[:N/2])
    freq = FFT_freq(time)
    num_freq_samples = freq.shape[0]
    freq_min = freq[0]
    freq_max = freq[-1]
    freq_gap = np.abs(freq[0]) - np.abs(freq[1])
    
    
#     plt.plot(freq, abs_freq, label='abs')
#     plt.plot(freq, fl_freq, label='fl')
#     #plt.plot(freq, mix_freq, label="mix")
#     plt.legend()
#     plt.show()
    
    # find midpoint of absorption and fluorescence peaks, centre freq range around this point
    freq_abs_max = freq[np.abs(abs_freq - np.amax(abs_freq)).argmin()]
    freq_fl_max = freq[np.abs(fl_freq - np.amax(fl_freq)).argmin()]
    peak_midpoint = (freq_abs_max + freq_fl_max) / 2.
    print "peak midpoint: " + str(peak_midpoint)
    # find index of midpoint freq
    midpoint_index = 0
    for i,w in enumerate(freq):
        if w - freq_gap < peak_midpoint < w + freq_gap:
            midpoint_index = i
            break
    print "midpoint index: " + str(midpoint_index)
    # freq range for abs and fl functions will then be number of samples from midpoint to nearest end of range
    abs_fl_freq_range =  0
    if midpoint_index >= num_freq_samples/2:
        samples_til_end = num_freq_samples - midpoint_index
        abs_fl_freq_range = samples_til_end if samples_til_end % 2 == 0 else samples_til_end - 1
    print "abs fl freq range: " + str(abs_fl_freq_range)
    
    # restrict freq samples, abs and fl functions to symmetric freq interval around midpoint of abs and fl peaks
    freq_interval = freq[midpoint_index-abs_fl_freq_range/2:midpoint_index+abs_fl_freq_range/2]
    abs_interval = abs_freq[midpoint_index-abs_fl_freq_range/2:midpoint_index+abs_fl_freq_range/2]
    fl_interval = fl_freq[midpoint_index-abs_fl_freq_range/2:midpoint_index+abs_fl_freq_range/2]
    print "fl interval: " + str(fl_interval.shape)
    
    # abs and fl function need to be defined in 2D space (dimensions w1 and w2, where fl is a function of w1 and abs a function of w2 and both 
    # constant in the other argument)
    num_freq_interval_samples = freq_interval.shape[0]
    fl_grid = np.vstack((fl_interval for i in range(num_freq_interval_samples)))
    abs_grid = np.vstack((abs_interval for i in range(num_freq_interval_samples))).T
    
    # now define mixing function over range twice as big as that for abs and fl functions (since its argument is a frequency difference)
    mix_grid = np.empty((num_freq_interval_samples, num_freq_interval_samples))
    for i,w1 in enumerate(freq_interval):
        for j,w2 in enumerate(freq_interval):
            w_diff = w1 - w2
            mix_grid[i,j] = mix_freq[np.abs(freq_interval-w_diff).argmin()] # this indexes mix_freq at index of element of freq nearest to w_diff  

    return 1./(np.pi**2) * integrate.simps(int.simps(fl_grid * abs_grid * mix_grid))

'''
Calculates exciton population transfer rates using modified Redfield theory

Initially will assume over-damped Brownian oscillator spectral density for low energy phonons and under-damped Brownian
oscillator spectral density for discrete high energy modes.
'''
def modified_redfield_relaxation_rates(site_hamiltonian, site_reorg_energies, cutoff_freq, high_energy_mode_params, temperature, num_expansion_terms=0, time_interval=0.5):
    time = np.linspace(0, time_interval, 16005)
    num_sites = site_hamiltonian.shape[0]
    
    # diagonalise site Hamiltonian to get exciton energies and eigenvectors
    evals, evecs = utils.sorted_eig(site_hamiltonian)
    
    # calculate site line broadening functions
    site_lbfs = []
    for reorg_energy in site_reorg_energies:
        site_lbfs.append(site_lbf(time, lbf_coeffs(reorg_energy, cutoff_freq, temperature, high_energy_mode_params, num_expansion_terms)))
    site_lbfs = np.array(site_lbfs, dtype='complex')
    
    # calculate exciton reorg energies and line broadening functions for individual excitons
    # first calculate the contribution to total reorganisation energy of sites due to high energy modes and include this in site reorg energies
    # (this currently assumes all sites have same spectral density)
    high_energy_reorg_energy = np.sum([mode_params[0]*mode_params[1] for mode_params in high_energy_mode_params]) if high_energy_mode_params is not None and high_energy_mode_params.any() else 0
    site_reorg_energies += high_energy_reorg_energy
    exciton_reorg_energies = []
    exciton_lbfs = []
    for exciton in evecs:
        exciton_reorg_energies.append(exciton_reorg_energy(exciton, site_reorg_energies))
        exciton_lbfs.append(exciton_lbf(exciton, site_lbfs))
    exciton_lbfs = np.array(exciton_lbfs, dtype='complex')
        
    # store reorg energies for exciton mixing in N(N-1) x 3 matrix and generalise function to calculate exciton reorg energies
    # store line broadening functions for exciton mixing in N(N-1) x 6 matrix
    num_transitions = num_sites*(num_sites - 1.)
    counter = 0
    mixing_reorg_energies = np.empty((num_transitions, 2))
    mixing_line_broadening_functions = np.empty((num_transitions, 4, time.size), dtype='complex')
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j:
                mixing_reorg_energies[counter, 0] = generalised_exciton_reorg_energy(np.array([evecs[j], evecs[i], evecs[j], evecs[j]]), site_reorg_energies)
                mixing_reorg_energies[counter, 1] = generalised_exciton_reorg_energy(np.array([evecs[i], evecs[i], evecs[j], evecs[j]]), site_reorg_energies)
                
                mixing_line_broadening_functions[counter, 0] = generalised_exciton_lbf(np.array([evecs[j], evecs[i], evecs[j], evecs[i]]), site_lbfs)
                mixing_line_broadening_functions[counter, 1] = generalised_exciton_lbf(np.array([evecs[j], evecs[i], evecs[j], evecs[j]]), site_lbfs)
                mixing_line_broadening_functions[counter, 2] = generalised_exciton_lbf(np.array([evecs[j], evecs[i], evecs[i], evecs[i]]), site_lbfs)
                mixing_line_broadening_functions[counter, 3] = generalised_exciton_lbf(np.array([evecs[i], evecs[i], evecs[j], evecs[j]]), site_lbfs)
                
                counter += 1
                
    # calculate fluoresence and absorption via FFT for each exciton
    abs_lineshapes = np.empty((num_sites, time.size), dtype='complex')
    fl_lineshapes = np.empty((num_sites, time.size), dtype='complex')
    for i in range(num_sites):
        abs_lineshapes[i] = absorption_line_shape(time, evals[i]+exciton_reorg_energies[i], exciton_lbfs[i])
        fl_lineshapes[i] = fluorescence_line_shape(time, evals[i]+exciton_reorg_energies[i], exciton_reorg_energies[i], exciton_lbfs[i])
    
    # calculate N function for each pair of excitons
    mixing_function = np.empty((num_sites, num_sites, time.size-5), dtype='complex')
    start_index = 0
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j:
                mixing_function[i,j] = modified_redfield_mixing_function(mixing_line_broadening_functions[start_index], mixing_reorg_energies[start_index], time)
                start_index += 1
    
    # return integrand and time arrays before integrating to check integrand decays within time interval
    #return abs_lineshapes, fl_lineshapes, mixing_function, time
    
    # put everything together to calculate modified Redfield rates between all excitons
    rates = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j:
                rates[i,j] = modified_redfield_integration_simps(abs_lineshapes[i][:-5], fl_lineshapes[j][:-5].conj(), mixing_function[i,j], time[:-5])
 
    return rates#, abs_lineshapes, fl_lineshapes, mixing_function, time

# site line broadening function, check against other function
def site_lbf_ed(time, coeffs):
    return np.array([np.sum([(coeff[0] / coeff[1]**2) * (np.exp(-coeff[1]*t) + (coeff[1]*t) - 1.) for coeff in coeffs]) for t in time], dtype='complex')

# first differential of site line broadening function, check against numerical differentiation
def site_lbf_dot_ed(time, coeffs):
    return np.array([np.sum([(coeff[0] / coeff[1]) * (1. - np.exp(-coeff[1]*t)) for coeff in coeffs]) for t in time], dtype='complex')

# second differential of site line broadening function, check against numerical differentiation
def site_lbf_dot_dot_ed(time, coeffs):
    return np.array([np.sum([coeff[0] * np.exp(-coeff[1]*t) for coeff in coeffs]) for t in time], dtype='complex')

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

