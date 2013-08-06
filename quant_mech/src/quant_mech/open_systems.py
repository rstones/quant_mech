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
import quant_mech.utils as utils

# Function to construct Liouvillian super operator
# jump_operators should be list of tuples (first entry of tuple is lindblad operator, second is rate)
def super_operator(H, jump_operators):
    print "Constructing super-operator..."
    I = np.eye(H.shape[0], H.shape[1])
    L = -1.j * (np.kron(H, I) - np.kron(I, H))
    for tup in jump_operators:
        A = tup[0]
        A_dagger = A.conj().T
        A_dagger_A = np.dot(A_dagger, A)
        L += tup[1] * (np.kron(A, A) - 0.5 * np.kron(A_dagger_A, I) - 0.5 * np.kron(I, A_dagger_A))
    return L
    

def markovian_dissipator(density_matrix, lindblad_operator, dissipation_rate):
    lindblad_operator_dagger = lindblad_operator.conj().T
    return dissipation_rate * (np.dot(lindblad_operator, np.dot(density_matrix, lindblad_operator_dagger)) - 0.5 * utils.anticommutator(np.dot(lindblad_operator_dagger, lindblad_operator), density_matrix))