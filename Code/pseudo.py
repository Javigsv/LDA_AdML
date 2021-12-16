import numpy as np
import scipy
from DataLoader import DataLoader


"""
The algorithm in short:

(The outher algorithm is EM)

===========================

0. Initialize priors (alpha, beta, eta)

1. (E-step) Calculate optimal gamma and phi for all documents with the VI-algorithm given the current priors

2. (M-step) Maximize the lower bound w.r.t the priors

  2.1 If the priors / lower bound haven't converged, return to step 1 and use the new priors

"""

'''
Variables:

V: Number of words of the vocabulary
N: Number of words of the document
M: Number of documents in the corpus
k: Number of possible topics

alpha: k x 1  ----    Parameter of the Dirichlet distribution of theta
beta: k x V   ----    Parameter of the Multinomial distribution that will generate word w conditioned on the topic z

theta: k x 1  ----    Parameter of the Multinomial distribution that will generate the topic z
z: k x 1      ----    Topic. It designs the beta parameter used in the multinomial 

gamma: k x 1  ----    Variational parameter for theta
phi: k x N    ----    Variational parameter for the topic z

'''

'''
TODO: 
1. Load the data using the class DataLoader
2. Decide the implementation of variables N, M, k and V : either global variables or parameters of the methods
3. Test first iterations
'''


def EM_algorithm():
  # Initalize priors (beta, eta, alpha)

  while not converged:

    # E-step
    phis, gammas = [], []
    for d in documents:
      phi, gamma = VI_algorithm(alpha, beta, eta)

      phis.append(phi), gammas.append(gamma)

    # M-step
    beta_new = calculate_beta(phis, gammas)

    alpha_new = calculate_alpha(phis, gammas)


def VI_algorithm(alpha, beta, eta):
  # Extended pseudocode from page 1005

  N, k = ,

  # Step 1-2: Initalize parameters
  phi_old, gamma_old, lambda_old = initialize_parameters(alpha, N, k)

  # Step 3-9: Run VI algorithm
  while convergence_criteria(phi_old, gamma_old, phi_new, gamma_new):
    # Iterate between phi, gamma and lambda

    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, beta)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha)

    # Calculate the new lambdas (not sure about this one)
    lambda_new = calculate_lambda(phi_new, eta)

    phi_old, gamma_old = phi_new, gamma_new

  return phi_old, gamma_old # Return the wanted parameters


def calculate_beta(phis, gammas):
  # Use the analytical expression in equation 9 page 1006
  beta = []

  for i in range(k):
    beta.append([])
    for j in range(V):
      
      beta[i][j] = 0

      for d in range(M):
        N = len(data[d])
        for n in range(N):
         beta[i][j] += phis[d][n][i] * 1 if j == data[d][n] else 0

    #TODO: Normalize each b[i]  

  return beta


def calculate_alpha(phis, gammas):
  # Use Newton-Raphson method suggested in A.2, page 1018-1019
  return alpha


def calculate_phi(gamma_old, beta):
  phi = []
  for n in range(N):
    phi_n = np.array([beta[i, word[n]] * np.exp(scipy.special.digamma(gamma_old[i])) for i in range(k)]) # According to step 6
    
    phi_n = phi_n / np.sum(phi_n) # Normalize phi since it's a probability (must sum up to 1)

    phi.append(phi_n)
  
  return phi


def calculate_gamma(phi_new, alpha):
  gamma = [alpha[i] + np.sum(phi_new[:,i]) for i in range(k)] # According to equation 7 on page 1004

  return gamma


def calculate_lambda(phi_new, eta):
  # Not sure about this one
  lambda_new = eta + [[sum_d sum_n phi[d,n,i] * w[d,n,j] for i in range()] for j in range()]

def initialize_parameters(alpha, N, k):
  phi = [[1 / k for i in range(k)] for j in range(k)]   # k is the number of priors (from the Dirichlet distribution)
  gamma = [alpha[i] + N / k for i in range(k)]       # N is the number of words, alpha the priors of the Dirichlet distribution
  lambd = 1                                             # I don't know
  return phi, gamma, lambd


def convergence_criteria(phi, gamma):
  # Implement convergence criteria
  pass