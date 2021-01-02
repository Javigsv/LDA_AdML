import numpy as np
from scipy.special import digamma
from DataLoader import DataLoader
import time


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

data: M x N   ----    List of M lists. Each sublist contains N elements according to the words of each document. Note that N could be different for each document

'''

'''
TODO: 
1. Check the computation of alpha
2. Optimise the computation of beta and lambda
3. Test first iterations
'''


def EM_algorithm(corpus, V, k):
  # Initalize priors (beta, eta, alpha)

  
  alpha_old, beta_old, eta = initialize_parameters_EM(V, k)

  # Initialize new parameters just to not fullfil convergence criteria
  alpha_new = alpha_old + 100
  beta_new = beta_old + 100

  M = len(corpus)

  it = 0
  while not convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new):
    it += 1
    print('EM Iteration:', it)

    # E-step
    phis, gammas = [], []
    print('\tE-Step...')
    for document in corpus:

      phi, gamma = VI_algorithm(alpha_old, beta_old, eta, document, corpus, V, k)

      phis.append(phi), gammas.append(gamma)

    # M-step
    print('\tM-Step...')
    beta_new = calculate_beta(phis, corpus, V, k)

    alpha_new = calculate_alpha(phis, gammas)

  
  return alpha_new, beta_new


def VI_algorithm(alpha, beta, eta, document, corpus, V, k):
  # Extended pseudocode from page 1005

  N = len(document)

  # Step 1-2: Initalize parameters
  phi_old, gamma_old, lambda_old = initialize_parameters_VI(alpha, N, k)

  # Initialize new parameters just to not fullfil convergence criteria
  phi_new = phi_old + 100
  gamma_new = gamma_old + 100

  # Step 3-9: Run VI algorithm
  it = 0
  while not convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new):
    # Iterate between phi, gamma and lambda
    it += 1
    #print('\tVI Iteration:', it)

    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, beta, document, k)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha, k)

    # Calculate the new lambdas (not sure about this one)
    #lambda_new = calculate_lambda(phi_new, eta, corpus, V, k)

    phi_old, gamma_old = phi_new, gamma_new

  return phi_old, gamma_old # Return the wanted parameters


def calculate_beta(phis, corpus, V, k):
  # Use the analytical expression in equation 9 page 1006
  print("Beta computation")
  M = len(corpus)

  beta = []

  for i in range(k):
    print(i)
    beta.append([])
    for j in range(V):
      print('\t',j)
      beta[i].append(0)

      #for d in range(M):

        # Approach 1: Slightly slower
        #beta[i][j] += np.sum(phis[d][np.array(corpus[d]) == j,i])

        # Approach 2: Considerably slower
        #N = len(corpus[d])
        #for n in range(N):
        #  help2 += phis[d][n,i] * 1 if j == corpus[d][n] else 0

      
      # Approach 3: Quickest but still iterating over M
      beta[i][j] = np.sum([np.sum(phis[d][np.array(corpus[d]) == j,i]) for d in range(M)])

    beta[i] = beta[i] / sum(beta[i]) # Normalizing

  print('alpha comp')

  return np.array(beta)


def calculate_alpha(phis, gammas, alpha):
  # Use Newton-Raphson method suggested in A.2, page 1018-1019

  '''
  alpha_new = alpha_old - H(alpha_old)^-1 * g(alpha_old)

  Let H = diag(h) + 1z1^T

  Now (H^-1g)_i = (g_i-c)/h_i where

  c = sum(g_j / h/j) / (z^-1 +  sum(h_j^-1))


  https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf page 1-2

  delta function delta(x) = 
                      1 if x = 0  
                      0 otherwise
  trigamma =  derivative of digamma

  log_p = not exactly sure

  '''

  alpha_old = alpha

  while not converged:

    q = -N * [trigamma(alpha[k]) for k in range(len(alpha))]

    z = N * trigamma(sum(alpha))

    d = digamma(sum(alpha))

    log_p_mean = 1 / N * sum(log_p)

    g = N * [d - digamma(alpha[k]) + log_p_mean]

    b = sum([g[j]/q[j] for j in range(len(alpha))]) / ( 1/z + sum([1/q[j] for j in range(len(alpha))]))

    alpha_new = [alpha_old - (g[k] - b) / q[k] for k in range(len(alpha))]

  return alpha_new


def calculate_phi(gamma_old, beta, document, k):

  N = len(document)

  phi = []
  for n in range(N):
    phi_n = beta[:, document[n]] * np.exp(digamma(gamma_old)) # According to step 6
    
    phi_n = phi_n / np.sum(phi_n) # Normalize phi since it's a probability (must sum up to 1)

    phi.append(phi_n)

  return np.array(phi)


def calculate_gamma(phi_new, alpha, k):
  # According to equation 7 on page 1004

  #gamma = [alpha[i] + np.sum(phi_new[:,i]) for i in range(k)] 
  gamma = alpha + np.sum(phi_new, 0)

  return gamma


def calculate_lambda(phi_new, eta, corpus, V, k):
  # Not sure about this one
  # lambda_new = eta + [[phi_new[d,n,i] * w[d,n,j] for i in range()] for j in range()]

  M = len(corpus)

  lambd = []

  for i in range(k):
    lambd.append([])
    for j in range(V):
      lambd[i].append(eta)

      for d in range(M):
        N = len(corpus[d])
        for n in range(N):
         lambd[i][j] += phi_new[d][n][i] * 1 if j == corpus[d][n] else 0

  return lambd


def initialize_parameters_VI(alpha, N, k):
  # phi = [[1 / k for i in range(k)] for j in range(k)]   # k is the number of priors (from the Dirichlet distribution)
  # gamma = [alpha[i] + N / k for i in range(k)]       # N is the number of words, alpha the priors of the Dirichlet distribution
  phi = np.ones((k,k)) * 1/k
  gamma = alpha.copy() + N / k
  lambd = 1                                             # I don't know
  return phi, gamma, lambd


def initialize_parameters_EM(V, k):
  # TODO: This is just an approach, it may be wrong

  alpha = np.random.rand(k)       
  beta = np.random.rand(k, V)
  eta = 1                                             
  return alpha, beta, eta


def convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.any(np.abs(phi_old - phi_new) > threshold):
    return False

  if np.any(np.abs(gamma_old - gamma_new) > threshold):
    return False

  return True


def convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.any(np.abs(alpha_old - alpha_new) > threshold):
    return False

  if np.any(np.abs(beta_old - beta_new) > threshold):
    return False

  return True


def load_data(filename):

  data_loader = DataLoader(filename)
  data, V = data_loader.load()

  return data, V


def main():

  k = 10

  filename = './Code/Reuters_Corpus_Vectorized.csv'

  corpus, V = load_data(filename)

  alpha, beta = EM_algorithm(corpus, V, k)

  print(alpha, beta)

if __name__ == "__main__":
    main()