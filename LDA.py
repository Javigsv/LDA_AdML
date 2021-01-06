import numpy as np
from scipy.special import digamma, polygamma
from scipy.special import gamma as gamma_function
import scipy
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

## Load data
def load_data(filename, num_documents):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


## Derivative of the digamma function (help-function)
def trigamma(a):
  return polygamma(1, a)


## Initialize EM parameters
def initialize_parameters_EM(V, k):
  np.random.seed(1)

  # E) I think that we should maybe encode sparcity into each into the Dirichlet. See https://youtu.be/o22cA1DhSMQ?t=1566 for how an alpha < 1 does this.
  """ approx_alpha = 0.01 # 0.1
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)
  input(alpha) """

  approx_alpha = 1
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)      

  # alpha = np.full(shape = k, fill_value = 50/k)

  beta = np.random.rand(k, V)
  for i in range(k):
    # Normalizing
    beta[i,:] = beta[i,:] / sum(beta[i,:])

  eta = 1

  return alpha, beta, eta


## Initialize VI parameters
def initialize_parameters_VI(alpha, N, k):
  phi = np.ones((N,k)) * 1/k
  gamma = alpha.copy() + N / k
  lambd = 1 # Should probably be changed
  return phi, gamma, lambd


## Calculate phi for document m
def calculate_phi(gamma, beta, document, k):

  N = len(document)

  phi = np.zeros((N,k))
  for n in range(N):
    for i in range(k):
      phi[n,i] = beta[i, document[n]] * np.exp(digamma(gamma[i]) - digamma(np.sum(gamma)))
  

  phi = phi/np.sum(phi, axis = 1)[:, np.newaxis]

  return phi


## Calculate gamma for document m
def calculate_gamma(phi, alpha, k):
  # According to equation 7 on page 1004
  # gamma = alpha + np.sum(phi, axis = 0)
  gamma = np.zeros(k)
  for i in range(0,k):
    gamma[i] = alpha[i] + np.sum(phi[:, i]) #updating gamma

  return gamma


## Check for convergence in VI-algorithm
def convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new, threshold = 1e-4):
  # Implement convergence criteria

  if np.any(np.abs(phi_old - phi_new) > threshold):
    return False

  if np.any(np.abs(gamma_old - gamma_new) > threshold):
    return False

  return True


## VI-algorithm run during the E-step for every document m
def VI_algorithm(k, document, alpha, beta, eta, debug = False):
  N = len(document)

  # Extended pseudocode from page 1005

  # Step 1-2: Initalize parameters
  phi_old, gamma_old, lambda_old = initialize_parameters_VI(alpha, N, k)

  # Step 3-9: Run VI algorithm
  it = 0
  while True:
    it += 1
    
    if debug:
      print('\tVI Iteration:', it)
    
    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, beta, document, k)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha, k)
  
    # Calculate the new lambdas (not sure about this one)
    #lambda_new = calculate_lambda(phi_new, eta, corpus, V, k)
  
    if convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new):
      break
    else:
      phi_old = phi_new
      gamma_old = gamma_new

  return phi_new, gamma_new
    

## To calculate beta in the M-step
def calculate_beta(phi, corpus, V, k):
  # Use the analytical expression in equation 9 page 1006
  beta = np.zeros((k,V))

  for i in range(k):
    for j in range(V):
      s = 0
      for d in range(len(corpus)):
        N = len(corpus[d])
        for n in range(N):
          if corpus[d][n] == j:
            s += phi[d][n,i]
      beta[i,j] = s

  # Normalize
  # beta =  beta / beta.sum(axis=1,keepdims=1)
  beta = beta/np.sum(beta, axis = 1)[:, np.newaxis]

  return beta


## Newton-Raphson function to calculate new alpha in the M-step
def calculate_alpha(gamma, alpha, M, k, nr_max_iterations = 1000, tolerance = 10 ** -2):
  # Use Newton-Raphson method with linear complexity suggested by Thomas P. Minka in
  # Estimating a Dirichlet distribution

  gamma = np.array(gamma)
  log_p_mean = np.sum((digamma(gamma)-np.tile(digamma(np.sum(gamma,axis=1)),(k,1)).T),axis=0)

  for it in range(nr_max_iterations):
    alpha_old = alpha

    # Calculate the observed efficient statistic
    # Here we are using that the expected sufficient statistics are equal to the observed sufficient statistics
    # for distributions in the exponential family when the gradient is zero
    log_p_mean = np.sum((digamma(gamma)-np.tile(digamma(np.sum(gamma,axis=1)),(k,1)).T),axis=0)

    g = M * (digamma(np.sum(alpha)) - digamma(alpha)) + log_p_mean

    # Calculate the diagonal of the Hessian
    h = M * trigamma(alpha)

    # Calculate the constant component of the Hessian
    z = M * trigamma(np.sum(alpha))

    # Calculate the constant
    b = np.sum(g/h) / (1/z + np.sum(1/h))

    # Update equation for alpha
    alpha = alpha + (g - b) / h

    if np.linalg.norm(alpha-alpha_old) < tolerance:
      break


  return alpha


## Copied straight of [Not used atm]
def calculate_alpha2(gamma, alphaOld, M, k, nr_max_iterations = 1000, tol = 10 ** -4):
  h = np.zeros(k)
  g = np.zeros(k)
  alphaNew = np.zeros(k)

  converge = 0
  while converge == 0:
    for i in range(0, k):
      docSum = 0
      for d in range(0, M):
        docSum += digamma(gamma[d][i]) - digamma(np.sum(gamma[d]))
      g[i] = M*(digamma(sum(alphaOld)) - digamma(alphaOld[i])) + docSum
      h[i] = M*trigamma(alphaOld[i])
    z =  M*trigamma(np.sum(alphaOld))
    c = np.sum(g/h)/(1/z + np.sum(1/h))
    step = (g - c)/h
    # print(step)
    alphaNew = alphaOld +step
    if np.linalg.norm(step) < tol:
    # if np.linalg.norm(alphaNew-alphaOld) < tol:
      
      converge = 1
    else:
      converge = 0
      alphaOld = alphaNew

  # print(alphaNew)
  # raise ValueError('A very specific bad thing happened.')
  return alphaNew


## Check for convergence in EM-algorithm
def convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new, debug = False, threshold = 1e-4):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.any(np.abs(alpha_old - alpha_new) > threshold):
    if debug:
      print("alpha not converged:")
      print("from")
      print(alpha_old)
      print("to")
      print(alpha_new)
    return False

  if np.any(np.abs(beta_old - beta_new) > threshold):
    if debug:
      print("beta not converged:")
      print("from")
      print(beta_old)
      print("to")
      print(beta_new)
    return False

  return True


## The lower bound
def lower_bound(alpha, beta, phis, gammas, k, V, corpus):
  L = 0
  # Parts on corpus level
  alpha_part_1 = np.log(gamma_function(np.sum(alpha))) - np.sum(np.log(gamma_function(alpha)))

  for (d, document) in enumerate(corpus):
    N = len(document)
    gamma = gammas[d]
    phi = phis[d]

    digamma_gamma = digamma(np.sum(gamma))

    # The first row
    row_1 =  alpha_part_1 + np.sum((alpha-1)*(digamma(gamma) - digamma_gamma))


    # The second row
    row_2 = np.sum(phi * np.tile((digamma(gamma) - digamma_gamma), (N,1)))


    # The third row
    row_3 = 0
    for n in range(N):
      for j in range(V):
        for i in range(k):
          if document[n] == j:
            row_3 += phi[n,i]*np.log(beta[i,j])


    # The fourth row
    # print("Test")
    # print(gamma)
    # print(-np.log(gamma_function(np.sum(gamma))))
    # print(gamma_function(np.sum(gamma)))
    # print(np.sum(np.log(gamma_function(gamma))))
    # print( - np.sum((gamma-1)*(digamma(gamma) - digamma_gamma)))
    row_4 = -np.log(gamma_function(np.sum(gamma))) + np.sum(np.log(gamma_function(gamma))) - np.sum((gamma-1)*(digamma(gamma) - digamma_gamma))

    # print(np.sum(gamma))
    # raise ValueError('A very specific bad thing happened.')

    # The fifth row
    row_5 = np.sum(phi * np.log(phi))

    # Printing values of rows
    rows = [row_1, row_2, row_3, row_4, row_5]
    print("\nDocument", d+1, ":")
    for (i,row) in enumerate(rows):
      print("Row", i+1, "::", row)

    L += sum(rows)
  
  return L


## LDA function
def LDA_algorithm(corpus, V, k):
  alpha_old, beta_old, eta_old = initialize_parameters_EM(V, k)
  M = len(corpus)

  it = 0
  ########################
  # --- EM-algorithm --- #
  ########################
  while True:
    # print(alpha_old)
    it += 1

    print("EM-iteration:", it)

    ##################
    # --- E-step --- #
    ##################
    print("\tE-step")
    phi, gamma = [], []

    for document in corpus:
      phi_d, gamma_d = VI_algorithm(k, document, alpha_old, beta_old, eta_old)
      phi.append(phi_d), gamma.append(gamma_d)
    
    ##################
    # --- M-step --- #
    ##################
    print("\tM-step")

    beta_new = calculate_beta(phi, corpus, V, k)

    alpha_new = calculate_alpha(gamma, alpha_old, M, k)

    ########################
    # --- Convergence? --- #
    ########################
    # print(lower_bound(alpha_new, beta_new, phi, gamma, k, V, corpus))
    if convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new, debug = True):
      print("Convergence after", it, "iterations!")
      break
    else:
      alpha_old = alpha_new
      beta_old = beta_new
    
  
  return [alpha_new, beta_new, phi, gamma]


## Main function
def main():
  
  k = 3

  filename = './Code/Reuters_Corpus_Vectorized.csv'

  corpus, V = load_data(filename, 20)

  parameters = LDA_algorithm(corpus, V, k)

  params = ["alpha", "beta", "phi", "gamma"]

  for (i,param) in enumerate(params):
    print(param + ":")
    print(parameters[i])


if __name__ == "__main__":
  main()