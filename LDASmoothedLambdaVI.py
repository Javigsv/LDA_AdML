'''
0. Initialize parameters

1. E-step - VI-algorithm

  1.1 For every document

    1.1.1 Calculate gamma

    1.1.2 Calculate phi
  
  1.2 Calculate lambdas

  1.3 Check for convergence in lower-bound

2. M-step

  2.1 NR for alpha

  2.2 NR for eta

3 Check for convergence in lower-bound

'''

import numpy as np
from scipy.special import digamma, polygamma
from scipy.special import gamma as gamma_function
from scipy.special import loggamma as loggamma
import scipy
from DataLoader import DataLoader
import time, csv
from datetime import datetime
import EachMovieParser.Eachmovie_parser as eachmovie


## Load data
def load_data(filename, num_documents = 10**6):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


## Derivative of the digamma function (help-function)
def trigamma(a):
  return polygamma(1, a)


## VI-algorithm run during the E-step for every document m
def VI_algorithm(k, V, corpus, phis_old, gammas_old, lambda_old, alpha, eta, alpha_sum, tolerance = 1e-2, debug = False):
  lower_bound_old = lower_bound_corpus(alpha, eta, lambda_old, phis_old, gammas_old, alpha_sum, k, V, corpus)

  # Extended pseudocode from page 1005
  it = 0
  while True:
    it += 1
    
    phis_new = []

    gammas_new = []

    for (d, document) in enumerate(corpus):
      # Calculate the new phis
      phi_new = calculate_phi(gammas_old[d], lambda_old, document, k)

      phis_new.append(phi_new)

      # Calculate the new gammas
      gamma_new = calculate_gamma(phis_new[d], alpha, k)

      gammas_new.append(gamma_new)
  
    # Calculate the new lambdas (not sure about this one)
    lambda_new = calculate_lambda(phis_new, eta, corpus, V, k)

    lower_bound_new = lower_bound_corpus(alpha, eta, lambda_new, phis_new, gammas_new, alpha_sum, k, V, corpus)

    convergence = abs((lower_bound_old-lower_bound_new) / lower_bound_old)
    # print(lower_bound_new)
    # print(convergence)
    
    if convergence < tolerance:
      break
    else:
      phis_old = phis_new
      gammas_old = gammas_new
      lambda_old = lambda_new
      lower_bound_old = lower_bound_new
  print("\tVI finished in", it, "iterations!")
  return phis_new, gammas_new, lambda_new


## Calculate phi for document m
def calculate_phi3(gamma, lambdas, document, k):
  N = len(document)

  # According to step 6
  phi = lambdas[:, document[:]].T * np.tile(np.exp(digamma(gamma) - digamma(np.sum(gamma))),(N,1))

  # phi2 = np.log(lambdas[:, document[:]].T) + np.tile(digamma(gamma) - digamma(np.sum(gamma)),(N,1))

  # Normalize phi since it's a probability (must sum up to 1)
  phi = phi/np.sum(phi, axis = 1)[:, np.newaxis]
  # phi2 = np.exp(phi2 - np.tile(scipy.special.logsumexp(phi2, axis = 1), (k,1)).T)

  return phi


## Calculate phi for document m
def calculate_phi(gamma, lambdas, document, k):
  N = len(document)

  phi = []
  for n in range(N):
    # According to step 6
    # phi_n = np.exp(digamma(lambdas[:, document[n]]) - digamma(np.sum(lambdas, axis=1))) * np.exp(digamma(gamma) - digamma(np.sum(gamma)))
    log_phi_n = digamma(lambdas[:, document[n]]) - digamma(np.sum(lambdas, axis=1)) + digamma(gamma) - digamma(np.sum(gamma))

    log_phi_n = log_phi_n - scipy.special.logsumexp(log_phi_n)
    # phi_n = beta[:, document[n]] * np.exp(digamma(gamma))
    # print(phi_n)
    # Normalize phi since it's a probability (must sum up to 1)
    # phi_n = phi_n / np.sum(phi_n)
    phi_n = np.exp(log_phi_n)

    phi.append(phi_n)
  
  phi = np.array(phi)
  
  return phi


## Calculate gamma for document m
def calculate_gamma(phi, alpha, k):
  # According to equation 7 on page 1004
  gamma = alpha + np.sum(phi, axis = 0)

  return gamma


## To calculate beta in the M-step
def calculate_lambda(phis, eta, corpus, V, k):
  lambdas = np.ones((k,V)) * eta

  for d in range(len(corpus)):
    N = len(corpus[d])
    for n in range(N):
      j = corpus[d][n]
      lambdas[:, j] += phis[d][n,:]
  
  # lambdas = lambdas/np.sum(lambdas, axis = 1)[:, np.newaxis]
  
  return lambdas


## The lower bound for a single document
def lower_bound_single(alpha, lambdas, phi, gamma, alpha_sum, k, document, debug = False):
  # Helpful things

  digamma_gamma_sum = digamma(np.sum(gamma))

  N = len(document)

  # Calculating the lower bound according to page 1020

  # First row
  likelihood = alpha_sum # First part of first row

  likelihood += np.sum((alpha-1)*(digamma(gamma) - digamma_gamma_sum)) # Second part of first row

  # Second row
  likelihood += np.sum(phi * np.tile((digamma(gamma) - digamma_gamma_sum), (N,1)))

  # Third row
  for n in range(N):
    # likelihood += np.sum(phi[n,:]*np.log(np.maximum(lambdas[:,document[n]], 1e-90)))
    likelihood += np.sum(phi[n,:] * (digamma(lambdas[:,document[n]]) - digamma(np.sum(lambdas, axis = 1))))

  # The fourth row
  likelihood += -loggamma(np.sum(gamma)) + np.sum(loggamma(gamma)) - np.sum((gamma-1)*(digamma(gamma) - digamma_gamma_sum))

  # The fifth row
  likelihood += np.sum(np.log(phi ** phi))

  return likelihood


## The lower bound for the whole corpus
def lower_bound_corpus(alpha, eta, lambdas, phi, gamma, alpha_sum, k, V, corpus):
  likelihood = 0

  for (d, document) in enumerate(corpus):
    likelihood += lower_bound_single(alpha, lambdas, phi[d], gamma[d], alpha_sum, k, document)

  likelihood += k * (loggamma(V * eta) - V * loggamma(eta)) 

  likelihood += (eta - 1) * ((np.sum(digamma(lambdas))) - V * np.sum(digamma(np.sum(lambdas, axis = 1))))

  likelihood -= np.sum(loggamma(np.sum(lambdas, axis = 1))) - np.sum(loggamma(lambdas)) + np.sum((lambdas-1) * (digamma(lambdas) - np.tile(digamma(np.sum(lambdas, axis = 1)),(V,1)).T))
  
  return likelihood


## Initialize EM parameters
def initialize_parameters_EM(V, k):
  np.random.seed(1)

  # E) I think that we should maybe encode sparcity into each into the Dirichlet. See https://youtu.be/o22cA1DhSMQ?t=1566 for how an alpha < 1 does this.
  """ approx_alpha = 0.01 # 0.1
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)
  input(alpha) """

  approx_alpha = 0.001
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)      

  approx_eta = 0.2
  eta = np.random.uniform(approx_eta - 0.1 * approx_eta, approx_eta + 0.1 * approx_eta) 

  return alpha, eta


## Initialize VI parameters
def initialize_parameters_VI(alpha, eta, corpus, k, V):
  np.random.seed(1)
  # M x N_d x k
  phi = []
  for document in corpus:
    phi.append(np.ones((len(document),k)) * 1/k)

  # M x k
  gamma = np.tile(alpha.copy(),(len(corpus),1)) + np.tile(np.array(list(map(lambda x: len(x),corpus))),(k,1)).T / k
  
  # lambdas = np.random.rand(k, V)

  # for i in range(k):
  #   # Normalizing
  #   lambdas[i,:] = lambdas[i,:] / sum(lambdas[i,:])

  lambdas = np.ones((k,V)) * (eta + len(corpus[0])/V) + np.random.rand(k, V) * 2
  
  return phi, gamma, lambdas


## Newton-Raphson function to calculate new alpha in the M-step
def calculate_alpha(gamma, alpha, M, k, nr_max_iterations = 1000, tolerance = 10 ** -4):
  # Use Newton-Raphson method with linear complexity suggested by Thomas P. Minka in
  # Estimating a Dirichlet distribution

  gamma = np.array(gamma)

  log_p_mean = np.sum((digamma(gamma)-np.tile(digamma(np.sum(gamma,axis=1)),(k,1)).T),axis=0)

  for it in range(nr_max_iterations):
    alpha_old = alpha

    # Calculate the observed efficient statistic
    # Here we are using that the expected sufficient statistics are equal to the observed sufficient statistics
    # for distributions in the exponential family when the gradient is zero
    # log_p_mean = np.sum((digamma(gamma)-np.tile(digamma(np.sum(gamma,axis=1)),(k,1)).T),axis=0)

    g = M * (digamma(np.sum(alpha)) - digamma(alpha)) + log_p_mean

    # Calculate the diagonal of the Hessian
    h = - M * trigamma(alpha)

    # Calculate the constant component of the Hessian
    z = M * trigamma(np.sum(alpha))

    # Calculate the constant
    b = np.sum(g/h) / (1/z + np.sum(1/h))

    # Update equation for alpha
    alpha = alpha - (g - b) / h
    # log_alpha = np.log(alpha) - (g - b) / h
    # alpha = np.exp(log_alpha)

    # if np.all(np.abs(g) < tolerance):
    if np.linalg.norm(alpha-alpha_old) < tolerance:
      break

    if np.any(alpha < 0):
      print("Alpha is negative!")

  return np.abs(alpha)


## Newton-Raphson function to calculate new eta in the M-step
def calculate_eta(lambdas, eta, V, k, nr_max_iterations = 1000, tolerance = 10 ** -4):

  for it in range(nr_max_iterations):
    # print(eta)
    eta_old = eta

    df = V * k * (digamma(V * eta) - digamma(eta)) + np.sum(digamma(lambdas)) - V * np.sum(digamma(np.sum(lambdas, axis = 1)))

    df2 = V * k * (V * trigamma(V * eta) - trigamma(eta))

    step = - df / df2
    
    # Update equation for alpha
    eta = eta + step

    if np.abs((eta-eta_old) / eta_old) < tolerance:
      break

    
    

  if np.any(eta < 0):
    print("Eta is negative!")

  return np.abs(eta)


## Taken from Blei
def calculate_eta2(lambdas, eta, V, k, nr_max_iterations = 1000, tolerance = 10 ** -4):
  initial_eta = 100

  di_lambda_sum = np.sum(digamma(lambdas)) - V * np.sum(digamma(np.sum(lambdas, axis = 1)))

  log_eta = np.log(initial_eta)

  for it in range(nr_max_iterations):
    eta = np.exp(log_eta)

    df = V * k * (digamma(V * eta) - digamma(eta)) + di_lambda_sum

    df2 = V * k * (V * trigamma(V * eta) - trigamma(eta))

    log_eta = log_eta - df / (df2 * eta + df)

    if np.abs(df) < tolerance:
      break
  # print(it)
  if it == nr_max_iterations - 1:
    print("Max reached!")

  return np.exp(log_eta)


## LDA function
def LDA_algorithm(corpus, V, k, tolerance = 1e-4):
  alpha_old, eta_old = initialize_parameters_EM(V, k)
  M = len(corpus)

  phis_old, gammas_old, lambda_old = initialize_parameters_VI(alpha_old, eta_old, corpus, k, V)

  alpha_sum_old = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))

  lower_bound_old = lower_bound_corpus(alpha_old, eta_old, lambda_old, phis_old, gammas_old, alpha_sum_old, k, V, corpus)
  
  print("Starting lower bound:", lower_bound_old)

  start_EM = time.time()
  it = 0
  ########################
  # --- EM-algorithm --- #
  ########################
  while True:
    # print(gammas_old.shape)
    it += 1

    print("\nEM-iteration:", it)

    ##################
    # --- E-step --- #
    ##################
    print("E-step...")
    start = time.time()
    
    # phi_old, gamma_old = phi_new, gamma_new
    phis_new, gammas_new, lambda_new = VI_algorithm(k, V, corpus, phis_old, gammas_old, lambda_old, alpha_old, eta_old, alpha_sum_old)

    stop = time.time()
    print('...completed in:', stop - start)
    ##################
    # --- M-step --- #
    ##################
    print("M-step...")
    start = time.time()
    
    print("\t... calculate alpha")
    alpha_new = calculate_alpha(gammas_old, alpha_old, M, k)
    
    alpha_sum_new = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))

    print("\t... calculate eta")

    eta_new = calculate_eta(lambda_old, eta_old, V, k)
    print(eta_old, "-->", eta_new)
    # eta_new_vector = calculate_alpha(lambda_old, eta_old * np.ones(V), k, V)
    # print(eta_new_vector)
    # eta_new = eta_new_vector[0]
    
    stop = time.time()
    print('...completed in:', stop - start)
    ########################
    # --- Convergence? --- #
    ########################

    print("Computing the lower bound...")
    start_n = time.time()
    lower_bound_new = lower_bound_corpus(alpha_new, eta_new, lambda_new, phis_new, gammas_new, alpha_sum_new, k, V, corpus)
    print("\t", lower_bound_new)
    stop_n = time.time()
    print('...completed in:', stop_n - start_n)

    if lower_bound_new < lower_bound_old:
      print("Oh no! The lower bound decreased...")

    # The change of the lower bound
    delta_lower_bound = abs((lower_bound_old-lower_bound_new) / lower_bound_old)
    print("The lower bound changed: {}%".format(np.round(100*delta_lower_bound, 5)))
    if delta_lower_bound < tolerance:
      print("Convergence after", it, "iterations!")
      break

    else:
      alpha_old = alpha_new
      eta_old = eta_new
      lower_bound_old = lower_bound_new
      alpha_sum_old = alpha_sum_new
    
  stop_EM = time.time()
  print('\nThe algorithm converged in', stop_EM - start_EM, "seconds")
  return [alpha_new, eta_new, phis_old, gammas_old, lambda_new]


## Print parameters
def print_parameters(parameters, printing = True):
  if printing:
    params = ["alpha", "eta", "phi", "gamma", "lambda"]
  
    for (i,param) in enumerate(params):
      print(param + ":")
      print(parameters[i])


## Main function reuters
def main_Reuters():
  # Initial parameters
  k = 10              # Number of topics
  num_documents = 200 #10**6

  # File directories
  vocab_file = './Code/Reuters_Corpus_Vocabulary.csv'
  filename = './Code/Reuters_Corpus_Vectorized.csv'

  # Load data
  corpus, V = load_data(filename, num_documents)
  
  nTraining = int(num_documents * 0.9)

  test = corpus[nTraining:]
  corpus = corpus[:nTraining]

  # Run the algorithm
  parameters = LDA_algorithm(corpus, V, k)
  print(parameters[0])
  print(parameters[1])
  # Print the parameters
  print_parameters(parameters, False)

  # # Print most likely topics and words
  # alpha = parameters[0]
  # num_topics = 5 # The number of topics that should be printed
  # topic_indices = print_likely_topics(alpha, num_topics)
  # beta = parameters[1]
  # print_top_words_for_all_topics(vocab_file, beta, top_x=15, k=k, indices = topic_indices)

  # phi = parameters[2]
  # gamma = parameters[3]
  # print(len(corpus), len(test))
  # print_perplexity(alpha, beta, phi, gamma, k, corpus, test)

  # print()


main_Reuters()