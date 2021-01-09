import numpy as np
from scipy.special import digamma, polygamma
from scipy.special import gamma as gamma_function
from scipy.special import loggamma as loggamma
import scipy
from DataLoader import DataLoader
import time, csv
from datetime import datetime



## TODO
# Fix the lower bound
# Make the calculation of beta more efficient


""" The algorithm in short:

(The outher algorithm is EM)

===========================

0. Initialize priors (alpha, beta, eta)

1. (E-step) Calculate optimal gamma and phi for all documents with the VI-algorithm given the current priors

2. (M-step) Maximize the lower bound w.r.t the priors

  2.1 If the priors / lower bound haven't converged, return to step 1 and use the new priors """


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
  #beta = np.ones((k, V)) * (1/V)
  for i in range(k):
    # Normalizing
    beta[i,:] = beta[i,:] / sum(beta[i,:])

  eta = 1

  return alpha, beta, eta


## Initialize VI parameters
def initialize_parameters_VI(alpha, corpus, k):
  # M x N_d x k
  phi = []
  for document in corpus:
    phi.append(np.ones((len(document),k)) * 1/k)

  # M x k
  gamma = np.tile(alpha.copy(),(len(corpus),1)) + np.tile(np.array(list(map(lambda x: len(x),corpus))),(k,1)).T / k

  lambd = 1 # Should probably be changed
  return phi, gamma, lambd


## Calculate phi for document m
def calculate_phi(gamma, beta, document, k):
  N = len(document)

  # According to step 6
  phi = beta[:, document[:]].T * np.tile(np.exp(digamma(gamma)),(N,1))

  # Normalize phi since it's a probability (must sum up to 1)
  phi = phi/np.sum(phi, axis = 1)[:, np.newaxis]

  return phi


## Calculate gamma for document m
def calculate_gamma(phi, alpha, k):
  # According to equation 7 on page 1004
  gamma = alpha + np.sum(phi, axis = 0)

  return gamma


## To calculate beta in the M-step
def calculate_beta(phi, corpus, V, k):
  beta = np.zeros((k,V))
  
  for d in range(len(corpus)):
    N = len(corpus[d])
    for n in range(N):
      j = corpus[d][n]
      beta[:, j] += phi[d][n,:]

  beta = beta/np.sum(beta, axis = 1)[:, np.newaxis]

  return beta


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


## The lower bound for a single document
def lower_bound_single(alpha, beta, phi, gamma, alpha_sum, k, document):
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
    likelihood += np.sum(phi[n,:]*np.log(np.maximum(beta[:,document[n]], 1e-90)))

  # The fourth row
  likelihood += -loggamma(np.sum(gamma)) + np.sum(loggamma(gamma)) - np.sum((gamma-1)*(digamma(gamma) - digamma_gamma_sum))

  # The fifth row
  likelihood += np.sum(np.log(phi ** phi))
  
  return likelihood


## The lower bound for the whole corpus
def lower_bound_corpus(alpha, beta, phi, gamma, alpha_sum, k, corpus):
  likelihood = 0

  for (d, document) in enumerate(corpus):
    likelihood += lower_bound_single(alpha, beta, phi[d], gamma[d], alpha_sum, k, document)

  return likelihood


## VI-algorithm run during the E-step for every document m
def VI_algorithm(k, document, phi_old, gamma_old, lambda_old, alpha, beta, eta, alpha_sum, tolerance = 1e-4, debug = False):
  N = len(document)

  lower_bound_old = lower_bound_single(alpha, beta, phi_old, gamma_old, alpha_sum, k, document)

  # Extended pseudocode from page 1005
  it = 0
  while True:
    it += 1
    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, beta, document, k)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha, k)
  
    # Calculate the new lambdas (not sure about this one)
    #lambda_new = calculate_lambda(phi_new, eta, corpus, V, k)

    lower_bound_new = lower_bound_single(alpha, beta, phi_new, gamma_new, alpha_sum, k, document)

    # if convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new):
    #   break
    if abs((lower_bound_old-lower_bound_new) / lower_bound_old) < tolerance:
      break
    else:
      phi_old = phi_new
      gamma_old = gamma_new
      lower_bound_old = lower_bound_new

  return phi_new, gamma_new


## LDA function
def LDA_algorithm(corpus, V, k, tolerance = 1e-4):
  alpha_old, beta_old, eta_old = initialize_parameters_EM(V, k)
  M = len(corpus)

  phi_old, gamma_old, lambda_old = initialize_parameters_VI(alpha_old, corpus, k)

  alpha_sum_old = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))

  lower_bound_old = lower_bound_corpus(alpha_old, beta_old, phi_old, gamma_old, alpha_sum_old, k, corpus)

  start_EM = time.time()
  it = 0
  ########################
  # --- EM-algorithm --- #
  ########################
  while True:
    # print(alpha_old)
    it += 1

    print("\nEM-iteration:", it)

    ##################
    # --- E-step --- #
    ##################
    print("E-step...")
    start = time.time()
    phi_new, gamma_new = [], []

    for (d,document) in enumerate(corpus):
      phi_d, gamma_d = VI_algorithm(k, document, phi_old[d], gamma_old[d], lambda_old, alpha_old, beta_old, eta_old, alpha_sum_old)
      phi_new.append(phi_d); gamma_new.append(gamma_d)
    
    phi_old, gamma_old = phi_new, gamma_new

    
    stop = time.time()
    print('...completed in:', stop - start)
    ##################
    # --- M-step --- #
    ##################
    print("M-step...")
    start = time.time()
    beta_new = calculate_beta(phi_old, corpus, V, k)

    alpha_new = calculate_alpha(gamma_old, alpha_old, M, k)

    alpha_sum_new = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))
    
    stop = time.time()
    print('...completed in:', stop - start)
    ########################
    # --- Convergence? --- #
    ########################

    print("Computing the lower bound...")
    start_n = time.time()
    lower_bound_new = lower_bound_corpus(alpha_new, beta_new, phi_old, gamma_new, alpha_sum_new, k, corpus)
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
      beta_old = beta_new
      lower_bound_old = lower_bound_new
      alpha_sum_old = alpha_sum_new
    
  stop_EM = time.time()
  print('\nThe algorithm converged in', stop_EM - start_EM, "seconds")
  return [alpha_new, beta_new, phi_old, gamma_old]


## Printing all the top-words
def print_top_words_for_all_topics(vocab_file, beta, top_x, k, indices = []):
  '''Used for getting the top_x highest probability words for each topic'''
  if len(indices) == 0:
    indices = list(range(k))

  # Read index-word vocabulary from file
  index_word_vocab = {}
  with open(vocab_file, mode='r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line in reader:
              if line:
                index_word_vocab[int(line[1])] = line[0]
                
  # Get most probable words for each topic
  for topic in indices:
    print('The ', top_x,' most probable words for topic with index: ', topic)
    word_distribution = beta[topic,:]
    indices = list(np.argsort(word_distribution))[-top_x:]
    for word_index in reversed(indices):
      word = index_word_vocab[word_index + 1] # + 1 since the dictionary starts at 1 and the word_indices in LDA starts at 0
      print(word)    


## Get the most likely topics
def most_likely_topics(alpha, num_topics):
  if num_topics > len(alpha):
    print("You chose too many topics, setting it to k")
    num_topics = len(alpha)

  ordered_topics = np.argsort(-alpha)

  return ordered_topics[:num_topics]
  

## The expected probability of a topic
def probability_topics(alpha, topic_indices):
  return alpha[topic_indices] / np.sum(alpha)


## Print function for likely topics
def print_likely_topics(alpha, num_topics):
  topic_indices = most_likely_topics(alpha, 10)

  probabilities = probability_topics(alpha, topic_indices)

  print("\nThe", len(topic_indices), "most likely topics and their expected probabilities are")
  for i in range(len(topic_indices)):
    print("\t", topic_indices[i], "- {}%".format(round(100*probabilities[i],1)))
  
  return topic_indices


## Print parameters
def print_parameters(parameters, printing = True):
  if printing:
    params = ["alpha", "beta", "phi", "gamma"]
  
    for (i,param) in enumerate(params):
      print(param + ":")
      print(parameters[i])


## Main function
def main():
  # Initial parameters
  k = 50              # Number of topics
  num_documents = 100

  # File directories
  vocab_file = './Code/Reuters_Corpus_Vocabulary.csv'
  filename = './Code/Reuters_Corpus_Vectorized.csv'

  # Load data
  corpus, V = load_data(filename, num_documents)

  # Run the algorithm
  parameters = LDA_algorithm(corpus, V, k)

  # Print the parameters
  print_parameters(parameters, True)

  # Print most likely topics and words
  alpha = parameters[0]
  num_topics = 5 # The number of topics that should be printed
  topic_indices = print_likely_topics(alpha, num_topics)
  beta = parameters[1]
  print_top_words_for_all_topics(vocab_file, beta, top_x=15, k=k, indices = topic_indices)


if __name__ == "__main__":
  main()