import numpy as np
from scipy.special import digamma, polygamma
from scipy.special import gamma as gamma_function
from scipy.special import loggamma as loggamma
import scipy
from DataLoader import DataLoader
import time, csv
from datetime import datetime
import EachMovieParser.Eachmovie_parser as eachmovie



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
def load_data(filename, num_documents = 10**6):

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

  approx_alpha = 0.00001
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)      

  # alpha = np.full(shape = k, fill_value = 50/k)

  approx_eta = 0.0001
  eta = np.random.uniform(approx_eta - 0.1 * approx_eta, approx_eta + 0.1 * approx_eta) 

  return alpha, eta


## Calculate phi for document m
def calculate_phi(gamma, lambdas, document, k):
  N = len(document)

  phi = []
  for n in range(N):
    # Computing in log-space to avoid underflow
    log_phi_n = digamma(lambdas[:, document[n]]) - digamma(np.sum(lambdas, axis=1)) + digamma(gamma) # - digamma(np.sum(gamma))

    # Normalize
    log_phi_n = log_phi_n - scipy.special.logsumexp(log_phi_n)

    # Back to normal space
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


## Calculate eta
def calculate_eta(lambdas, eta, V, k, nr_max_iterations = 1000, tolerance = 10 ** -4):
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


## Calculate eta in log-space
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


## Calculate the lower bound
def calculate_lower_bound(alpha, eta, lambdas, phi, gamma, k, V, corpus):
  M = len(corpus)

  likelihood = 0

  for d in range(M):
    likelihood += calc_lower_bound(alpha, eta, lambdas, phi[d], gamma[d], k, V, corpus[d])

  likelihood += - np.sum(loggamma(np.sum(lambdas, axis = 1))) + np.sum(loggamma(lambdas)) + np.sum((eta - lambdas) * (digamma(lambdas) - np.tile(digamma(np.sum(lambdas, axis = 1)),(V,1)).T))

  likelihood += k * (loggamma(V * eta) - V * loggamma(eta)) + M * (loggamma(np.sum(alpha)) - np.sum(loggamma(alpha)))

  return likelihood


## Terms in lower bound for every document
def calc_lower_bound(alpha, eta, lambdas, phi, gamma, k, V, document):
  digamma_gamma_sum = digamma(np.sum(gamma))

  N = len(document)
  # print((digamma(lambdas[:,document]) - np.tile(digamma(np.sum(lambdas, axis = 1)),(N,1)).T))
  # Phis
  likelihood = np.sum(phi * (np.tile((digamma(gamma) - digamma_gamma_sum), (N,1)) + (digamma(lambdas[:,document]).T - np.tile(digamma(np.sum(lambdas, axis = 1)),(N,1))))) + np.sum(np.log(phi ** phi))

  # Gammas
  likelihood += -loggamma(np.sum(gamma)) + np.sum(loggamma(gamma)) + np.sum((alpha-gamma)*(digamma(gamma) - digamma_gamma_sum))

  return likelihood


## VI-algorithm run during the E-step for every document m
def VI_algorithm(document, k, V, lambdas, alpha, eta, gamma_old, tolerance = 1e-4, debug = False):
  # gamma_old = np.ones(k)
  
  # Extended pseudocode from page 1005
  it = 0
  while True:
    it += 1

    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, lambdas, document, k)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha, k)

    convergence = np.linalg.norm(gamma_new-gamma_old) / k

    if convergence < tolerance:
      break
    else:
      gamma_old = gamma_new
  
  return phi_new, gamma_new


## Running inner EM to calculate lambda
def EM_algorithm_lambda(corpus, k, V, alpha, eta, lambda_old, gamma_old, tolerance = 1e-4):
  lower_bound_old = -1e+40

  M = len(corpus)

  # lambda_old = 0.01 * np.random.rand(k,V)

  it = 0
  while True:
    it += 1
    print("\n\tEM-iteration (inner):", it)
    print("\tE-step (inner)...")
    start = time.time()
    phi_new, gamma_new = [], []

    for d in range(M):
      phi_d, gamma_d = VI_algorithm(corpus[d], k, V, lambda_old, alpha, eta, gamma_old[d])
      phi_new.append(phi_d); gamma_new.append(gamma_d)

    stop = time.time()
    print('\t...completed in:', stop - start)

    print("\tM-step (inner)...")
    start = time.time()
    lambda_new = calculate_lambda(phi_new, eta, corpus, V, k)
    stop = time.time()
    print('\t...completed in:', stop - start)

    print('\tComputing lower bound...')
    start = time.time()
    lower_bound_new = calculate_lower_bound(alpha, eta, lambda_new, phi_new, gamma_new, k, V, corpus)
    stop = time.time()
    print('\t\t', lower_bound_new)
    print('\t...completed in:', stop - start)

    convergence = abs((lower_bound_new-lower_bound_old) / lower_bound_old)
    # print(convergence)
    if convergence < tolerance:
      break
    else:
      lower_bound_old = lower_bound_new
      lambda_old = lambda_new
      gamma_old = gamma_new

  return phi_new, gamma_new, lambda_new


## LDA function
def LDA_algorithm(corpus, V, k, tolerance = 1e-4):
  alpha_old, eta_old = initialize_parameters_EM(V, k)
  lambda_old = np.random.rand(k,V)
  gamma_old = [np.ones(k) for i in range(len(corpus))]

  M = len(corpus)

  lower_bound_old = -1e+40

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
    print("E-step (outer)...")
    start = time.time()

    phi_new, gamma_new, lambda_new = EM_algorithm_lambda(corpus, k, V, alpha_old, eta_old, lambda_old, gamma_old)
    
    stop = time.time()
    print('\n...completed in:', stop - start)
    ##################
    # --- M-step --- #
    ##################
    print("M-step (outer)...")
    start = time.time()
    eta_new = calculate_eta(lambda_new, eta_old, V, k)

    alpha_new = calculate_alpha(gamma_new, alpha_old, M, k)

    
    stop = time.time()
    print('...completed in:', stop - start)
    ########################
    # --- Convergence? --- #
    ########################
    # print("Eta:", eta_new)
    print("Computing the lower bound...")
    start_n = time.time()
    lower_bound_new = calculate_lower_bound(alpha_new, eta_new, lambda_new, phi_new, gamma_new, k, V, corpus)
    print("\t", lower_bound_new)
    stop_n = time.time()
    print('...completed in:', stop_n - start_n)

    if lower_bound_new < lower_bound_old:
      print("Oh no! The lower bound decreased...")

    # The change of the lower bound
    delta_lower_bound = abs((lower_bound_old-lower_bound_new) / lower_bound_old)
    print("The lower bound changed: {}%".format(np.round(100*delta_lower_bound, 5)))
    print("-----------------------------------------------------")
    if delta_lower_bound < tolerance:
      print("Convergence after", it, "iterations!")
      break

    else:
      alpha_old = alpha_new
      eta_old = eta_new
      lower_bound_old = lower_bound_new
      lambda_old = lambda_new
      gamma_old = gamma_new
    
  stop_EM = time.time()
  print('\nThe algorithm converged in', stop_EM - start_EM, "seconds")
  print('\nM =', M)
  print('k = ', k, '\n')
  return [alpha_new, eta_new, lambda_new, phi_new, gamma_new]


## Computing the perplexity for a given corpus
def perplexity(alpha, beta, phi, gamma, alpha_sum, k, corpus):
  
  L = lower_bound_corpus(alpha, beta, phi, gamma, alpha_sum, k, corpus)

  N = 0
  for doc in corpus:
    N += len(doc)

  return np.exp(-L/N)


## Printing all the top-words
def print_top_words_for_all_topics(vocab_file, lambas, top_x, k, indices = []):
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
    word_distribution = lambas[topic,:]
    word_distribution = word_distribution / np.sum(word_distribution)
    indices = list(np.argsort(word_distribution))[-top_x:]
    for word_index in reversed(indices):
      word = index_word_vocab[word_index + 1] # + 1 since the dictionary starts at 1 and the word_indices in LDA starts at 0
      print("\t", word, "- {}%".format(round(100*word_distribution[word_index],3)))    


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


## Smooth beta, adding pseudocount
def smooth_beta(beta):
  V = beta.shape[1]
  k = beta.shape[0]

  num_smoothings = 0
  for j in range(V):
    if np.sum(beta[:,j]) == 0:
      num_smoothings += 1
      beta[:,j] = np.ones(k)

  print(num_smoothings, "smoothings were needed!")

  return beta


## Print perplexity
def print_perplexity(alpha, beta, phi, gamma, k, training, test):

  alpha_sum = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha))

  beta = smooth_beta(beta)

  print(phi[0].shape)
  print('\nPerplexity of the training set (', len(training), ' documents ):')
  training_perp = perplexity(alpha, beta, phi, gamma, alpha_sum, k, training)
  print(training_perp)

  print('\nPerplexity of the test set (', len(test), ' documents ):')
  phis, gammas, lambdas = initialize_parameters_VI(alpha, test, k)
  print(alpha.shape, beta.shape)
  for d in range(len(test)):
    phi_new, gamma_new = VI_algorithm(k, test[d], phis[d], gammas[d], lambdas, alpha, beta, 0, alpha_sum, tolerance=1e-6, debug = False)
    phis[d], gammas[d] = phi_new, gamma_new

  test_perp = perplexity(alpha, beta, phis, gammas, alpha_sum, k, test)
  print(test_perp)


## printing predictive perplexity
def print_predictive_perplexity(alpha, beta, test, test_removed):
  k = len(alpha)

  alpha_sum = loggamma(np.sum(alpha)) - np.sum(loggamma(alpha))

  phis, gammas, lambdas = initialize_parameters_VI(alpha, test, k)

  beta = smooth_beta(beta)

  for d in range(len(test)):
    phi_new, gamma_new = VI_algorithm(k, test[d], phis[d], gammas[d], lambdas, alpha, beta, 0, alpha_sum, tolerance=1e-6, debug = False)
    phis[d], gammas[d] = phi_new, gamma_new
  
  perplexity = predictive_perplexity(beta, gammas, test_removed)
  print("The predictive perplexity is...")
  print("\t", perplexity)


## Calculating predictive perplexity
def predictive_perplexity(beta, gamma, test_removed):
  exponent = 0

  # for (user, word) in enumerate(test_removed):
  #   c = loggamma(np.sum(gamma[user])) - np.sum(loggamma(gamma[user])) - np.sum(np.log(gamma[user]))
  #   # print(beta[:, word] * (gamma[user] / (1 + gamma[user])))
  #   print(c)
  #   print(gamma[user])
  #   f = np.log(np.sum(beta[:, word] * (gamma[user] / (1 + gamma[user]))))
  #   # print(f)
  #   exponent += c + f
  # print(exponent)
  # perplexity = np.exp(- exponent / len(test_removed))

  for (user, word) in enumerate(test_removed):
    exponent += np.log(np.sum(beta[:, word] * gamma[user] / np.sum(gamma[user])))
  
  perplexity = np.exp(- exponent / len(test_removed))

  return perplexity


## Main function reuters
def main_Reuters():
  # Initial parameters
  k = 20             # Number of topics
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
  [alpha, eta, lambdas, phi, gamma] = LDA_algorithm(corpus, V, k)

  # Print the parameters
  # print_parameters(parameters, False)

  # Print most likely topics and words
  num_topics = 5 # The number of topics that should be printed
  topic_indices = print_likely_topics(alpha, num_topics)
  print_top_words_for_all_topics(vocab_file, lambdas, top_x=15, k=k, indices = topic_indices)

  # print(len(corpus), len(test))
  # print_perplexity(alpha, eta, lambdas, phi, gamma, k, corpus, test)

  # print()


## Main function eachmovie
def main_each_movie():
  # Initial parameters
  k = 10             # Number of topics
  num_users = 3200
  num_test_users = 390

  # File directories
  filename = "filtered_corpus.txt"

  # Load data
  training_data, test_data, removed_movies, V = eachmovie.dataloaderEachmovie(filename, num_users, num_test_users)
  print("\nNumber of users in training data:", len(training_data))
  print("Number of users in training data:", len(test_data))
  print("Number of unique reviewed movies:", V)

  # Run the algorithm
  parameters = LDA_algorithm(training_data, V, k, 1e-4)

  # Print the parameters
  print_parameters(parameters, False)

  alpha, beta = parameters[0], parameters[1]

  print_predictive_perplexity(alpha, beta, test_data, removed_movies)  

  # phi = parameters[2]
  # gamma = parameters[3]
  # print(len(corpus), len(test))
  # print_perplexity(alpha, beta, phi, gamma, k, corpus, test)


main_Reuters()


# main_each_movie()