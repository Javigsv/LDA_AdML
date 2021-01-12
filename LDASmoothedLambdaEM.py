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


## Calculate phi for document m
def calculate_phi(gamma, lambdas, document, k):
  N = len(document)

  phi = []
  for n in range(N):
    # Computing in log-space to avoid underflow
    log_phi_n = digamma(lambdas[:, document[n]]) - digamma(np.sum(lambdas, axis=1)) + digamma(gamma) # - digamma(np.sum(gamma))

    log_phi_n = log_phi_n - scipy.special.logsumexp(log_phi_n)

    phi_n = np.exp(log_phi_n)

    phi.append(phi_n)

    # According to step 6
    # phi_n = np.exp(digamma(lambdas[:, document[n]]) - digamma(np.sum(lambdas, axis=1))) * np.exp(digamma(gamma) - digamma(np.sum(gamma)))
    

    
    # phi_n = beta[:, document[n]] * np.exp(digamma(gamma))
    # print(phi_n)
    # Normalize phi since it's a probability (must sum up to 1)
    # phi_n = phi_n / np.sum(phi_n)
    

    
  
  phi = np.array(phi)
  
  return phi


## Calculate phi for document m
def calculate_phi_debug(gamma, beta, document, k):
  N = len(document)
  print(beta[:, document[:]].T)
  print(np.tile(np.exp(digamma(gamma)),(N,1)))
  print(document[0])
  # According to step 6
  phi = beta[:, document[:]].T * np.tile(np.exp(digamma(gamma)),(N,1))
  # print(phi)
  # Normalize phi since it's a probability (must sum up to 1)
  phi = phi/np.sum(phi, axis = 1)[:, np.newaxis]

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
    # likelihood += np.sum(phi[n,:]*np.log(np.maximum(beta[:,document[n]], 1e-90)))
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


## VI-algorithm run during the E-step for every document m
def VI_algorithm(document, k, V, phi_old, gamma_old, lambdas, alpha, eta, alpha_sum, tolerance = 1e-4, debug = False):
  # document = np.array(doc)
  # print(document)
  lower_bound_old = lower_bound_single(alpha, lambdas, phi_old, gamma_old, alpha_sum, k, document)
  
  # Extended pseudocode from page 1005
  it = 0
  while True:
    it += 1

    # Calculate the new phis
    phi_new = calculate_phi(gamma_old, lambdas, document, k)

    # Calculate the new gammas
    gamma_new = calculate_gamma(phi_new, alpha, k)

    lower_bound_new = lower_bound_single(alpha, lambdas, phi_new, gamma_new, alpha_sum, k, document, debug = False)

    convergence = abs((lower_bound_old-lower_bound_new) / lower_bound_old)

    if convergence < tolerance:
      break
    else:
      phi_old = phi_new
      gamma_old = gamma_new
      lower_bound_old = lower_bound_new

  return phi_new, gamma_new


## LDA function
def LDA_algorithm(corpus, V, k, tolerance = 1e-4):
  alpha_old, eta_old = initialize_parameters_EM(V, k)
  M = len(corpus)

  phi_old, gamma_old, lambda_old = initialize_parameters_VI(alpha_old, eta_old, corpus, k, V)

  alpha_sum_old = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))

  lower_bound_old = lower_bound_corpus(alpha_old, eta_old, lambda_old, phi_old, gamma_old, alpha_sum_old, k, V, corpus)

  print("Starting lower bound:", lower_bound_old)

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

    iteration = 0
    for d in range(len(corpus)):
      if (d) % int(len(corpus) / 10) == 0 and d > 0:
        iteration += 1
        print("\t{}% of documents converged".format(iteration * 10))
      phi_d, gamma_d = VI_algorithm(corpus[d], k, V, phi_old[d], gamma_old[d], lambda_old, alpha_old, eta_old, alpha_sum_old)
      phi_new.append(phi_d); gamma_new.append(gamma_d)
    
    phi_old, gamma_old = phi_new, gamma_new

    lambda_new = calculate_lambda(phi_new, eta_old, corpus, V, k)
    
    stop = time.time()
    print('...completed in:', stop - start)
    ##################
    # --- M-step --- #
    ##################
    print("M-step...")
    start = time.time()
    eta_new = calculate_eta(lambda_old, eta_old, V, k)
    print("\t", eta_new)

    alpha_new = calculate_alpha(gamma_old, alpha_old, M, k)

    alpha_sum_new = loggamma(np.sum(alpha_old)) - np.sum(loggamma(alpha_old))
    
    stop = time.time()
    print('...completed in:', stop - start)
    ########################
    # --- Convergence? --- #
    ########################
    print("Eta:", eta_new)
    print("Computing the lower bound...")
    start_n = time.time()
    lower_bound_new = lower_bound_corpus(alpha_new, eta_new, lambda_new, phi_new, gamma_new, alpha_sum_new, k, V, corpus)
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
      lambda_old = lambda_new
      lower_bound_old = lower_bound_new
      alpha_sum_old = alpha_sum_new
    
  stop_EM = time.time()
  print('\nThe algorithm converged in', stop_EM - start_EM, "seconds")
  return [alpha_new, eta_new, lambda_new, phi_old, gamma_old]


## Computing the perplexity for a given corpus
def perplexity(alpha, beta, phi, gamma, alpha_sum, k, corpus):
  
  L = lower_bound_corpus(alpha, beta, phi, gamma, alpha_sum, k, corpus)

  N = 0
  for doc in corpus:
    N += len(doc)

  return np.exp(-L/N)


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
  k = 10             # Number of topics
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
  # print_perplexity(alpha, beta, phi, gamma, k, corpus, test)

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