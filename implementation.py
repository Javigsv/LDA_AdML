import numpy as np
from scipy.special import digamma, polygamma
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

## EM-algorithm for whole corpus
def EM_algorithm(corpus, V, k):
  # Initalize priors (beta, eta, alpha)
  alpha_old, beta_old, eta = initialize_parameters_EM(V, k)

  # Initialize new parameters just to not fullfil convergence criteria
  alpha_new = alpha_old + 100
  beta_new = beta_old + 100

  M = len(corpus)

  it = 0
  while True:
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
    beta_new = calculate_beta3(phis, corpus, V, k)

    alpha_new = calculate_alpha(gammas, alpha_old, M, k)

    if convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new):
      break
    else:
      alpha_old = alpha_new
      beta_old = beta_new


  return alpha_new, beta_new, phis, gammas


## VI-algorithm for a single document in the corpus
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


## Derivative of the digamma function (help-function)
def trigamma(a):
  return polygamma(1, a)


## To calculate beta in the M-step
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


## New version of above function with inspriation from https://github.com/akashii99/Topic-Modelling-with-Latent-Dirichlet-Allocation/blob/master/LDA_blei_implement.ipynb
## Was not able to complete it
def calculate_beta2(phis, coprus, V, k):
  print("Beta computation")
  for j in range (V):
    # Construct w_mn == j of same shape as phi
    w_mnj = [np.tile((document == j),(k,1)).T for document in coprus]
    print(w_mnj)
    beta[j,:] = np.sum(np.array(list(map(lambda x: np.sum(x,axis=0),phis*w_mnj))),axis=0)

  # Normalize across states so beta represents probability of each word given the state
  for i in range(k):
    beta[:,i] = beta[:,i]/sum(beta[:,i])

  return beta


def calculate_beta3(phis, corpus, V, k):
  beta = np.zeros((V, k))
  for j in range(V):
    for m in range(len(corpus)):
      w_hot = np.array([np.tile((word == j), (k,1)).T for word in corpus[m]])
      beta[j,:] += np.sum( w_hot[:,0,:] * phis[m] , axis = 0)

  for i in range(k):
    beta[:,i] = beta[:,i]/sum(beta[:,i])

  return beta.T



## Newton-Raphson function to calculate new alpha in the M-step
def calculate_alpha(gammas, alpha, M, k, nr_max_iterations = 1000, tolerance = 10 ** -4):
  # Use Newton-Raphson method with linear complexity suggested by Thomas P. Minka in
  # Estimating a Dirichlet distribution

  gammas = np.array(gammas)
  for iteration in range(nr_max_iterations):
    alpha_old = alpha

    # Calculate the observed efficient statistic
    # Here we are using that the expected sufficient statistics are equal to the observed sufficient statistics
    # for distributions in the exponential family when the gradient is zero
    log_p_mean = np.sum((digamma(gammas) - np.tile(digamma(np.sum(gammas,axis=1)),(k,1)).T),axis=0)

    # Calculate the gradient
    g = M * (digamma(np.sum(alpha)) - digamma(alpha)) + log_p_mean

    # Calculate the diagonal of the Hessian
    h = -M * trigamma(alpha)

    # Calculate the constant component of the Hessian
    z = M * trigamma(np.sum(alpha))

    # Calculate the constant
    b = np.sum(g/h) / (1/z + np.sum(1/h))

    # Update equation for alpha
    alpha = alpha - (g - b) / h

    if np.linalg.norm(alpha-alpha_old) < tolerance:
      # print("Newton Raphson Converged in", iteration, "steps!")
      break

  return alpha


## Calculate phi for document m
def calculate_phi(gamma, beta, document, k):

  # Length of document m (N_m)
  N = len(document)

  phi = []
  for n in range(N):
    # According to step 6
    phi_n = beta[:, document[n]] * np.exp(digamma(gamma) - digamma(np.sum(gamma)))

    # Normalize phi since it's a probability (must sum up to 1)
    phi_n = phi_n / np.sum(phi_n)

    phi.append(phi_n)

  phi = np.array(phi)

  return phi


## Calculate gamma for document m
def calculate_gamma(phi, alpha, k):
  # According to equation 7 on page 1004

  #gamma = [alpha[i] + np.sum(phi[:,i]) for i in range(k)]
  gamma = alpha + np.sum(phi, axis = 0)

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


## Initialize VI parameters
def initialize_parameters_VI(alpha, N, k):
  # phi = [[1 / k for i in range(k)] for j in range(k)]   # k is the number of priors (from the Dirichlet distribution)
  # gamma = [alpha[i] + N / k for i in range(k)]       # N is the number of words, alpha the priors of the Dirichlet distribution
  phi = np.ones((k,k)) * 1/k
  gamma = alpha.copy() + N / k
  lambd = 1                                             # I don't know
  return phi, gamma, lambd


## Initialize EM parameters
def initialize_parameters_EM(V, k):
  # TODO: This is just an approach, it may be wrong

  # E) I think that we should maybe encode sparcity into each into the Dirichlet. See https://youtu.be/o22cA1DhSMQ?t=1566 for how an alpha < 1 does this.
  """ approx_alpha = 0.01 # 0.1
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)
  input(alpha) """

    # I think alpha is ok
  alpha = np.random.uniform(approx_alpha - 0.1 * approx_alpha, approx_alpha + 0.1 * approx_alpha, k)      

  # Beta should probably be normalized
  beta = np.random.rand(k, V)
  for i in range(k):
    beta[i,:] = beta[i,:] / sum(beta[i,:])

  eta = 1
  return alpha, beta, eta


## Check for convergence in VI-algorithm
def convergence_criteria_VI(phi_old, gamma_old, phi_new, gamma_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.any(np.abs(phi_old - phi_new) > threshold):
    return False

  if np.any(np.abs(gamma_old - gamma_new) > threshold):
    return False

  return True


## Check for convergence in EM-algorithm
def convergence_criteria_EM(alpha_old, beta_old, alpha_new, beta_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.any(np.abs(alpha_old - alpha_new) > threshold):
    print("alpha not converged")
    print(alpha_old)
    print(alpha_new)
    return False

  if np.any(np.abs(beta_old - beta_new) > threshold):
    print("beta not converged")
    print(beta_old)
    print(beta_new)
    return False

  return True


## New version
def convergence_criteria_EM2(alpha_old, beta_old, alpha_new, beta_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong

  if np.linalg.norm(alpha_old - alpha_new) < threshold:
    print("alpha converged")
    if np.linalg.norm(beta_old - beta_new) < threshold:
      print("beta converged")
      return True
    else:
      print("beta not converged")
      print(beta_old)
      print(beta_new)
  else:
    print("alpha not converged")
    print(alpha_old)
    print(alpha_new)

  return False


## New version
def convergence_criteria_EM3(alpha_old, beta_old, alpha_new, beta_new, threshold = 1e-6):
  # Implement convergence criteria

  # TODO: This is just an approach, it may be wrong


  if np.linalg.norm(beta_old - beta_new) < threshold:
    print("beta converged")
    if np.linalg.norm(alpha_old - alpha_new) < threshold:
      print("alpha converged")
      return True
    else:
      print("alpha not converged")
      print(alpha_old)
      print(alpha_new)
  else:
      print("beta not converged")
      print(beta_old)
      print(beta_new)

  return False


## Load data
def load_data(filename, num_documents):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


def main():

  k = 10

  filename = './Code/Reuters_Corpus_Vectorized.csv'

  corpus, V = load_data(filename, 5)

<<<<<<< Updated upstream

  alpha, beta = EM_algorithm(corpus, V, k)
=======
  alpha, beta, phi, gamma = EM_algorithm(corpus, V, k)
>>>>>>> Stashed changes

  # print(alpha, beta)

if __name__ == "__main__":
    main()
