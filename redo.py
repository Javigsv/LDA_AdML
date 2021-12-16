import numpy as np
import scipy
from scipy import special
from scipy.special import gamma as gamma_function
from scipy.special import digamma, polygamma
from DataLoader import DataLoader

def Estep(k, d, alpha, beta, corpusMatrix, tol):    
    
  #storing the total number of words and the number of unique words
  document = corpus[d]
  N = len(document)
  
  #initialize phi and gamma
  oldPhi  = np.full(shape = (N,k), fill_value = 1/k)
  gamma = alpha + N/k
  newPhi = oldPhi
  converge = 0 
  
  
  count = 0
  
  while converge == 0:
    # print("Iteration", count+1, "of VI")
    newPhi  = np.zeros(shape = (N,k))
    for n in range(0, N):
      for i in range(0,k):
          newPhi[n,i] = (beta[i, document[n]])*np.exp(scipy.special.psi(gamma[i]) - scipy.special.psi(np.sum(gamma)))
    newPhi = newPhi/np.sum(newPhi, axis = 1)[:, np.newaxis] #normalizing the rows of new phi

    # print(newPhi.shape)

    for i in range(0,k):
      gamma[i] = alpha[i] + np.sum(newPhi[:, i]) #updating gamma

    # print(newPhi)
    criteria = (1/(N*k)*np.sum((newPhi - oldPhi)**2))**0.5
    if criteria < tol:
      converge = 1
    else:
      oldPhi = newPhi
      count = count +1
      converge = 0
  return (newPhi, gamma)

def calculate_alpha(gamma, alphaOld, M, k, nr_max_iterations = 1000, tol = 10 ** -4):
  h = np.zeros(k)
  g = np.zeros(k)
  alphaNew = np.zeros(k)

  converge = 0
  while converge == 0:
    for i in range(0, k):
      docSum = 0
      for d in range(0, M):
        docSum += digamma(gamma[d][i]) - digamma(np.sum(gamma[d]))
      # print(docSum)
      g[i] = M*(digamma(sum(alphaOld)) - digamma(alphaOld[i])) + docSum
      h[i] = M*trigamma(alphaOld[i])
    z =  M*trigamma(np.sum(alphaOld))
    c = np.sum(g/h)/(1/z + np.sum(1/h))
    step = (g - c)/h
    # print(step)
    alphaNew = alphaOld +step
    # if np.linalg.norm(step) < tol:
    if np.linalg.norm(alphaNew-alphaOld) < tol:
      
      converge = 1
    else:
      converge = 0
      alphaOld = alphaNew

  # print(alphaNew)
  # raise ValueError('A very specific bad thing happened.')
  return alphaNew

#Update alpha using linear Newton-Rhapson Method#
def alphaUpdate(k, M, alphaOld, gamma, tol):
  h = np.zeros(k)
  g = np.zeros(k)
  alphaNew = np.zeros(k)

  converge = 0
  while converge == 0:
    for i in range(0, k):
      docSum = 0
      for d in range(0, M):
        docSum += scipy.special.psi(gamma[d][i]) - scipy.special.psi(np.sum(gamma[d]))
      g[i] = M*(scipy.special.psi(sum(alphaOld)) - scipy.special.psi(alphaOld[i])) + docSum
      h[i] = M*scipy.special.polygamma(1, alphaOld[i])
    z =  -scipy.special.polygamma(1, np.sum(alphaOld))
    c = np.sum(g/h)/(1/z + np.sum(1/h))
    step = (g - c)/h
    alphaNew = alphaOld +step
    if np.linalg.norm(step) < tol:
      converge = 1
    else:
      converge = 0
      alphaOld = alphaNew
  print(alphaNew)
  return alphaNew

def Mstep(k, V, M, phi, gamma, alphaOld, corpusMatrix, tol):
  #Calculate beta#

  beta = np.zeros(shape = (k,V))

  for i in range(0,k):
    for j in range(0,V):
      wordSum = 0
      for d in range(0,M):
        Nd = len(corpus[d])
        for n in range(Nd):
          if corpus[d][n] == j:
            wordSum += phi[d][n,i]
      beta[i,j] = wordSum
  #Normalize the rows of beta#
  beta = beta/np.sum(beta, axis = 1)[:, np.newaxis]

  ##Update ALPHA##
  alphaNew = alphaUpdate(k, M, alphaOld, gamma, tol)
  return(alphaNew, beta)

def LDA(k, V, corpus, tol):

    
  ##Check for proper input##
  if isinstance(k, int) != True or k <= 1:
    print("Number of topics must be a positive integer greater than 1")
    return
  
  if tol <=0:
    print("Convergence tolerance must be positive")
    return
  
  
  M = len(corpus)
  output = []
  
  converge = 0

  np.random.seed(1)
  #initialize alpha and beta for first iteration
  #alphaOld = 10*np.random.rand(k)
  alphaOld = np.full(shape = k, fill_value = 50/k)

  betaOld = np.random.rand(k, V)
  betaOld = betaOld/np.sum(betaOld, axis = 1)[:, np.newaxis]
  
  while converge == 0:
    print(alphaOld)
    phi = []
    gamma = []
    #looping through the number of documents
    print("E-step")
    for d in range(0,M): #M is the number of documents
      # print("Document:", d)
      phiT, gammaT = Estep(k, d, alphaOld, betaOld, corpus, tol)
      phi.append(phiT)
      gamma.append(gammaT)
        
    print("M-step")
    alphaNew, betaNew = Mstep(k, V, M, phi, gamma, alphaOld, corpus, tol)

    print(lower_bound(alphaNew, betaNew, phi, gamma, k, V, corpus))

    if np.linalg.norm(alphaOld - alphaNew) < tol or np.linalg.norm(betaOld - betaNew) < tol:
      converge =1
    else: 
      converge =0
      alphaOld = alphaNew
      betaOld = betaNew

    
  output.append([phi, gamma, alphaNew, betaNew])
      
  return output


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



## Load data
def load_data(filename, num_documents):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


k = 3

filename = './Code/Reuters_Corpus_Vectorized.csv'

corpus, V = load_data(filename, 7)

tol = 1e-4

output = LDA(k, V, corpus, tol)

print(output)