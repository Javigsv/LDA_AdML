import numpy as np
import scipy
from scipy import special
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
  # print(np.sum(beta, axis = 1))
  # print(beta)
  # raise ValueError('A very specific bad thing happened.')
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

    if np.linalg.norm(alphaOld - alphaNew) < tol or np.linalg.norm(betaOld - betaNew) < tol:
      converge =1
    else: 
      converge =0
      alphaOld = alphaNew
      betaOld = betaNew

    
  output.append([phi, gamma, alphaNew, betaNew])
      
  return output

## Load data
def load_data(filename, num_documents):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


k = 5

filename = './Code/Reuters_Corpus_Vectorized.csv'

corpus, V = load_data(filename, 10)

tol = 1e-4

print(LDA(k, V, corpus, tol))