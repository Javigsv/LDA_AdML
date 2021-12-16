

import numpy as np
import scipy
from DataLoader import DataLoader



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


def unigram_algorithm(corpus, V, laplace_smoothing = 0):

    beta = np.zeros(V)

    for doc in corpus:
        for word in doc:
            beta[word] +=1

    N = np.sum(beta)

    smoothed_beta = (beta + laplace_smoothing)/(N + laplace_smoothing*V) 

    return smoothed_beta
    

def log_prob_of_doc(doc, beta):

  log_prob = np.sum(np.log([beta[word] for word in doc]))

  return log_prob


def perplexity(beta, corpus):

    NTotal = np.sum([len(doc) for doc in corpus])
    pTotal = np.sum([log_prob_of_doc(doc, beta) for doc in corpus])

    return np.exp(-pTotal/NTotal)


def smoothing(beta):
  
  smoothed = beta + 1/beta.size
  smoothed = smoothed / np.sum(smoothed)

  return smoothed

## Main function
def main():
  # Initial parameters
  num_documents =  10**6
  laplace_smoothing = 1

  # File directories
  vocab_file = './Code/Guardian_Vocabulary.csv'
  filename = './Code/Guardian_Vectorized.csv'

  # Load data
  corpus, V = load_data(filename, num_documents)
  nTraining = len(corpus) - int(len(corpus)*0.1)
  test = corpus[nTraining:]
  corpus = corpus[:nTraining]

  # Run the algorithm
  beta = unigram_algorithm(corpus, V, laplace_smoothing)

  # Print the parameters
  print('beta:', beta)

  # Smoothing
  print()
  if (np.any(beta==0)):
    print(np.count_nonzero(beta==0),'standard smoothings were needed')
    print()
    beta = smoothing(beta)
  if not laplace_smoothing == 0:
    print(laplace_smoothing, 'alpha for Laplace smoothing used')
    print()

  print('Perplexity of the training set (', len(corpus),'documents ):')
  perp =  perplexity(beta, corpus)
  print(perp)

  print('Perplexity of the test set (', len(test),'documents ):')
  perp = perplexity(beta, test)
  print(perp)

  print()


if __name__ == "__main__":
  main()