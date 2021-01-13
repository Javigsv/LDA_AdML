import numpy as np
import pickle
from LDAunsmoothed import print_top_words_for_all_topics


## Estimate theta for a document - JOAR
def get_topic_proportions_v1(phis):
    pass

## Estimate theta for a document - EMIL
def get_topic_proportions_v2(gamma):
    input(gamma)
    

## Get topic proportions over time. Joar from this function we will call the functions above
## ... by looping through documents
def get_topics_over_time():
    pass

def main():
    gammas = np.load('gamma2_k50_backup_reuters.npy')
    print(gammas)
    single_test_gamma = gammas[0]
    get_topic_proportions_v2(gamma)
    input(gammas.shape)


def aftermath():
    print('aftermath')
    
    beta = np.load('beta_k50_Guardian.npy')
    with open('phis_k50_Guardian.pkl', 'rb') as infile: # this is how to open the pkl file afterwards
        phis = pickle.load(infile)
    gammamatrix = np.load('gamma_k50_Guardian.npy')
    print(gammamatrix)

    for phi in phis:
        print(phi)

    print(beta)

    vocab_file = './Code/Guardian_Vocab.csv'
    #print_top_words_for_all_topics(vocab_file, beta, top_x=15, k=50)

    with open('phis_k50_Guardian.pkl', 'rb') as infile: # this is how to open the pkl file afterwards
        phis = pickle.load(infile)
    
    for phimatrix in phis:
      print(phimatrix.shape) 

if __name__=='__main__':
    #main()
    aftermath()