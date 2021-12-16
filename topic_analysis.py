import numpy as np
import pickle
from LDAunsmoothed import print_top_words_for_all_topics


## Estimate theta for a document - JOAR
def get_topic_proportions_v1(phis):
    topic_proportions = []

    for document in range(np.shape(phis)[0]):
        topic_proportions.append(np.sum(phis[document],axis = 0)/np.shape(phis[document])[0])
    return topic_proportions

## Estimate theta for a document - EMIL
def get_topic_proportions_v2(gammas):
    '''Computes the expected values of the q(theta | gamma) for each document <=> estimated topic proportions'''
    # TODO -Add moving avg. smoothing / regression splines

    alpha_sum_vector = np.sum(gammas, axis=1)
    exp_topic_proportions = gammas / alpha_sum_vector[:,None]
    print(exp_topic_proportions.shape)
    #print(np.sum(exp_topic_proportions, axis=1))   

    return exp_topic_proportions

## Get topic proportions over time. Joar from this function we will call the functions above
## ... by looping through documents
def get_topics_over_time(gammas, topic_indices, topic_names, timestep = 'month'):
    '''Function to visualize topic evolution over time'''
    
    # Calculate estimated topic proportions
    exp_topic_proportions = get_topic_proportions_v2(gammas) # Emils
    est_topic_proportions = get_topic_proportions_v2(gammas) # Joars

    if timestep=='month':

        pass
    elif timestep=='week':
        pass
    else:
        input('INVALID TIMESTEP')

    # Choose top


    pass

def main():
    # Load computed data
    gammas = np.load('./Official Param Results/gamma_k50_Guardian.npy')
    with open('phis_k50_Guardian.pkl', 'rb') as infile: # this is how to open the pkl file afterwards
        phis = pickle.load(infile)
    
    get_topic_proportions_v2(gammas)





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
    main()
    #aftermath()