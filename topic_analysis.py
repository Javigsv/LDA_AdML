import numpy as np

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

if __name__=='__main__':
    main()