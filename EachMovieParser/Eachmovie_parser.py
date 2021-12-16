import numpy as np
import pandas as pd
import random

def eachmovie_parser(filename, training_size, test_size):
    
    datafile = './' + filename
    movieset = pd.read_csv(datafile,  delimiter=" ",header=None, dtype= int)
    movieset_array = np.array(movieset) #shape (2811718,10)
    movie_column = movieset_array[:,0]
    user_column = movieset_array[:,1]
    rating_column = movieset_array[:,2]
    unique_users = np.unique(user_column)#(X[:,1]) #Finds all the unique user IDs
    unique_movies = np.unique(movie_column)#(X[:,0]) #All the unique movies
    #--------------------FIXED GAPS IN MOVIE INDEXES---------------------------------------------------
    index_dict = {}
    for i in range(len(unique_movies)):
        index_dict[str(unique_movies[i])] = i
    for i in range(len(movie_column)):
        temporary = index_dict[str(movie_column[i])]
        movie_column[i] = temporary
    #------------------------------------------------------------------------------------------------

    X = np.c_[movie_column,user_column]
    X = np.c_[X,rating_column]
    
    movie_vocabulary = {}
    users = {}
    tempcorpus =[]
    corpus = []
    for movie in unique_movies:
        movie_vocabulary[str(movie)] = movie
    print("Number of unique reviews: " + str(np.shape(X[:,1])[0]))
    i = 0
    userID = 0
    popped_elements = 0
    for user in unique_users:
        tempcorpus.append([[],[]])
        users[str(user)] = [[],[]]#Key = user, value = list with two rows
        k = True

        while k == True:
            if X[i,1] == user and i < (np.shape(X[:,1])[0]- 1):
                movie = X[i,0]
                #users[str(user)][0].append(movie)#movies
                rating = X[i,2]
                #users[str(user)][1].append(rating) #rating
                if rating >= 3:
                    tempcorpus[userID][0].append(movie)
                    tempcorpus[userID][1].append(rating)
                i += 1

            else:
                k = False
              
        count = 0
        for rating in tempcorpus[userID][1]:
            if rating >= 3:
                count += 1   

        if count < 100:
            tempcorpus.pop()
            popped_elements += 1
        else:
            userID += 1         

    print("Number of users with less than 100 positive reviews: " + str(popped_elements))
    print("Number of users with more than 100 positive reviews: " + str(61265-popped_elements))
    for user in range(training_size + test_size):
        corpus.append([])
        corpus[user] = tempcorpus[user][0]
    #---------------Count unique movies ----------------------------
    vocabulary = []
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            vocabulary.append(corpus[i][j])
    V = len(np.unique(vocabulary))
    #---------------Count unique movies ----------------------------

    #--------------------FIXED GAPS IN MOVIE INDEXES---------------------------------------------------
    vocabulary_dict = {}
    for i in range(V):
        vocabulary_dict[str(np.unique(vocabulary)[i])] = i
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            temporary = vocabulary_dict[str(corpus[i][j])]
            corpus[i][j] = temporary
    #------------------------------------------------------------------------------------------------

    return tempcorpus, corpus, users, V

def print_corpus(corpus):
    for i in range(len(corpus)):
        for j in range(len(corpus[i][0])):
            print("User " + str(i) + ": movie " + str(corpus[i][0][j]))
            print("User " + str(i) + ": rating " + str(corpus[i][1][j]))
    
def divide_corpus(corpus, training_size, test_size): #Code to divide corpus into training and testing. Testing-data should have one unobserved movie.
    random.seed(1337)
    test_data = []
    training_data = corpus
    removed_movies = []
    for i in range(test_size): #Separates corpus into observed and unobserved sets.
        idx = random.randint(0,len(training_data)-1-i)
        temp = training_data.pop(idx)
        test_data.append(temp)

    for i in range(len(test_data)): #Removes one of the movies and its rating.
        
        removed_movie = test_data[i].pop()
        removed_movies.append(removed_movie)
    return training_data,test_data,removed_movies

def write_corpus(corpus):
    
    f1 = open("filtered_corpus.txt","w")
    for i in range(len(corpus)):
        for j in range(len(corpus[i][0])):
            f1.write(str(corpus[i][0][j])+ " " + str(i+1) +" "+ str(corpus[i][1][j]) + "\n")
    f1.close()

def dataloaderEachmovie(filename, training_size,test_size):

    '''--------------------------------------------'''
    '''Max sum of training_size + test_size = 3667!'''
    '''--------------------------------------------'''
    
    filtered_corpus, corpus, users, V = eachmovie_parser(filename,training_size, test_size) # Corpus is an array:[([movies 1 to N],[ratings 1 to N]),....]. corpus[i] gives you the ith user-vector
    #write_corpus(filtered_corpus)
    training_data, test_data, removed_movies = divide_corpus(corpus,training_size,test_size) #training_data is like corpus but with 390 removes user-arrays
                                                               #test_data contains 390 user-array, where one movie in each array have been replaced with np.NaN
    return training_data, test_data, removed_movies, V



if __name__ == '__main__':
    filename =  "filtered_corpus.txt"
    training_size = 3277
    test_size = 390
    dataloaderEachmovie(filename, training_size, test_size)
    print("EachMovieParser done!")