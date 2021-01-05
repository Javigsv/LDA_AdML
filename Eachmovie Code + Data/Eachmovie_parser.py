import numpy as np
import pandas as pd
import random

def eachmovie_parser():
    

    movieset = pd.read_csv(r'C:\Users\joar_\OneDrive\Skrivbord\KTH2\DD2434 - Advanced Machine Learning\LDA Project\python\eachmovie_triple.data',  delimiter=" ",header=None)
    movieset_array = np.array(movieset) #shape (2811718,10)
    movie_column = movieset_array[:,3]
    user_column = movieset_array[:,6]
    rating_column = movieset_array[:,9] - 1
    X = np.c_[movie_column,user_column]
    X = np.c_[X,rating_column]
    unique_users = np.unique(X[:,1]) #Finds all the unique user IDs
    unique_movies = np.unique(X[:,0]) #All the unique movies
    movie_vocabulary = {}
    users = {}
    corpus =[]
    for movie in unique_movies:
        movie_vocabulary[str(movie)] = movie
    print("Number of unique reviews: " + str(np.shape(X[:,1])[0]))
    i = 0
    userID = 0
    popped_elements = 0
    for user in unique_users:
        corpus.append([[],[]])
        users[str(user)] = [[],[]]#Key = user, value = list with two rows
        k = True

        while k == True:
            if X[i,1] == user and i < (np.shape(X[:,1])[0]- 1):
                movie = X[i,0]
                users[str(user)][0].append(movie)#movies
                rating = X[i,2]
                users[str(user)][1].append(rating) #rating
                corpus[userID][0].append(movie)
                corpus[userID][1].append(rating)
                i += 1

            else:
                k = False
              
        count = 0
        dictcount = 0
        for rating in corpus[userID][1]:
            if rating >= 3:
                count += 1   

        if count < 100:
            corpus.pop()
            popped_elements += 1
        else:
            userID += 1 


        #Code remnant from when i used a dictionary to contain all user-arrays.      
        for rating in users[str(user)][1]:
            if rating >= 3:
                dictcount += 1

        
        if dictcount < 100: #Removes users with less than 100 positive reviews
            users.pop(str(user))
            #popped_elements += 1

        

    print("Number of users with less than 100 positive reviews: " + str(popped_elements))
    print("Number of users with more than 100 positive reviews: " + str(61265-popped_elements))
    
    return corpus, users

def print_corpus(corpus):
    for i in range(len(corpus)):
        for j in range(len(corpus[i][0])):
            print("User " + str(i) + ": movie " + str(corpus[i][0][j]))
            print("User " + str(i) + ": rating " + str(corpus[i][1][j]))
    
def divide_corpus(corpus): #Code to divide corpus into training and testing. Testing-data should have one unobserved movie.
    random.seed(1337)
    unobserved = []
    observed = corpus
    for i in range(390): #Separates corpus into observed and unobserved sets.
        idx = random.randint(0,len(observed)-1-i)
        temp = observed.pop(idx)
        unobserved.append(temp)

    for i in range(len(unobserved)): #Removes one of the movies and its rating.
        idx = random.randint(0,len(unobserved[i][0])-1)
        unobserved[i][0][idx] = np.NaN
        unobserved[i][1][idx] = np.NaN

    return observed, unobserved

def write_file(observed_corpus,unobserved_corpus):
    f1 = open("Observed_data.txt","w")
    f2 = open("Unobserved_data.txt", "w")
    f1.write("Movie User Rating \n")
    for i in range(len(observed_corpus)):
        for j in range(len(observed_corpus[i][0])):
            f1.write(str(observed_corpus[i][0][j])+ " " + str(i+1) +" "+ str(observed_corpus[i][1][j]) + "\n")
    f1.close()

    f2.write("Movie User Rating \n")
    for i in range(len(unobserved_corpus)):
        for j in range(len(unobserved_corpus[i][0])):
            f2.write(str(unobserved_corpus[i][0][j])+ " " + str(i+1) +" "+ str(unobserved_corpus[i][1][j]) + "\n")
    f2.close()

def write_corpus(corpus):
    
    f1 = open("filtered_corpus.txt","w")
    for i in range(len(corpus)):
        for j in range(len(corpus[i][0])):
            f1.write(str(corpus[i][0][j])+ " " + str(i+1) +" "+ str(corpus[i][1][j]) + "\n")
    f1.close()

def main():
    corpus, users = eachmovie_parser() # Corpus is an array:[([movies 1 to N],[ratings 1 to N]),....]. corpus[i] gives you the ith user-vector
    write_corpus(corpus)
    observed_corpus, unobserved_corpus = divide_corpus(corpus) #observed_corpus is like corpus but with 390 removes user-arrays
    write_file(observed_corpus,unobserved_corpus)                                                          #unobserved_corpus contains 390 user-array, where one movie in each array have been replaced with np.NaN
    



if __name__ == '__main__':
    main()