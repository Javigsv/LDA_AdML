from mixture_of_unigram_model import mixture_of_unigram
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from DataLoader import DataLoader
import csv
import sys

log = open("mixture.log", "a")
sys.stdout = log

np.set_printoptions(threshold=sys.maxsize)


def load_data(filename, num_documents = 10**6):

  data_loader = DataLoader(filename)
  data, V = data_loader.load(num_documents)

  return data, V


def ocurrences(corpus, V):
  M = len(corpus)

  x = np.zeros((M,V))

  for m, doc in enumerate(corpus):
    for word in doc:
      x[m, word] += 1

  return x

def get_words_from_indexes(indices):
  # Read index-word vocabulary from file
  index_word_vocab = {}
  with open(vocab_file, mode='r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for line in reader:
      if line:
        index_word_vocab[int(line[1])] = line[0]

  words = []
  for ind in indices:
    word = index_word_vocab[ind]
    words.append(word)

  return words


# Initial parameters
num_documents =  2000
laplace_smoothing = 2
k = 15

# File directories
vocab_file = './Code/Reuters_Corpus_Vocabulary.csv'
filename = './Code/Reuters_Corpus_Vectorized.csv'

# Load data
corpus, V = load_data(filename, num_documents)
nTraining = len(corpus) - int(len(corpus)*0.1)
test = corpus[nTraining:]
corpus = corpus[:nTraining]

unlab_data = ocurrences(corpus, V)
unlab_data_test = ocurrences(test, V)

print(unlab_data.shape[0], 'documents in the training set')
print(unlab_data_test.shape[0], 'documents in the test set')


# Model initialization
model=mixture_of_unigram(range(0,V),topic_num=k,fix_labeled_doc=False,alpha=laplace_smoothing)


# Adding unlabeled data 10 x V to the model and running 1 iteration to initialize parameters
model.add_unlabeled_doc(unlab_data[:10,:])
model.train(iteration=1)


# Adding the rest of the data M-10 x V to the model and running more iterations
model.add_unlabeled_doc(unlab_data[10:,:])
nIt = 20
model.train(iteration=nIt)

#use trained model to predict which topic do new documents belong to as below.
#v_test_doc is also a 2d array specifying the count of each word in each document. It is of size #_of_documents_in_test_doc*len(show_word).
print("test predict topic\n",model.predict(unlab_data_test).argmax(axis=1))

#if you want to get a raw distribution over topic for each document for further analysis.
print("test predict topic distribution\n",model.predict(unlab_data_test))


#show nWords words that best represent each topic.
print("topic\n",model.get_topic(show_word=10))

nWords = 10
topic_indices = model.get_topic(show_word=nWords)

for topic, ind in enumerate(topic_indices):
  topic_words = get_words_from_indexes(ind)
  print('The', nWords, 'most probable words for topic with index:', topic)
  for word in topic_words:
    print(word)
  print()



#you can also access the distribution over topic of the documents in the model.
# print("labeled data topic distribution\n",model.labeled_doc2topic)
print("unlabeled data topic distribution\n",model.doc2topic) #this one for unlabeled document.

#get the topic these document belongs to is the same.
#print("labeled data topic\n",model.labeled_doc2topic.argmax(axis=1))
print("unlabeled data topic\n",model.doc2topic.argmax(axis=1)) #this one for unlabeled document.


print(model.perplexity(test))
exit()

