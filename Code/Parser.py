'''Parser for turning each SGML document in the Reuters-21578 dataset into a vector of integers'''

from bs4 import BeautifulSoup as bs4
import os
import nltk 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re 
import csv


class ReutersCorpusPreprocessor():
    '''Class for preprocessing ONLY the Reuters 21578 corpus. Some of the methods are specifically adapted for the Reuters 21578 dataset
    but the same base could potentially be used for other Corpora. Used for both vectorizing each document in the corpus (and creating an accompanying 
    vocabulary) and a single test for seeing that the vectorization was in fact correct'''


    def __init__(self):
        self.document_counter = 0   # counts the amount of documents iterated when vectorizing/testing
        self.M = 0  # no of documents in corpus
        self.N = 0  # no of total words in corpus (not unique words, unique_words be found in the size of corpus word_index_vocab)
        self.word_index_vocab = {}  # maps from a word to its index for all unique words in the corpus
        self.unique_words_counter = 1   # Used to index through 
        self.csv_writer_vectorized_documents = None    # stores the csv file object for the vectorized documents. For writing the word_index_vocabulary no instance variable is needed

        # These are for testing purposes only
        self.index_word_vocab = {}  # Maps from index to word
        self.corpus_file_generator = None # A generator for feeding each vectorized document in the vectorized file


    def process_word_1(self, word):
        '''Function for processing an indivudual word. Can be used after filtering out certain words'''
        
        # Handle numbers. Right now we treat all numbers (years, prices, large numbers, small numbers) the same, but this could be changed
        contains_number = re.search(".*\d.*", word)
        if contains_number:     
            processed_word = '__NuMBeR'
            return processed_word

        processed_word = word.lower()   # For now we just .lower() non-numbers. We could add a Lemmatizer or Stemmer here if we want to

        return processed_word


    def word_filter_1(self, word):
        '''Function for filtering out certain words (e.g. stopwords). Note that this function only returns True/False. In other words it only chooses which words to discard and does no preprocessing of its own'''

        # Lowercase for comparison
        word = word.lower()

        # Catch words that are 100% non-alphanumerical
        word = ' ' + word + ' ' # this probably isn't the optimal way to do it, but works for a single run
        fully_non_alpha_num = re.search(" \W+ ", word)
        if fully_non_alpha_num:
            return False
        word = word[1:-1]

        # Catch Stopwords
        stopWords = set(stopwords.words('english'))     # OBS: Now we use 150+ stopwords, use slice to get the first 50 [0:50]    
        stopWords.add('reuter'); stopWords.add('reuters'); stopWords.add("'s"); stopWords.add("'ve")
        
        #print('-' + word + '-')
        return word not in stopWords


    def write_word_index_vocab(self, filename):
        '''Function for writing a csv document with the word-index pairs'''

        with open(filename + '.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            for key, value in self.word_index_vocab.items():
                csv_writer.writerow([key, value])


    def vectorize_document(self, wordlist):
        '''Turns a preprocessed document into a vector (here a list where each word is represented with an index)'''
        document_vector = []
        
        for word in wordlist:
            try: 
                index = self.word_index_vocab[word]
                document_vector.append(index)
            except KeyError:
                index = self.unique_words_counter
                self.unique_words_counter += 1
                self.word_index_vocab[word] = index
                document_vector.append(index)

        self.csv_writer.writerow(document_vector)


    def preprocess_reuters_sgml(self, abs_filepath, testing=False):    
        '''Preprocessing function for a single sgml file from the reuters 21578 dataset'''

        with open(abs_filepath) as filehandle:
            soup = bs4(filehandle, 'lxml')

            self.flag = False   # temporary

            for article in soup.find_all(modApte_training_samples_screened):
            
                #print(article.prettify(), '\n')
                text_tag = article.find('text')     # OBS: must use .find() here since the tag's name is "text"
                #title = text_tag.title.string      # should title be included or only bodytext?
                #print(title)
                body = text_tag.find_all(text=True, recursive=False)    # this is apparently how you do it with bs4. It returns a list with "\n" for tags and a non-empty string for the actual text
                #print(body)
                body_string = ' '.join(body)
                #input(body_string)
                body_string = body_string.replace('/',' ')
                sentences = [nltk.word_tokenize(t) for t in nltk.sent_tokenize(body_string)]
                
                # Filter and process the words in the document
                final_word_list = [self.process_word_1(word) 
                                    for sentence in sentences 
                                        for word in sentence
                                            if self.word_filter_1(word)]              

                if len(final_word_list) == 0:
                    break

                if not testing: # if we are creating the vectorized file and vocabulary
                    #input(' '.join(final_word_list))
                    self.vectorize_document(final_word_list)
                    self.document_counter += 1
                    
                else:   # if we are iterating through to test
                    line_in_indices = next(self.corpus_file_generator)
                    line_in_words = [self.index_word_vocab[i] for i in line_in_indices]
                    #final_word_list.append('electroPOPO')
                    #line_in_words.append('olga')
                    
                    documents_are_equal = all((line_in_words[i] == final_word_list[i] for i in range(len(final_word_list)))) and len(line_in_words) == len(final_word_list) # second clause to catch empty documents

                    """ print('\n')
                    
                    for i in range(len(final_word_list)):
                        print(line_in_words[i], final_word_list[i])

                    print('\n')
                    print(documents_are_equal)
                    print(final_word_list)
                    print(line_in_words)
                    input() """

                    if documents_are_equal:
                        pass
                    """ print(line_in_words)
                    print(final_word_list)
                    print(documents_are_equal)
                    input('\n') """
                    

                    if not documents_are_equal:
                        if not self.flag:
                            print(old_final_word_list)
                            print(old_line_in_words)
                            self.flag = True
                        print('WAIT A MINUTE')
                        print(final_word_list)
                        input(line_in_words)
                    
                    # temporary
                    old_final_word_list = final_word_list
                    old_line_in_words = line_in_words

                    self.document_counter += 1
        if not testing:
            print(self.document_counter, ' documents have been processed')
        if testing:
            print(self.document_counter, ' documents have been tested')

    def parse_reuters_21578_corpus(self, corpus_file, word_index_vocabulary_file, testing = False): # rename function?, creating / testing
        '''Used to loop through all sgml_files '''
        my_path = os.path.abspath(os.path.dirname(__file__))
        datapath = '../Reuters_data'
        first_datafile = '/reut2-000.sgm'

        if not testing:
            with open(corpus_file + '.csv', mode='w') as csv_file:
                self.csv_writer = csv.writer(csv_file, delimiter=',')
    
                for i in range(22): # since the data is constant and we only do it once we can hardcode here
                    """ if i == 1:
                        break """
                    if i < 10:
                        filepath = first_datafile[0:9] + str(i) + first_datafile[10:] 
                    elif 10 <= i and i < 22:
                        filepath = first_datafile[0:8] + str(i) + first_datafile[10:]
                    else:
                        raise ValueError('Unexpected filenumber')

                    rel_filepath = datapath + filepath 
                    abs_filepath = os.path.join(my_path, rel_filepath)
                    
                    self.preprocess_reuters_sgml(abs_filepath, testing)

            self.write_word_index_vocab(word_index_vocabulary_file)
        
        elif testing:
            for i in range(22): # since the data is constant and we only do it once we can hardcode here
                    if i < 10:
                        filepath = first_datafile[0:9] + str(i) + first_datafile[10:] 
                    elif 10 <= i and i < 22:
                        filepath = first_datafile[0:8] + str(i) + first_datafile[10:]
                    else:
                        raise ValueError('Unexpected filenumber')

                    rel_filepath = datapath + filepath 
                    abs_filepath = os.path.join(my_path, rel_filepath)
                    
                    self.preprocess_reuters_sgml(abs_filepath, testing)


    def create_corpus_file_generator(self, corpus_file):
        '''Used to create a generator from the vectorized file to be tested'''
        with open(corpus_file + '.csv', mode='r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line in reader:
                if line:    # Some lines are blank for now
                    yield(line)  


    def test_vectorization(self, corpus_file, word_index_vocabulary_file):
        '''Test the vocabulary and the vectorized file of the part of the corpus that was vectorized'''

        self.document_counter = 0

        # Set Index to Word Vocab
        self.index_word_vocab =  {}
        with open(word_index_vocabulary_file + '.csv', mode='r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for line in reader:
                if line:    # Some lines are blank for now
                    self.index_word_vocab[line[1]] = line[0]
        
        # Create generator for our text
        self.corpus_file_generator = self.create_corpus_file_generator(corpus_file)
        self.parse_reuters_21578_corpus(corpus_file, word_index_vocabulary_file, testing = True)

        # Results,


### Class helper function

def modApte_training_samples_screened(tag):
    '''Function to use in the beautiful soup find_all() method for the REUTERS-21578 dataset. It selects the training tags of the Modified Apte Split and 
    also screens out the articles with topic attribute='YES but containing no TOPICS categories'''
    
    #Official
    return tag.name=='reuters' and tag['topics'] == "YES" and tag['lewissplit'] == "TRAIN" and tag.topics.d

    # The one below seems more reasonable but has 10794 matching documents:
    #tag.name=='reuters' and tag['topics'] == "YES" and (tag['lewissplit'] == "TRAIN" or tag['lewissplit'] == 'TEST') and tag.topics.d

    # Use this to select from earn topic
    #return tag.name=='reuters' and tag['topics'] == "YES" and tag['lewissplit'] == "TRAIN" and any(d.string =='earn' for d in tag.topics.children)


def main():
    corpus_file = 'Reuters_Corpus_Vectorized'; word_index_vocabulary_file = 'Reuters_Corpus_Vocabulary'

    cp = ReutersCorpusPreprocessor()
    #cp.parse_reuters_21578_corpus(corpus_file, word_index_vocabulary_file)
    cp.test_vectorization(corpus_file, word_index_vocabulary_file)
    
    print(cp.document_counter)

if __name__ == '__main__':
    main()


    # TODO: 4) Git and Github 5) Skriva om vad vi gjorde 6) LABELS!