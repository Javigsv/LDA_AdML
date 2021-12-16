import numpy as np
import time

class mixture_of_unigram():
	def __init__(self,show_word,topic_num=10,fix_labeled_doc=True,alpha=2):
		self.topic_num=topic_num
		#show_word is a list of string(word) corresponding to word matrixes passed in.
		self.show_word=show_word
		self.word_num=len(show_word)
		self.alpha=alpha
		self.fix_labeled_doc=fix_labeled_doc #if True topic for labeled document would not change.

		self.labeled_word_matrix=None
		self.labeled_doc2topic=None #Z in the original paper.
		self.word_matrix=None	#unlabeled document.
		self.doc2topic=None		#unlabeled document. Z in the original paper.

		#randomly initialize topic to word distribution, 10**-200 to avoid divide by 0
		self.topic2word=np.random.uniform(10**-200,1,[self.topic_num,self.word_num])
		self.topic2word=self.topic2word/self.topic2word.sum(axis=1,keepdims=True)

		#randomly initialize topic sampling probability
		self.topic=np.random.uniform(10**-200,1,[self.topic_num])
		self.topic=self.topic/self.topic.sum()

	def add_labeled_doc(self,word_matrix,topics):
		#word_matrix is the same as the one returned by sklearn.vectorizer()
		#topics is a list of int(topic index).
		
		#create one hot label, which means we set the expectation as 1.
		topics=np.array(topics)
		doc2topic=np.zeros([word_matrix.shape[0],self.topic_num])
		doc2topic[np.arange(word_matrix.shape[0]),topics]=1
		
		if self.labeled_word_matrix is None:
			self.labeled_word_matrix=word_matrix.copy()
			self.labeled_doc2topic=doc2topic
		else:
			assert(word_matrix.shape[1]==self.labeled_word_matrix.shape[1])
			self.labeled_word_matrix=np.concatenate((self.labeled_word_matrix,word_matrix),axis=0)
			self.labeled_doc2topic=np.concatenate((self.labeled_doc2topic,doc2topic),axis=0)

	def add_unlabeled_doc(self,word_matrix):
		#word_matrix is the same as the one returned by sklearn.vectorizer()
	
		if self.word_matrix is None:
			self.word_matrix=word_matrix.copy()
		else:
			assert(word_matrix.shape[1]==self.word_matrix.shape[1])
			self.word_matrix=np.concatenate((self.word_matrix,word_matrix),axis=0)

	def train(self,iteration=10):
		for i in range(iteration):
			start = time.time()
			print('Running iteration', i)
			if self.labeled_word_matrix is None and self.word_matrix is None:
				#don't need to do anything
				return
			elif self.labeled_word_matrix is not None and self.word_matrix is None:
				#only self.labeled_word_matrix has document in it. Analytic solution. Only need to do one time.

				self.__maximization(self.labeled_word_matrix,self.labeled_doc2topic)
				if self.fix_labeled_doc:
					return
				else:
					self.labeled_doc2topic=self.__expectation(self.labeled_word_matrix)

	
			elif self.labeled_word_matrix is None and self.word_matrix is not None:
				#only unlabeled self.word_matrix has document. Unsupervised learning might not do well.
				self.doc2topic=self.__expectation(self.word_matrix)
				self.__maximization(self.word_matrix,self.doc2topic)
	
	
			else:
				#Both have document.
		
				#calculate expectation for unlabeled document.
				self.doc2topic=self.__expectation(self.word_matrix)
				if self.fix_labeled_doc is False:
					self.labeled_doc2topic=self.__expectation(self.labeled_word_matrix)
	
				word_matrix=np.concatenate((self.labeled_word_matrix,self.word_matrix),axis=0)
				doc2topic=np.concatenate((self.labeled_doc2topic,self.doc2topic),axis=0)
	
				self.__maximization(word_matrix,doc2topic)

			end = time.time()
			tot_time = end-start
			if tot_time > 10:
				print('Completed in', tot_time,'seconds')
			
	def __maximization(self,word_matrix,doc2topic):
		#doc2topic of size doc_num*topic.
		#new topic sampling probability.
		assert np.all((doc2topic.sum(axis=0)+self.alpha-1)>=0) and ((word_matrix.shape[0]+self.topic_num*(self.alpha-1))!=0)  ,"vectorizer error or reset alpha"
		
		self.topic=(doc2topic.sum(axis=0)+(self.alpha-1))/(word_matrix.shape[0]+self.topic_num*(self.alpha-1))


		#new topic to word probability.
		#word_matrix and doc2topic should have the same shape[0].
		word_matrix=word_matrix.reshape(word_matrix.shape[0],-1,self.word_num)
		self.topic2word=(self.alpha-1)+(doc2topic.reshape(word_matrix.shape[0],self.topic_num,-1)*word_matrix).sum(axis=0)
		assert np.all(self.topic2word>=0) and np.all(self.topic2word.sum(axis=1)!=0) ,"vectorizer error or reset alpha"
		self.topic2word=self.topic2word/(self.topic2word.sum(axis=1,keepdims=True))

	def __expectation(self,word_matrix):
		
		#calculate the conditional probability of documents belonging to a topic using current self.topic and self.topic2word.

		assert(word_matrix.shape[1]==self.word_num)

		#probability of generating words in a document.
		word_prob=word_matrix.reshape(word_matrix.shape[0],self.word_num,-1)
		topic2word=self.topic2word.transpose([1,0]).reshape(-1,self.word_num,self.topic_num)
		word_prob=np.power(topic2word,word_prob)
		#divide by a common number to improve floating point precision. Would not change the result.
		word_prob=word_prob/(word_prob.mean(axis=2,keepdims=True)+10**-200)
		word_prob=np.prod(word_prob,axis=1)
		
		#probability of generating a specific topic.
		doc2topic=np.full([word_matrix.shape[0],self.topic_num],1.)
		doc2topic=doc2topic*self.topic

		#entire probability	
		doc2topic*=word_prob
		#If sum is 0 then assign value 1 to every topic, resulting in uniform distribution.
		doc2topic[(doc2topic.sum(axis=1,keepdims=True)==0).repeat(self.topic_num,axis=1)]=1
		doc2topic=doc2topic/doc2topic.sum(axis=1,keepdims=True)
		
		"""
		for i_doc,doc in enumerate(self.word_matrix):
			for i_topic in range(self.topic_num):
				doc2topic[i_doc][i_topic]*=self.topic[i_topic]
				for i_word,word in enumerate(doc):
					doc2topic[i_doc][i_topic]*=(self.topic2word[i_topic][i_word])**word
		doc2topic=doc2topic/doc2topic.sum(dim=1)
		"""
		
		#return of size word_matrix.shape[0]*self.topic_num
		return doc2topic

	"""
	def predict(self,word_matrix):
		doc2topic=self.__expectation(word_matrix)
		#doc2topic is a probability distribution over topic.
		#return the topic with highest probabily.
		return doc2topic.argmax(axis=1)
	"""
	def predict(self,word_matrix):
		#return the distribution directly.
		doc2topic=self.__expectation(word_matrix)
		#doc2topic is a probability distribution over topic.
		return doc2topic

	def get_topic(self,show_word=3):
		topic=[]
		for i in range(self.topic_num):
			topic.append([self.show_word[o] for o in self.topic2word[i].argsort()[:-show_word-1:-1]])
		return topic



	def log_sum_exp(self, x):

		xp = np.max(x)

		lse = xp + np.log(np.sum(np.exp(x - xp)))

		return lse
		


	def perplexity(self, test_corpus):
		
		log_p_corpus = 0
		tot_N = 0
		for doc in test_corpus:
			values = []
			for z in range(self.topic_num):
				log_p_doc_given_z = 0
				for w in doc:
					p_w_given_z = self.topic2word[int(z),int(w)]
					log_p_doc_given_z += np.log(p_w_given_z)
				
				log_p_z = np.log(self.topic[z])
				values.append(log_p_z + log_p_doc_given_z)			

			log_p_corpus += self.log_sum_exp(np.array(values))
			tot_N += len(doc)

		exp = -log_p_corpus / tot_N
		return np.exp(exp)


	def predictive_perplexity(self, test_corpus, removed):
		
		log_p_corpus = 0
		M = len(test_corpus)
		for m, user in enumerate(test_corpus):
			values = []
			values_wobs = []
			wrem = removed[m]
			for z in range(self.topic_num):
				
				p_wrem_given_z = self.topic2word[z, wrem]
				log_p_wrem_given_z = np.log(p_wrem_given_z)
				log_p_z = np.log(self.topic[z])
				
				values.append(log_p_wrem_given_z + log_p_z)



			log_p_corpus += self.log_sum_exp(np.array(values))

		exp = -log_p_corpus / M
		return np.exp(exp)

	
	
	def predictive_perplexity2(self, test_corpus, removed):
		
		log_p_corpus = 0
		M = len(test_corpus)
		for m, user in enumerate(test_corpus):
			values = []
			values_wobs = []
			wrem = removed[m]
			for z in range(self.topic_num):
				log_p_wobs_given_z = 0
				for w in user:
					p_w_given_z = self.topic2word[int(z),int(w)]
					#print('\t\t', p_w_given_z, np.log(p_w_given_z))
					log_p_wobs_given_z += np.log(p_w_given_z)
				
				log_p_z = np.log(self.topic[z])
				log_p_wrem_given_z = self.topic2word[int(z),int(wrem)]
				#print('\t', log_p_doc_given_z, np.exp(log_p_doc_given_z))
				values.append(log_p_wrem_given_z + log_p_z + log_p_wobs_given_z)
				values_wobs.append(log_p_z + log_p_wobs_given_z)


			log_p_corpus += self.log_sum_exp(np.array(values)) - self.log_sum_exp(np.array(values_wobs))

		exp = -log_p_corpus / M
		return np.exp(exp)
		

