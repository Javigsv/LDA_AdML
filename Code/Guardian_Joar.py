

def calculate_topic_proportions(corpus,phi,K):
    topic_proportions = []

    for document in corpus:
        topic_proportions.append([])
        for topic in range(K):
            topic_sum = 0
            for word in document:            
                topic_sum += phi[document][word][topic]
            topic_sum = topic_sum/len(document)
            topic_proportions[document].append(topic_sum)  
    
    return topic_proportions