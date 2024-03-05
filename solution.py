import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import math

# texall - list of documents; each entry corresponds to a document which 
# is a list of words. (Simply, sentences.)
# • lbAll list of documents' labels.
# • voc - set of all distinct words in the train set.
# • cat - set of document categories

# Pw - matrix of class-conditional probablities p(x|wi)
# P - a vector of class priors p(wi)

def readTrainData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    texAll = []
    lbAll = []
    voc = []
    for line in lines:
        splitted = line.split('\t')
        lbAll.append(splitted[0])
        texAll.append(splitted[1].split())
        words = splitted[1].split()
        for w in words:
            voc.append(w)
    voc = set(voc)
    cat = set(lbAll)
    return texAll, lbAll, voc, cat

texAll, lbAll, voc, cat = readTrainData('r8-train-stemmed.txt')

def learn_NB_text():
    #Create prior vector, for each category, go through and count how many times it appears and divide it by the total number of words.
    P = np.array([lbAll.count(category)/len(lbAll) for category in cat])

    vec_list = []
    
    # List of all possible words
    cols = list(voc.copy()) 

    # Incase we encounter an unknown word. This will be used for Laplace Smoothing later.
    cols.append('unknown word') 

    for category in cat:
        
        # Will hold all probability vectors
        sentence_list = [] 
        for index, value in enumerate(lbAll):
            if value == category:
                # Create sentence from list of words, then append to list
                sentence_list.append(' '.join(texAll[index]))  

        # At this point we will have a list of all the sentences.

        # create countVectorizer object with voc as vocabulary 
        count_vec = CountVectorizer(vocabulary=list(voc))
        tdm = count_vec.fit_transform(sentence_list)

        # Create a DataFrame
        df = pd.DataFrame(tdm.toarray(), columns=count_vec.get_feature_names_out()) 

        # Sum number of appearances of every word in each sentence
        arr = np.array(df.sum(axis=0)) 

        # append unknown word probability
        arr = np.append(arr,0) 

        # calculate probability (including laplace smoothing)
            # tdm.sum() is total number of words in current class
            # len(voc) is number of distinct words
        arr = (arr + 1) / (tdm.sum() + len(voc))  

        # append 1D array of probabilities of specific category to our list.
        vec_list.append(arr) 

    Pw = np.array(vec_list)
    Pw = pd.DataFrame(Pw, index=list(cat), columns=cols)
    return Pw, P

texAllTest, lbAllTest, vocTest, catTest = readTrainData('r8-test-stemmed.txt')

def ClassifyNB_text(Pw,P):
    predicted = 0
    #iterate through every sentence in text and predict category with naive bayes classifier

    for index,list_of_words in enumerate(texAllTest): 
        #in some edge cases, there will be a probability that is very close to 0, so it's log will be -infinity.
        max_prob = -math.inf 
        prediction = ''

        #now we calculate P(sentence|category)*P(category) for each category and choose highest number as our prediction.
        for i,category in enumerate(cat): 
            log_sum = 0

            #calculate log(P(sentence|category))
            for word in list_of_words: 
                if word in voc:
                    log_sum += math.log(Pw.loc[category,word])    
                else:
                    log_sum += math.log(Pw.loc[category,'unknown word'])    
            
            # calculate log(P(sentence|category)*P(category))
            log_sum+=math.log(P[i]) 

            if log_sum > max_prob:  #if we find a probability that is higher.
                max_prob = log_sum
                prediction = category       
        # check if prediction was correct
        if lbAllTest[index] == prediction:
            predicted += 1
    return predicted/len(texAllTest)


Pw, P = learn_NB_text()
suc = ClassifyNB_text(Pw, P)

# Correctly predicted percentage: 0.9643672910004568
print(f"{suc}") 



