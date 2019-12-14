
# misc
import csv
import random
import itertools
import collections
import numpy as np 
import pandas as pd
import math
import string

#nltk
from   nltk.corpus       import stopwords
from   nltk.collocations import BigramCollocationFinder
from   nltk.metrics      import BigramAssocMeasures
from   nltk.classify     import NaiveBayesClassifier, SklearnClassifier
import nltk.classify.util 
import nltk.metrics
import nltk

#sklearn
from   sklearn.model_selection import cross_validate
from   sklearn.svm             import LinearSVC, SVC
import liwc
import warnings
warnings.filterwarnings("ignore")


# Creating features functions
# Exclude key words from stopwords list
# Keeping some words that might have meaning for sentiment
stoplist = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 
                                                 'no', 'not', 'only', 'such', 'few', 'so', 
                                                 'too', 'very', 'just', 'any', 'once'))

### Importing Positive and Negative Files
print('Importing csv files with positive reviews')
pos_data = []
with open('./data/positive-data.csv', 'r') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        pos_data.append(val[0]) 

print('Importing csv files with negative reviews')
neg_data = []
with open('./data/negative-data.csv', 'r') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        neg_data.append(val[0]) 

def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new


def word_split_sentiment(data):
    data_new = []
    for (word, sentiment) in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append((word_filter, sentiment))
    return data_new


def word_features(words): 
    return dict([(word, True) for word in words])


# BOW filter stopswords and punctuation
def bag_of_words_stopwords(words):
    words_clean = []
 
    for word in words:
        word = word.lower()
        if word not in stoplist and word not in string.punctuation:
            words_clean.append(word)
    
    words_dictionary = dict([word, True] for word in words_clean)
    
    return words_dictionary

def POS_features(words):
    tagged_words = nltk.pos_tag(words)
    document_bigrams = nltk.bigrams(words)
    features = {}
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


def stopwords_filtered_word_features(words):
    return dict([(word, True) for word in words if word not in stoplist])


def bigram_word_features(words, 
                      score_fn=BigramAssocMeasures.chi_sq, 
                      n=500):
    bigram_finder = BigramCollocationFinder.from_words(words)
    #bigrams = bigram_finder.nbest(score_fn, n) 
    try: bigrams = bigram_finder.nbest(score_fn, n)
    except: bigrams = [ ]
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def bigram_word_features_swords(words, 
                                score_fn=BigramAssocMeasures.chi_sq, 
                                n=200):
    all_words_without_punctuation = [word for word in words if word not in string.punctuation]
    bigram_finder = BigramCollocationFinder.from_words(all_words_without_punctuation)
    try: bigrams = bigram_finder.nbest(score_fn, n)
    except: bigrams = [ ]
    return dict([(ngram, True) for ngram in itertools.chain(all_words_without_punctuation, bigrams) if ngram not in stoplist])


#### Function to evaluate classifiers:
#### Takes features as input and number of folds for cross-validation
#### Used NLTK Naive Bayes Classifier to compare against SKLearn SVM Classifier

# Calculating Precision, Recall & F-measure
def evaluate_classifier(feature_x, n_folds=5): 
    
    # 5-fold default for cross-validation  
    # train_feats = 75% of pos_data + 75% of neg_data
    # test_feats  = 25% of pos_data + 25% of neg_data
    
    neg_feats  = [(feature_x(i), 'neg') for i in word_split(neg_data)]
    pos_feats  = [(feature_x(i), 'pos') for i in word_split(pos_data)]
        
    neg_cutoff = int(len(neg_feats)*0.75)
    pos_cutoff = int(len(pos_feats)*0.75)
 
    train_feats = neg_feats[:neg_cutoff] + pos_feats[:pos_cutoff]
    test_feats  = neg_feats[neg_cutoff:] + pos_feats[pos_cutoff:]
            
    #classifier_list = ['NB','SVM']
    classifier_list = ['NB']
    for cl in classifier_list:
        
        if cl == 'NB':
            classifierName = 'Naive Bayes'
            # Using NLTK NaiveBayesClassifier
            classifier = NaiveBayesClassifier.train(train_feats)
        else:
            classifierName = 'SVM'
            # Using sklearn classifier
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(train_feats)
            
        ref_sets  = collections.defaultdict(set)
        test_sets = collections.defaultdict(set)
 
        for i, (feats, label) in enumerate(test_feats):
                ref_sets[label].add(i)
                observed = classifier.classify(feats)
                test_sets[observed].add(i)
 
        accuracy      = nltk.classify.util.accuracy(classifier, test_feats)
    
        pos_precision = nltk.precision(ref_sets['pos'], test_sets['pos'])
        pos_recall    = nltk.recall(   ref_sets['pos'], test_sets['pos'])
        pos_fmeasure  = nltk.f_measure(ref_sets['pos'], test_sets['pos'])
        
        neg_precision = nltk.precision(ref_sets['neg'], test_sets['neg'])
        neg_recall    = nltk.recall(   ref_sets['neg'], test_sets['neg'])
        neg_fmeasure  = nltk.f_measure(ref_sets['neg'], test_sets['neg'])
        
        print( '-------------------------------------'     )
        print( 'Result for Single Fold' + '(' + classifierName + ')' )
        print( '-------------------------------------'     )
        print( 'accuracy : {:.4F}'.format(accuracy))
        print( 'precision: {:.4F}'.format((pos_precision + neg_precision) / 2))
        print( 'recall   : {:.4F}'.format((pos_recall    + neg_recall)    / 2))
        print( 'f-measure: {:.4F}'.format((pos_fmeasure  + neg_fmeasure)  / 2))            
        #classifier.show_most_informative_features()
    
    print( '\n')
    
    ## CROSS VALIDATION
    train_feats = neg_feats + pos_feats   
    
    # Shuffle training set  
    random.shuffle(train_feats)  
        
    for cl in classifier_list:
        
        subset_size = int(len(train_feats)/n_folds)
        accuracy      = []
        pos_precision = []
        pos_recall    = []
        neg_precision = []
        neg_recall    = []
        pos_fmeasure  = []
        neg_fmeasure  = []
        cv_count = 1
        
        print( '--------------------------' )
        print( 'Beginning Cross-validation' )
        print( '--------------------------' )
        
        for i in range(n_folds):        
            testing_this_round  = train_feats[i*subset_size:][:subset_size]
            training_this_round = train_feats[:i*subset_size] + train_feats[(i+1)*subset_size:]
            
            if cl == 'NB':
                classifierName = 'Naive Bayes'
                # Using NLTK NaiveBayesClassifier
                classifier = NaiveBayesClassifier.train(training_this_round)
            else:
                classifierName = 'SVM'
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(training_this_round)
                    
            ref_sets = collections.defaultdict(set)
            test_sets = collections.defaultdict(set)
            
            for i, (feats, label) in enumerate(testing_this_round):
                ref_sets[label].add(i)
                observed = classifier.classify(feats)
                test_sets[observed].add(i)
            
            cv_accuracy      = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = nltk.precision(ref_sets['pos'], test_sets['pos'])
            cv_pos_recall    = nltk.recall(   ref_sets['pos'], test_sets['pos'])
            cv_pos_fmeasure  = nltk.f_measure(ref_sets['pos'], test_sets['pos'])        
            cv_neg_precision = nltk.precision(ref_sets['neg'], test_sets['neg'])
            cv_neg_recall    = nltk.recall(   ref_sets['neg'], test_sets['neg'])
            cv_neg_fmeasure  = nltk.f_measure(ref_sets['neg'], test_sets['neg'])
            
            print('Fold: {} Acc       : {:.4F}'.format(cv_count, cv_accuracy))
            print('Fold: {} pos_prec  : {:.4F} neg_prec  : {:.4F}'.format(cv_count, cv_pos_precision,cv_neg_precision))
            print('Fold: {} pos_recall: {:.4F} neg_recall: {:.4F}'.format(cv_count, cv_pos_recall,   cv_neg_recall))
            print('Fold: {} pos_fmeas : {:.4F} neg_fmeas : {:.4F}'.format(cv_count, cv_pos_fmeasure, cv_neg_fmeasure))
            print( '--' )
            
            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)
            
            cv_count += 1
                
        print( '----------------------------------------------------------' )
        print( '{}-Fold Cross Validation results for {} Classifier'.format(n_folds, classifierName ))
        print( '----------------------------------------------------------' )
        print( 'accuracy : {:.4F}'.format(sum(accuracy) / n_folds))
        print( 'precision: {:.4F}'.format((sum(pos_precision)/n_folds + sum(neg_precision)/n_folds) / 2))
        print( 'recall   : {:.4F}'.format((sum(pos_recall)/n_folds    + sum(neg_recall)/n_folds)    / 2))
        print( 'f-measure: {:.4F}'.format((sum(pos_fmeasure)/n_folds  + sum(neg_fmeasure)/n_folds)   / 2))
        print( '\n' )

# #### LIWC function
def read_words():
  poslist = []
  neglist = []
  #print('SentimentLexicons/liwcdic2007.dic')
  flexicon = open('SentimentLexicons/liwcdic2007.dic', encoding='latin1')
  # read all LIWC words from file
  wordlines = [line.strip() for line in flexicon]
  for line in wordlines:
    if not line == '':
      items = line.split()
      word = items[0]
      classes = items[1:]
      for c in classes:
        if c == '126':
          poslist.append( word )
        if c == '127':
          neglist.append( word )
  return (poslist, neglist)
poslist, neglist = read_words()

# LIWC Features filteering stopwords (and punctuation)
def liwc_features(words):
    poslist, neglist = read_words()
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stoplist and word not in string.punctuation:
            words_clean.append(word)
    
    features = {}
    Pos = 0
    Neg = 0
    for word in words_clean:
        if word in poslist:
                Pos += 1
        if word in neglist:
                Neg += 1
        features['poscount'] = Pos
        features['negcount'] = Neg    
    return features


# ### Multiple-classifier function with cross-validation 
def evaluate_mult_classifiers(feature_x, n_folds=5): 
    
    # 5-fold default for cross-validation  
    # train_feats = 75% of pos_data + 75% of neg_data
    # test_feats  = 25% of pos_data + 25% of neg_data
    
    neg_feats  = [(feature_x(i), 'neg') for i in word_split(neg_data)]
    pos_feats  = [(feature_x(i), 'pos') for i in word_split(pos_data)]
        
    neg_cutoff = int(len(neg_feats)*0.75)
    pos_cutoff = int(len(pos_feats)*0.75)
 
    train_feats = neg_feats[:neg_cutoff] + pos_feats[:pos_cutoff]
    test_feats  = neg_feats[neg_cutoff:] + pos_feats[pos_cutoff:]
            
    classifier_list = ['NB','SVM']
    
    ## CROSS VALIDATION
    train_feats = neg_feats + pos_feats   
    
    # Shuffle training set  
    random.shuffle(train_feats)  
        
    for cl in classifier_list:
        
        subset_size = int(len(train_feats)/n_folds)
        accuracy      = []
        pos_precision = []
        pos_recall    = []
        neg_precision = []
        neg_recall    = []
        pos_fmeasure  = []
        neg_fmeasure  = []
        cv_count = 1
        
        print( '--------------------------' )
        print( 'Beginning Cross-validation' )
        print( '--------------------------' )
        
        for i in range(n_folds):        
            testing_this_round  = train_feats[i*subset_size:][:subset_size]
            training_this_round = train_feats[:i*subset_size] + train_feats[(i+1)*subset_size:]
            
            if cl == 'NB':
                classifierName = 'Naive Bayes'
                # Using NLTK NaiveBayesClassifier
                classifier = NaiveBayesClassifier.train(training_this_round)
            else:
                classifierName = 'SVM'
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(training_this_round)
                    
            ref_sets = collections.defaultdict(set)
            test_sets = collections.defaultdict(set)
            
            for i, (feats, label) in enumerate(testing_this_round):
                ref_sets[label].add(i)
                observed = classifier.classify(feats)
                test_sets[observed].add(i)
            
            cv_accuracy      = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = nltk.precision(ref_sets['pos'], test_sets['pos'])
            cv_pos_recall    = nltk.recall(   ref_sets['pos'], test_sets['pos'])
            cv_pos_fmeasure  = nltk.f_measure(ref_sets['pos'], test_sets['pos'])        
            cv_neg_precision = nltk.precision(ref_sets['neg'], test_sets['neg'])
            cv_neg_recall    = nltk.recall(   ref_sets['neg'], test_sets['neg'])
            cv_neg_fmeasure  = nltk.f_measure(ref_sets['neg'], test_sets['neg'])
            
            print('Fold: {} Acc       : {:.4F}'.format(cv_count, cv_accuracy))
            print('Fold: {} pos_prec  : {:.4F} neg_prec  : {:.4F}'.format(cv_count, cv_pos_precision,cv_neg_precision))
            print('Fold: {} pos_recall: {:.4F} neg_recall: {:.4F}'.format(cv_count, cv_pos_recall,   cv_neg_recall))
            print('Fold: {} pos_fmeas : {:.4F} neg_fmeas : {:.4F}'.format(cv_count, cv_pos_fmeasure, cv_neg_fmeasure))
            print( '--' )
            
            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)
            
            cv_count += 1
                
        print( '----------------------------------------------------------' )
        print( '{}-Fold Cross Validation results for {} Classifier'.format(n_folds, classifierName ))
        print( '----------------------------------------------------------' )
        print( 'accuracy : {:.4F}'.format(sum(accuracy) / n_folds))
        print( 'precision: {:.4F}'.format((sum(pos_precision)/n_folds + sum(neg_precision)/n_folds) / 2))
        print( 'recall   : {:.4F}'.format((sum(pos_recall)/n_folds    + sum(neg_recall)/n_folds)    / 2))
        print( 'f-measure: {:.4F}'.format((sum(pos_fmeasure)/n_folds  + sum(neg_fmeasure)/n_folds)   / 2))
        print( '\n' )


def main():

    print('Baseline all words features')
    evaluate_classifier(word_features, 5)

    print('Stopwords Feature') 
    evaluate_classifier(stopwords_filtered_word_features)

    print('Using Bigram Features')
    evaluate_classifier(bigram_word_features)

    print('Using bigram features and stopwords')
    evaluate_classifier(bigram_word_features_swords)

    print('Evaluating Bag of words filtering stopwords and punctuations')
    evaluate_classifier(bag_of_words_stopwords)

    print('POS features')
    evaluate_classifier(POS_features)

    print('Running Classifier with LIWC Features')
    evaluate_classifier(liwc_features)

    print('Comparing Naive Bayes with Sklearn LinearSVC Classifier') 
    print('Using Word Features')
    evaluate_mult_classifiers(word_features)

    ### Using POS Tags features
    # evaluate_mult_classifiers(POS_features)

if __name__ == "__main__": main()