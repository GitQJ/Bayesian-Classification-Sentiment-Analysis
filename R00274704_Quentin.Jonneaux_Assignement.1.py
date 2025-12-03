#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 20:54:14 2025

@author: Quentin
"""

# Importing necessary libraries
import pandas as pd # Pandas for data reading and manipulation
import numpy as np # Numpy for mathematical calculation and manipulating arrays
import re # Regular expression for pre-processing
from math import log # Logarithm for working likelihoods
from sklearn import metrics # Scikit Learn metrics for computing confusion matrices and accuracy scores
from sklearn import model_selection # Scikit Learn model_selection to apply K-Fold cross validation

###############################################################################
# Task1 - Splitting and counting the reviews

def split_count():
    
    # Reading the data and store it in a dataframe (Global variable for reuse in other functions)
    global movies
    movies = pd.read_excel('/Users/Quentin/Desktop/Hdip Data Science and Analytics/Year 1/Semester 3/COMP8043 - Machine Learning/Assignment 1/movie_reviews.xlsx',sheet_name='Sheet1')
    
    # Subsetting data with training indicator and store training reviews (Global variable for reuse in other functions)
    global training_data
    training_data = movies[movies['Split'] == 'train']['Review'].to_numpy() # Converting to Numpy array to compute metrics
    
    # Subsetting data with training indicator and store training labels (Global variable for reuse in other functions)
    global training_labels
    training_labels = movies[movies['Split'] == 'train']['Sentiment'].to_numpy() # Converting to Numpy array to compute metrics
    
    
    # Subsetting data with test indicator and store test reviews (Global variable for reuse in other functions)
    global test_data
    test_data = movies[movies['Split'] == 'test']['Review'].to_numpy() # Converting to Numpy array to compute metrics
    
    # Subsetting data with test indicator and store test labels (Global variable for reuse in other functions)
    global test_labels
    test_labels = movies[movies['Split'] == 'test']['Sentiment'].to_numpy() # Converting to Numpy array to compute metrics
    
    # Storing number of positive review in the training set (Global variable for reuse in other functions)
    global nb_pos_reviews_train
    nb_pos_reviews_train = len(movies[(movies['Split'] == 'train') & (movies['Sentiment'] == 'positive')])
    
    # Storing number of negative review in the training set (Global variable for reuse in other functions)
    global nb_neg_reviews_train
    nb_neg_reviews_train = len(movies[(movies['Split'] == 'train') & (movies['Sentiment'] == 'negative')])
    
    
    # Printing number of positive reviews in the training set
    print('Number of positive reviews in training set: ', nb_pos_reviews_train)
    # Printing number of negative reviews in the training set
    print('Number of negative reviews in training set: ', nb_neg_reviews_train)
    # Printing number of positive reviews in the test set
    print('Number of positive reviews in evaluation set: ', len(movies[(movies['Split'] == 'test') & (movies['Sentiment'] == 'positive')]))
    # Printing number of negative reviews in the test set
    print('Number of negative reviews in evaluation set: ', len(movies[(movies['Split'] == 'test') & (movies['Sentiment'] == 'negative')]))
    
    # Returning training data, training labels, test data, test labels
    return training_data, training_labels, test_data, test_labels

###############################################################################
# Task2 - Extract relevant features

# Input parameters: training data from split_count, minimum length of words, minimum number of word occurences
def extract_features(training_data,min_Word_Length,min_Word_Occurence):
    
    # Declare a list to store all words in training reviews
    all_word_list = []
    
    # For each review in training set
    for text in training_data:
        # Storing the text
        original_text = text
        # Removing all non-alphanumeric characters of the text
        clean_text = re.sub(r'[^a-zA-Z0-9]', ' ', original_text)
        # Removing triple spaces resulting from previous cleaning step
        clean_text = re.sub(' +', ' ', clean_text)
        # Lowercasing all words in of the text
        lower_text = clean_text.lower()
        # Splitting text to get a list of each word
        review_word_list = lower_text.split()
        # Extending all word list with above list
        all_word_list.extend(review_word_list)
    
    # Declare a dictionnary mapping words to number of occurence in the training set
    word_Occurences = {}
    
    # For every word in the training set (using list created in previous steps)
    for word in all_word_list:
        # If word length respects minimum length parameter
        if (len(word)>=min_Word_Length):
            # If word is present in dictionnary
            if (word in word_Occurences):
                # Increase occurence count of the word by 1
                word_Occurences[word] = word_Occurences[word] + 1
            # If word not present in dictionnary
            else:
                # Create the word key with a value of 1
                word_Occurences[word]=1    
    
    
    # Declare a list of words respecting parameters
    main_training_word_list = []
    
    # For each word key in above dictionnary
    for word in word_Occurences:
        # if word meet minimum occurence parameter
        if word_Occurences[word]>=min_Word_Occurence:
            # Optional - Display word and count in console (uncomment next line if needed)
            print(word + ":" + str(word_Occurences[word]))
            
            # Append word to list of word respecting parameters
            main_training_word_list.append(word)
    
    # Optional - Display list of words respecting parameters in console (uncomment next line if needed)
    print(main_training_word_list)
    
    # Returning list of words respecting parameters
    return main_training_word_list

###############################################################################
# Task3 - count feature frequencies

# Input parameters: training data from split_count, list of words respecting parameters from extract_features
def count_frequencies(training_data,main_training_word_list):
    
    # Storing indexes of positive labels in training set
    train_pos_indexes = np.where([training_labels=='positive'][0])
    # Storing positive reviews in training set using labels above
    train_pos_reviews = training_data[train_pos_indexes]
    
    # Storing indexes of negative labels in training set
    train_neg_indexes = np.where([training_labels=='negative'][0])
    # Storing negative reviews in training set using labels above
    train_neg_reviews = training_data[train_neg_indexes]
    
    # Declaring dictionaries to count presence of word in positive reviews and negative reviews, respectively
    pos_word_occurences= {}
    neg_word_occurences= {}

    # For each word in the word list respecting parameters
    for word in main_training_word_list:
        # For each review in all positive reviews in training set
        for review in train_pos_reviews:
            # If word is present in review
            if word in review.lower().split():
                # If word present in positive review dictionary
                if word in pos_word_occurences:
                    # Increase occurence count by 1
                    pos_word_occurences[word] = pos_word_occurences[word] + 1
                # If word not present in positive review dictionnary
                else:
                    # Create the word key with a value of 1
                    pos_word_occurences[word]=1
        # If word not present in dictionary after all reviews
        if word not in pos_word_occurences.keys():
            # Create the word key with a value of 0
            pos_word_occurences[word]=0
    
    # For each word in the word list respecting parameters
    for word in main_training_word_list:
        # For each review in all negative reviews in training set
        for review in train_neg_reviews:
            # If word is present in review
            if word in review.lower().split():
                # If word present in negative review dictionary
                if word in neg_word_occurences:
                    # Increase occurence count by 1
                    neg_word_occurences[word] = neg_word_occurences[word] + 1
                # If word not present in positive review dictionnary
                else:
                    # Create the word key with a value of 1
                    neg_word_occurences[word]=1
        # If word not present in dictionary after all reviews
        if word not in neg_word_occurences.keys():
            # Create the word key with a value of 0
            neg_word_occurences[word]=0
            
    # Optional - Display occurence dictionnaries in console (uncomment next 2 lines if needed)
    print(pos_word_occurences)
    print(neg_word_occurences)
       
    # Retuning word occurence dictionnaries (one for each types)
    return pos_word_occurences,neg_word_occurences
    
###############################################################################
# Task4 - calculate feature likelihoods and priors

# Input parameters: word occurence dictionnaries from count_frequencies, number of positive and negative reviews from split_count
def calc_likelihood_prior(pos_word_occurences,neg_word_occurences,nb_pos_reviews_train,nb_neg_reviews_train):
    
    # Storing prior of positive reviews
    pos_prior = nb_pos_reviews_train/(nb_pos_reviews_train+nb_neg_reviews_train)
    # Optional - Display prior of positive reviews in the console (uncomment next line if needed)
    print('Positive prior: ',pos_prior)
    
    # Storing prior of negative reviews
    neg_prior = nb_neg_reviews_train/(nb_pos_reviews_train+nb_neg_reviews_train)
    # Optional - Display prior of negative reviews in the console (uncomment next line if needed)
    print('Negative prior: ',neg_prior)
    
    #  Declare a laplace smoothing parameter of 1
    laplace_alpha = 1
    
    # Declare a likelihood dictionnary
    likelihood_dict={}
    
    # For each word in the positive occurence dictionnary
    for word in pos_word_occurences.keys():
        
        # Storing number of times word is appearing in a positive review
        nb_times_word_in_pos_review = pos_word_occurences[word]
        # Storing likelihood that the word is present in positive review (applying the laplace smoothing)
        likelihood_pos = (nb_times_word_in_pos_review + laplace_alpha)/(nb_pos_reviews_train + 2 * laplace_alpha)
    
        # Storing number of times word is appearing in a negative review
        nb_times_word_in_neg_review = neg_word_occurences[word]
        # Storing likelihood that the word is present in negative review (applying the laplace smoothing)
        likelihood_neg = (nb_times_word_in_neg_review + laplace_alpha)/(nb_neg_reviews_train + 2 * laplace_alpha)
        
        # Map the word in the likelihood dictionary to positive and negative likelihoods
        likelihood_dict[word] = [likelihood_pos,likelihood_neg]
    
    # Optional - Display likelihood dictionary in the console (uncomment next line if needed)
    print(likelihood_dict)
    
    # Return likelihood dictionary and priors
    return likelihood_dict,pos_prior,neg_prior

###############################################################################
# Task5 - maximum likelihood classification

# Input parameters: likelihood dictionary and priors from calc_likelihood_prior, review text to classify
def classify_max_likelihood(likelihood_dict,pos_prior,neg_prior,new_text):
    
    # Declaring logarithmic likelihoods as 0 (for both positive and negative likelihoods)
    logLikelihood_pos = 0
    logLikelihood_neg = 0
        
    # For each word in the likelihood dictionary
    for word in likelihood_dict.keys():
        # If word present in the review text to classify
        if word in new_text.split():
            # Store both likelihoods of word
            likelihood_word_pos = likelihood_dict[word][0]
            likelihood_word_neg = likelihood_dict[word][1]

            # Add word logarithmic likelihoods to related class likelihood
            logLikelihood_pos = logLikelihood_pos + log(likelihood_word_pos)
            logLikelihood_neg = logLikelihood_neg + log(likelihood_word_neg)
        
    # Optional - Display logarithmic likelihoods in console (uncomment next 2 lines if needed)
    print(logLikelihood_pos)
    print(logLikelihood_neg)
    
    # Minimum-error-rate classification
    # If difference of logarithmic likelihoods (Positive - Negative) greater than difference of logarithmic priors (Negative - Positive)
    if logLikelihood_pos - logLikelihood_neg > log(neg_prior) - log(pos_prior):
        # Predicting review is positive
        prediction = 'positive'
        # Optional - Display positive prediction in console (uncomment next line if needed)
        print('review is:', prediction)
    
    # If difference of logarithmic likelihoods (Positive - Negative) NOT greater than difference of logarithmic priors (Negative - Positive)
    else:
        # Predicting review is negative
        prediction = 'negative'
        # Optional - Display negative prediction in console (uncomment next line if needed)
        print('review is:', prediction)
    
    #  Return predicted sentiment label
    return(prediction)

###############################################################################
# Task6 - evaluation of results

# Declaring a main script to execute
def main():
    
    # Storing data and labels from training and test sets from split_count (Task1)
    training_data, training_labels, test_data, test_labels = split_count()
    
    # Creating a K-Fold cross validation procedure (Using 5 folds, shuffling indexes, setting random_state as 42 for reproducibility)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create a dictionary mapping each minimum word length to mean accuracies
    mean_accuracies = {}
    
    # For each minimum word length parameter (1,2,3,4,5,6,7,8,9,10)
    for i in range(1,11):
        
        # Declare a list of accuracies for each subset
        accuracies = []
        
        # For each train and test indexes in the K-fold procedure split
        for train_index, test_index in kf.split(training_data, training_labels):
                # Get the training word list using indexes used for training by K-Fold procedure, minimum length parameter and an arbitrary minimum word occurence parameters (4000 here) from extract_features (Task 2)
                main_training_word_list = extract_features(training_data[train_index],i,2000)
                # Get positive and negative occurences from count_frequencies (Task 3)
                pos_word_occurences,neg_word_occurences = count_frequencies(training_data,main_training_word_list)
                # Get likelihood dictionary and priors from calc_likelihood_prior (Task 4)
                likelihood_dict,pos_prior,neg_prior = calc_likelihood_prior(pos_word_occurences,neg_word_occurences,nb_pos_reviews_train,nb_neg_reviews_train)
                
                # Declare a list to store predictions for test subset
                preds = []
                # For each review text in test subset
                for text in training_data[test_index]:
                    # Predict class of review
                    pred = classify_max_likelihood(likelihood_dict, pos_prior, neg_prior,text)
                    # Append prediction to list
                    preds.append(pred)
                
                # Optionnal - Compute confusion matrix and display in console (uncomment next 2 lines if needed)
                C = metrics.confusion_matrix(training_labels[test_index], preds)
                print(C)

                # Store accuracyy score of classifier
                accuracy = metrics.accuracy_score(training_labels[test_index], preds)
                # Append accuracy score to list of accuracies
                accuracies.append(accuracy)
                
        # Compute and Store mean of accuracies in the list
        mean_accuracy = np.mean(accuracies)
        # Map parameter with mean accuracy to
        mean_accuracies[i]= mean_accuracy
    
    # Optional - Display mean accuracies dictionary in the console (uncomment next line if needed)
    print(mean_accuracies)
    
    # For each key-value pair in mean accuracies dictionary
    for k, v in mean_accuracies.items():
        # if value is highest mean accuracy
        if v == max(mean_accuracies.values()):
            # Store key parameter as best parameter
            best_param = k
            # Leave the loop
            break
    
    
    # Optional - Display best parameter in the console (uncomment next line if needed)
    print('Best minimum word length parameter is: ',best_param)
    
    # Get the training word list using training data, best minimum length parameter and an arbitrary minimum word occurence parameters (2000 here) from extract_features (Task 2)
    main_training_word_list = extract_features(training_data,best_param,2000)
    # Get positive and negative occurences from count_frequencies (Task 3)
    pos_word_occurences,neg_word_occurences = count_frequencies(training_data,main_training_word_list)
    # Get likelihood dictionary and priors from calc_likelihood_prior (Task 4)
    likelihood_dict,pos_prior,neg_prior = calc_likelihood_prior(pos_word_occurences,neg_word_occurences,nb_pos_reviews_train,nb_neg_reviews_train)
    
    # Declare a list to store predictions for test data
    test_preds = []
    # For each review text in test data
    for text in test_data:
        # Predict class of review
        test_pred = classify_max_likelihood(likelihood_dict, pos_prior, neg_prior,text)
        # Append prediction to list
        test_preds.append(test_pred)
        
    # Compute confusion matrix for final evaluation
    C = metrics.confusion_matrix(test_labels, test_preds)
    
    # Storing confusion matrix values
    test_true_negative = C[0,0]
    test_true_positive = C[1,1]
    test_false_negative = C[1,0]
    test_false_positive = C[0,1]
    
    # Display confusion matrix for the classification in the console
    print(C)
    
    # Display The percentage of true positives, true negatives, false positives and false negatives in the console
    print("% True positives: ",(test_true_positive/len(test_labels)) * 100, "%")
    print("% True negatives: ",(test_true_negative/len(test_labels)) * 100, "%")
    print("% False positives: ",(test_false_positive/len(test_labels)) * 100, "%")
    print("% False negatives: ",(test_false_negative/len(test_labels)) * 100, "%")
    
    # Storing the classification accuracy score
    test_accuracy = metrics.accuracy_score(test_labels, test_preds)
    # Display the classification accuracy score in the console
    print("Test accuracy score: ", test_accuracy * 100, "%")


###############################################################################
# Task7 - Optional, Trying classifier on newly written reviews 

    # Storing a new positive review (Movie: The Green Mile)
    positive_text = "The Green Mile is a deeply moving and unforgettable cinematic experience. Tom Hanks shines as Paul Edgecomb, a prison guard on death row, whose life is profoundly changed by John Coffey, played with remarkable sensitivity by Michael Clarke Duncan. Coffey, a gentle giant wrongly convicted of a heinous crime, possesses a mysterious and miraculous gift. The film masterfully blends elements of drama, fantasy, and suspense, creating a compelling narrative that explores themes of faith, redemption, and the inherent value of human life. The supporting cast delivers equally strong performances, adding depth and authenticity to this emotionally resonant story. More than just a prison drama, The Green Mile is a powerful testament to the human spirit's capacity for compassion and empathy, even in the darkest of circumstances. It's a film that stays with you long after the credits roll, prompting reflection on the complexities of morality and the possibility of miracles. While the subject matter is heavy, the film is ultimately uplifting, reminding us to look beyond appearances and recognize the humanity in everyone. A truly exceptional and emotionally rewarding movie."
    # Predicting the sentiment of review
    new_pred = classify_max_likelihood(likelihood_dict, pos_prior, neg_prior,positive_text)
    # Display prediction of new positive review in the console
    print('New review of The Green Mile is predicted: ', new_pred)
    
    # Storing a new negative review (Movie: Nocturnal Animals)
    negative_text = "Nocturnal Animals is a visually stunning but ultimately hollow and pretentious film that prioritizes style over substance. The narrative, split between a wealthy art gallery owner's present and the violent manuscript written by her ex-husband, feels disjointed and emotionally manipulative. The thriller sequences are gratuitously graphic, serving more as shock value than contributing meaningfully to the film's themes. While Amy Adams and Jake Gyllenhaal deliver solid performances, their characters remain largely unsympathetic and underdeveloped, trapped in a cycle of regret and trauma that's difficult to connect with. The film's biggest flaw is its self-indulgent tone and heavy-handed symbolism. It tries too hard to be clever and provocative, resulting in a convoluted plot and uneven pacing. The connection between the two storylines feels forced and the ending leaves a lingering sense of emptiness rather than genuine emotional resonance. Ultimately, Nocturnal Animals is a visually impressive but ultimately unsatisfying experience, leaving the viewer feeling more frustrated than moved. It's a case of style over substance that fails to deliver on its initial promise."
    # Predicting the sentiment of review
    new_pred = classify_max_likelihood(likelihood_dict, pos_prior, neg_prior,negative_text)
    # Display prediction of new negative review in the console
    print('New review of Nocturnal Animals is predicted: ', new_pred)
    
# Executing main function
main()