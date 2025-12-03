# Bayesian-Classification-Sentiment-Analysis
Sentiment Analysis prediction algorithm for a Movies review dataset

# Summary
The purpose of this assignment deals with the creation of a Bayesian Classification algorithm and demonstrate the understanding of the algorithm code and how the classifier function.

The Excel file “movie_reviews.xlsx” on Canvas contains movie reviews from IMDb along with their associated binary sentiment polarity labels (“positive” or ”negative”) and a split indicator to determine if we are supposed to use the corresponding review for training your classifier (“train”) or for the evaluation (“test”).

The creation of the classifier is broken into 6 tasks. Those 6 tasks involves:
- Reading and Manipulating the data contained in the Excel file to create Training data, Training Labels, Test data and Test labels. The function related also display the number of positive and negative reviews in each set (Training and Evaluation)
- Extracting relevant features by identifying most important words to train classifier and pre-process the words by applying transformations (removing non-alphanumeric characters, lowercasing)
- Counting feature frequencies according to parameters of word length and minimum number of occurrences for likelihoods and priors calculations
- Classify review according using logarithmic calculation of likelihoods and priors to perform a Minimum-error-rate classification
- Evaluate the results using a k-fold cross-validation procedure, hyper-parameter tuning to maximize model accuracy, and assess accuracy on the test set.

After reading the data, we can see we are dealing with 49,999 reviews labelled with individual sentiment. Reviews are split between positive and negative reviews as well as a train/test split. Especially:
- 12500 positive reviews are present in the training set
- 12500 negative reviews are present in the training set
- 12499 positive reviews are present in the evaluation set
- 12500 negative reviews are present in the evaluation set

For a minimum number of word occurrence in the training set of 2000, the best parameter for the minimum word length to be 4. After final evaluation on the test set, we managed to output a confusion matrix of:
[11064 1436]
[6001 6498]

From the above matrix, we could the following percentages and accuracy score:
- 25.99 % of True positives
- 44.26 % of True negatives
- 5.74 % of False positives
- 24.01 % of False negatives
- Accuracy score of 70.25 %

We also test our classifier, using the best parameters, on newly created reviews. The new positive review has been classified as "positive" and the new negative review has been classified as "negative", which prove the classifier working.

# Tools
- Python
- Pandas
- Numpy
- Sci-kit Learn
