# Sentiment Analysis using Tensorflow

The dataset used for this project can be found (here)[http://ai.stanford.edu/~amaas/data/sentiment/]. The data was first extracted from individual files and merged into a single dataframe with respective ratings, and then serialized into a train.pickle and test.pickle file. Then, the data was parsed. This step involved multiple normalization steps like lowercasing, replacing accents, stemming, removing stopwords, etc. before using a CountVectorizer to convert this into a matrix with a vocabulary of 2000 top used words. The result was a tuple (trainX, trainY, testX, testY) that was then serialized into the movies.data file.

Finally, sentiment.py runs AdamOptimizer on a 2-layer NN with sigmoid activation function. The model is trained through Tensorflow's automatic differentiation and optimization APIs which does most of the heavy-lifting. We then run the model on the test set getting an accuracy of about 83%.
