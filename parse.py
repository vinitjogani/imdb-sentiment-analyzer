import re, pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

def normalize(review):
    review = review.lower()
    review = only_words.sub("", review)
    return review

def stem(words):
    return [stemmer.stem(w).strip() for w in words]

def replace_stopwords(words):
    return [w for w in words if w not in redundant and w != ""]

def transform(review):
    review = normalize(review)
    words = review.split()
    words = stem(words)
    words = replace_stopwords(words)
    review = ' '.join(words)
    return review

only_words = re.compile(r"[^a-zA-Z ]")
stemmer = PorterStemmer()
redundant = [normalize(w) for w in stopwords.words("english")]
redundant = stem(redundant)

def get_data():
    print("[+] Reading data...")
    train = pd.read_pickle("train.pickle")
    test = pd.read_pickle("test.pickle")

    print("[+] Transforming training data...")
    train["review"] = train["review"].apply(transform)
    train["rating"] = train["rating"] / 10

    print("[+] Transforming test data...")
    test["review"] = test["review"].apply(transform)
    test["rating"] = test["rating"] / 10

    print("[+] Almost done...")
    return train, test

def vectorize(train, test):
    # Initialize count vectorizer
    counter = CountVectorizer(analyzer='word', max_features=2000)
    # Vectorize data
    trainX = counter.fit_transform(train['review'])
    testX = counter.transform(test['review'])
    # Extract target
    trainY = train['rating']
    testY = test['rating']
    # Return everything
    pickle.dump(counter, open("vectorization.data", "wb"))
    return trainX, trainY, testX, testY

def main():
    train, test = get_data()
    trainX, trainY, testX, testY = vectorize(train, test)
    pickle.dump((trainX, trainY, testX, testY), open("movies.data", "wb"))

if __name__ == "__main__":
    main()