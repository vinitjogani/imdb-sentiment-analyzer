import numpy as np, pickle, re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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

model = pickle.load(open('model1.model', 'rb'))
counter = pickle.load(open('vectorization.data', 'rb'))

def predict(layers, testX):
    W1, b1 = layers[0]
    W2, b2 = layers[1]

    X = testX.T.todense()
    layer1 = 1 / (1 + np.exp(np.matmul(W1, X) + b1))
    layer2 = 1 / (1 + np.exp(np.matmul(W2, layer1) + b2))

    return np.array(layer2).flatten()

review = "_"
while review.strip() != "":
    review = input("Enter a review: ")
    dataX = counter.transform([transform(review)])

    predictions = predict(model, dataX)
    print(predictions, int(predictions > 0.57))
