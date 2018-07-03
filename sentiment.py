import tensorflow as tf, numpy as np, pandas as pd, pickle

def sigmoid(z):
    return 1 / (1 + tf.exp(-z))

def activate(W, b, X):
    return sigmoid(tf.matmul(W, X) + b)

def init_layer(inputs, outputs, i):
    W = tf.Variable(np.random.random((outputs, inputs)) - 0.5, name='W'+str(i))
    b = tf.Variable(np.random.random((outputs, 1)) - 0.5, name='b'+str(i))
    return W, b

def fit(X, y):
    # Initialize session
    session = tf.Session()
    n, m = X.shape[1], X.shape[0]

    # Input placeholder
    layer0 = tf.placeholder(shape = X.T.shape, dtype = tf.float64)
    output = tf.placeholder(shape = y.shape, dtype = tf.float64)
    # Layer parameters
    W1, b1 = init_layer(n, 60, 1)
    W2, b2 = init_layer(60, 1, 2)
    # Layer activations
    layer1 = activate(W1, b1, layer0)
    layer2 = activate(W2, b2, layer1)

    # Calculate error
    error = tf.reduce_sum(tf.square(layer2 - output)) / m
    # Initialize optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(error)

    # Initialize global variables
    init = tf.global_variables_initializer()
    session.run(init)

    # Run gradient descent for 100 iterations
    feed = {layer0: X.T.todense(), output: y}
    for i in range(200):
        print("Iteration", i + 1)

        # Run 1 step of gradient descent
        session.run(optimizer, feed_dict=feed)

        # Print cost
        print(session.run(error, feed_dict=feed))

    # Return a model representation
    model = [(session.run(W1), session.run(b1)), (session.run(W2), session.run(b2))]
    return model

def predict(layers, testX):
    W1, b1 = layers[0]
    W2, b2 = layers[1]

    X = testX.T.todense()
    layer1 = 1 / (1 + np.exp(np.matmul(W1, X) + b1))
    layer2 = 1 / (1 + np.exp(np.matmul(W2, layer1) + b2))

    return np.array(layer2).flatten()

def main(force = False):
    trainX, trainY, testX, testY = pickle.load(open("movies.data", "rb")) 

    model = None
    if force:
        model = fit(trainX, np.vectorize(int)(trainY > 0.5))
        pickle.dump(model, open("model1.model", "wb"))
    else:
        model = pickle.load(open('model1.model', 'rb'))
    
    predictions = predict(model, testX)
    true_positives = sum((predictions > .5) & (testY > .5))
    true_negatives = sum((predictions <= .5) & (testY <= .5))

    print(true_positives, sum(testY > 0.5))
    print(true_negatives, sum(testY <= 0.5))

    print((true_positives + true_negatives) / len(testY))

    print("DONE")

main(False)