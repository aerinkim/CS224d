#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function
    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    return x/np.sqrt(np.sum(x**2,axis=1).reshape(-1,1))


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for *ONE* predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component) -> v_hat is an input, why do you call it predicted???? (D,1)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens (V,D)
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction (scalar)
    gradPred -- the gradient with respect to the predicted word vector (D,1)
    grad -- the gradient with respect to ***ALL OTHER*** word vectors (D,V)
    """

    score_numerator = np.exp(outputVectors.dot(predicted)) # (V,D)*(D,1) = (V,1)
    score_denominator = np.sum(np.exp(outputVectors.dot(predicted))) #(scalar)
    probability = score_numerator / score_denominator # for every word, (V,1)

    cost = -np.log( probability[target] ) # the cost for the target word only. (scalar)
    probability[target] -= 1

    # Below two lines are where everyone gets lost lol.
    gradPred = outputVectors.T.dot(probability) # (D,V)*(V,1)=(D,1) # d/dVc
    grad = np.outer(probability,predicted) # (V,1) -outer- (D,1) = (V,D) # d/dVo

    return cost, gradPred, grad
    #               (D,1)  (D,V)

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models
    
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample size.

    Note: See test_word2vec below for dataset's initialization."""

    # Sampling of indices is done for you. Do not modify this if you wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K)) # I don't understand why they extend but let's move on.

    grad           = np.zeros_like(outputVectors) # (V,D)
    sampledVectors = np.zeros_like(outputVectors) # (V,D)
    for i in indices[1:]:
        sampledVectors[i] += outputVectors[i]

    p_target_predicted = sigmoid(outputVectors[target].dot(predicted)) # (1,D)*(D,1)=scalar
    p_k_predicted = sigmoid(-sampledVectors.dot(predicted)) # (V,D)*(D,1)=(V,1)
    cost = -np.log(p_target_predicted) - np.sum(np.log(p_k_predicted))

    gradPred = (p_target_predicted - 1)*outputVectors[target] + (1-p_k_predicted).T.dot(sampledVectors)
    # (1,D)              scalar                 (1,D)                   (V,1).T             (V,D)     

    grad[target] = (p_target_predicted - 1)*predicted    #- (1-p_target_predicted)*predicted <<- commented out because it goes 0
    # (1,D)              scalar                 (1,D)                   scalar       (1,D)         
    for i in xrange(K): # grad for ALL OTHER WORDS (including k)
        k = indices[i+1]
        grad[k] +=  (1 - p_k_predicted[k])*predicted
        # (D,1)           scalar            (D,1)  
    return cost, gradPred, grad
    #               (1,D)  (V,D)


    #How this function is called:
    #skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.
    Return:
    cost -- the cost function value for the skip-gram model
    gradIn -- (inputVectors.shape)
    gradOut -- (outputVectors.shape)
    """
    cost = 0.0 # cost of all pairs inside of window
    gradIn = np.zeros(inputVectors.shape)   #(V,D)
    gradOut = np.zeros(outputVectors.shape) #(V,D)

    for i in contextWords:
        target_index = tokens[i]
        cost_, gradIn_, gradOut_ = word2vecCostAndGradient(inputVectors[tokens[currentWord]], target_index, outputVectors, dataset)
        cost += cost_
        gradIn[tokens[currentWord]] += gradIn_
        gradOut += gradOut_

    """ How our return values are used in the outer sum.
    cost += c / batchsize / denom               # WHY? Look at the cost function. It divides (averages) the cost by the number of whole vocabulary.
    grad[:N/2, :] += gin / batchsize0 / denom      
    grad[N/2:, :] += gout / batchsize / denom  
    """
    return cost, gradIn, gradOut 


    #How this function is called:
    # cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec
    Arguments/Return specifications: same as the skip-gram model
    """
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # Create one (1) vector which is a sum of all context words (tf.reduce_sum)
    one_context_vector = np.zeros_like(inputVectors[0])
    for i in contextWords:
        context_index = tokens[i]
        one_context_vector += inputVectors[context_index]
    
    one_context_vector = 1.0/(2*C)*one_context_vector
    target_index = tokens[currentWord]

    cost, gradIn_, gradOut_ = word2vecCostAndGradient(one_context_vector, target_index, outputVectors, dataset)

    gradIn = np.zeros(inputVectors.shape)
   
    for word in zip(contextWords): 
        gradIn[tokens[word[0]]] += gradIn_ / (2*C)

    gradOut += gradOut_
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
############### ##############################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:] # Why N/2? Because N is concatenation of input vector & output vector. We train 2 vectors (input/output) for one word.
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C) # set the context size as a random number
        centerword, context = dataset.getRandomContext(C1)
        """
        In [231]: getRandomContext(3)
        Out[231]: ('d', ['d', 'e', 'e', 'b', 'e', 'c']) # generating psuedo sentence like: devil egg eats behind ET cat.

        In [232]: getRandomContext(2)
        Out[232]: ('d', ['e', 'b', 'c', 'd'])

        In [233]: getRandomContext(1)
        Out[233]: ('d', ['b', 'e'])
        """
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1 # Why do we need this line?

    #   c, gin, gout = skipgram
    #      (currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    #        dataset, word2vecCostAndGradient=softmaxCostAndGradient):
        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom                # WHY? because cost function ! it divdes the cost by the number of whole vocabulary
        grad[:N/2, :] += gin / batchsize / denom     # gin is a matrix size of N/2
        grad[N/2:, :] += gout / batchsize / denom    # gout is a matrix size of N/2

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    # Where is vec?
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    #                           word2vec_sgd_wrapper
    #(word2vecModel, tokens, wordVectors, dataset, C,
    #                     word2vecCostAndGradient=softmaxCostAndGradient)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    #   skipgram (currentWord, C, contextWords,
    #   tokens,         inputVectors,        outputVectors,     dataset, word2vecCostAndGradient=softmaxCostAndGradient)
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], 
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()