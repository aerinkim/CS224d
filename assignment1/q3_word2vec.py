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
    grad -- the gradient with respect to all the other word vectors (D,V)
    """

    score_numerator = np.exp(outputVectors.dot(predicted)) # (V,D)*(D,1) = (V,1)
    score_denominator = np.sum(np.exp(outputVectors.dot(predicted))) #(scalar)
    probability = score_numerator / score_denominator # for every word, (V,1)

    cost = -np.log( probability[target] ) # the cost for the target word only. (scalar)
    
    probability[target] -= 1
    # These two lines are where everyone gets lost lol.
    gradPred = outputVectors.T.dot(probability) # (D,V)*(V,1)=(D,1) # Because it's d/dvc
    grad = np.outer(probability,predicted) # (V,1) -outer- (D,1) = (V,D) # d/dW
 
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    grad = np.zeros_like(outputVectors)

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K)) # why did you extend? the fuck?

    sampleVectors = outputVectors[indices[1:], :]

    p_target_predicted = sigmoid(outputVectors[target].dot(predicted)) # (1,D) * (D,1) = scalar
    p_k_predicted = -sigmoid(sampleVectors.dot(predicted)) # (K,D) * (D,1) = (K,1)

    cost = -np.log(p_target_predicted) -np.log(np.sum(p_k_predicted))

    gradPred = (p_target_predicted - 1)*outputVectors[target] + (1-p_k_predicted).T.dot(sampleVectors)
    # (1,D)              scalar                 (1,D)                   (K,1).T             (k,D)     

    grad[target] =  (p_target_predicted - 1)*predicted
    # (1,D)                 scalar             (1,D)  

    for i in xrange(K):
        k = indices[k+1]
        grad[k] =  (1 - pk_predicted[k])*predicted
        # (1,D)          scalar            (1,D)  
    return cost, gradPred, grad


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
    grad -- the gradient with respect to the word vectors
    """
    cost = 0.0 # cost of all pairs inside of window
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for i in contextWords:
        target_index = tokens[i]
        cost_, gradIn_, gradOut_ = word2vecCostAndGradient(inputVectors[tokens[currentWord]], target_index, outputVectors, dataset)
        cost += cost_
        grad_In[tokens[currentWord]] += gradIn_
        gradOut += gradOut_

    """ How cost, gradIn, gradOut are used in outer sum.
    cost += c / batchsize / denom                # WHY? because cost function ! it divdes the cost by the number of whole vocabulary
    grad[:N/2, :] += gi n / batchsize / denom      
    grad[N/2:, :] += gout / batchsize / denom  
    """
    return cost, gradIn, gradOut 


    #How this function is called:
    # cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # Create one (1) vector which is a sum of all context words (tf.reduce_sum)
    for i in contextWords:
        context_index = tokens[i]
        one_context_vector += inputVectors[context_index]
 
    target_index = tokens[currentWord]

    cost_, gradIn_, gradOut_ = word2vecCostAndGradient(one_context_vector, target_index, outputVectors, dataset)
    cost += cost_
    grad_In[tokens[currentWord]] += gradIn_
    gradOut += gradOut_
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

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
            denom = 1 # if it's CBOW then changes denom to what???

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