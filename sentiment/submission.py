#!/usr/bin/python

from collections import defaultdict
import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    features = dict()
    for word in x.split():
        features[word] = features[word] + 1 if word in features else 1
    return features

    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes: 
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. 
    - The identity function may be used as the featureExtractor function during testing.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = defaultdict(float)  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    
    def predictor(x) -> int:
        return 1 if dotProduct(featureExtractor(x), weights) >= 0 else -1

    for _ in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            if y * dotProduct(weights, features) <= 0:
                increment(weights, eta * y, features)

        print(f'Train loss: {evaluatePredictor(trainExamples, predictor)}, Validation loss: {evaluatePredictor(validationExamples, predictor)}')

    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        
        phi = {k: random.random() for k in weights.keys()}
        y = 1 if dotProduct(phi, weights) >= 0 else -1
        
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)

        x_stripped = x.replace(' ', '')
        n_grams = [x_stripped[i:i + 3] for i in range(len(x) - 3)]
        n_gram_counts = defaultdict(int)
        n_gram_counts = {x:n_gram_counts[x] + 1 for x in n_grams}
        
        return n_gram_counts

        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f:
def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor,
                        weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 4: k-means
############################################################


def kmeans(examples: List[Dict[str, float]], K: int, maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 27 lines of code, but don't worry if you deviate from this)
    
    def assign(x):
        distances = [sum([abs(x[key] - c[key])**2 for key in c]) for c in centroids]
        return distances.index(min(distances))
    
    def calcCentroids():
        assignedCount = [0] * len(centroids)

        for i, x in enumerate(examples):
            increment(centroids[z[i]], 1, x)
            assignedCount[z[i]] += 1
        for i, c in enumerate(centroids):
            for feat in c:
                c[feat] /= assignedCount[i]
    
    centroids = random.sample(examples, K)
   
    z = []
    for i in range(maxEpochs):
        z, z_old = [assign(x) for x in examples], z
        if z == z_old:
            break
        calcCentroids()

    loss = 0
    for i, x in enumerate(examples):
        loss += sum([abs(x[key] - centroids[z[i]][key])**2 for key in centroids[z[i]]])

    return (centroids, z, 4)
    
    # END_YOUR_CODE
