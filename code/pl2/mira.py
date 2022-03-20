# mira.py
# -------


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        # For each element in Cgrid
        for c in Cgrid:
            # Iterate max_iterations
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, " for C value ", c
                # For each training data
                for i in range(len(trainingData)):
                    # Obtain the predicted class
                    predictedClass = self.classify([trainingData[i]])[0]
                    # Obtain the real class
                    realClass = trainingLabels[i]
                    # If the prediction is not correct
                    if predictedClass != realClass:
                        # Make copy of f (to not modify the real one)
                        f = trainingData[i].copy()
                        # Obtain tau
                        val = ((self.weights[predictedClass] - self.weights[realClass]) * f + 1.0) / 2.0 * (f * f)
                        tau = min(val, c)
                        # Multiply all the elements by tau, so now f is tau*f
                        for key in f.keys():
                            f[key] *= tau
                        # Adjust weights with tau*f
                        self.weights[realClass] += f
                        self.weights[predictedClass] -= f


    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


