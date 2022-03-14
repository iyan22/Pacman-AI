# classificationAgents.py
# -----------------------


from pacman import Directions
from game import Agent

import random
import game
import util

class DummyOptions:
    def __init__(self):
        self.data = "pacman"
        self.training = 25000
        self.test = 100
        self.odds = False
        self.weights = False


import perceptron_pacman

class ClassifierAgent(Agent):
    def __init__(self, trainingData=None, validationData=None, classifierType="perceptron", agentToClone=None, numTraining=3):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']
        if(classifierType == "perceptron"):
            classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,numTraining)
        self.classifier = classifier
        self.featureFunction = enhancedFeatureExtractorPacman
        args = {'featureFunction': self.featureFunction,
                'classifier':self.classifier,
                'printImage':None,
                'trainingData':trainingData,
                'validationData':validationData,
                'agentToClone': agentToClone,
        }
        options = DummyOptions()
        options.classifier = classifierType
        runClassifier(args, options)
    def getAction(self, state):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman
        features = self.featureFunction(state)
        
        action =  self.classifier.classify([features])[0]

        return action

def scoreEvaluation(state):
    return state.getScore()
