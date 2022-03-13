# multiAgents.py
# --------------
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance
        # Initialise score
        score = 0
        # If successor state is win
        if successorGameState.isWin():
            score = float('inf')
        else:
            # Current position of Pacman
            pos = currentGameState.getPacmanPosition()
            # Calculate current food distances
            foodDistances = []
            for food in currentGameState.getFood().asList():
                foodDistances.append(manhattanDistance(pos, food))
            # Calculate successor food distances
            newFoodDistances = []
            for food in newFood.asList():
                newFoodDistances.append(manhattanDistance(newPos, food))
            # Calculate successor ghost distances
            newGhostDistances = []
            for ghostState in newGhostStates:
                newGhostDistances.append(manhattanDistance(newPos, ghostState.getPosition()))
            # Calculate difference of minimum food distances
            difMinFoodDistance = min(foodDistances) - min(newFoodDistances)
            # Calculate difference of current and successor
            difScore = successorGameState.getScore() - currentGameState.getScore()
            # If there is any ghost near reduce score
            if min(newGhostDistances) <= 1:
                score -= 15
            else:
                # If minimum food distance is reduced
                if difMinFoodDistance > 0:
                    score += 5
                # If score is reduced
                if difScore > 0:
                    score += 10
        return score



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimaxDecision(gameState):
            # Initialise values
            maxVal = float('-inf')
            minimaxAction = None
            # Obtain Pacman actions
            actions = gameState.getLegalActions(0)
            for action in actions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                pacmanVal = minValue(pacmanState, 0, 1)
                if pacmanVal > maxVal:
                    maxVal = pacmanVal
                    minimaxAction = action
            return minimaxAction

        def maxValue(gameState, currentDepth):
            # If is a final state or in actual depth
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            # Initialise values
            val = float('-inf')
            # Obtain Pacman actions
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                val = max(val, minValue(pacmanState, currentDepth, 1))
            return val

        def minValue(gameState, currentDepth, agentIndex):
            # If is a final state
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # Initialise values
            val = float('inf')
            ghostNum = gameState.getNumAgents() - 1
            # Obtain ghost actions
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                # Generate ghost successor given the action
                ghostState = gameState.generateSuccessor(agentIndex, action)
                # If is the last ghost
                if agentIndex == ghostNum:
                    val = min(val, maxValue(ghostState, currentDepth + 1))
                else:
                    val = min(val, minValue(ghostState, currentDepth, agentIndex + 1))
            return val

        return minimaxDecision(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaDecision(gameState):
            # Obtain Pacman actions
            actions = gameState.getLegalActions(0)
            # Initialise values
            maxVal = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            minimaxAction = None
            for action in actions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                pacmanValue = minValue(pacmanState, 0, 1, alpha, beta)
                if pacmanValue > maxVal:
                    maxVal = pacmanValue
                    minimaxAction = action
                alpha = max(alpha, maxVal)
            return minimaxAction

        def maxValue(gameState, currentDepth, alpha, beta):
            # If is a final state or in actual depth
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            val = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                val = max(val, minValue(pacmanState, currentDepth, 1, alpha, beta))
                if val > beta:
                    return val
                alpha = max(alpha, val)
            return val

        def minValue(gameState, currentDepth, agentIndex, alpha, beta):
            # If is a final state
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            val = float('inf')
            ghostNum = gameState.getNumAgents() - 1
            # Obtain ghost actions
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                # Generate ghost successor given the action
                ghostState = gameState.generateSuccessor(agentIndex, action)
                # If is the last ghost
                if agentIndex == ghostNum:
                    val = min(val, maxValue(ghostState, currentDepth + 1, alpha, beta))
                else:
                    val = min(val, minValue(ghostState, currentDepth, agentIndex + 1, alpha, beta))
                if val < alpha:
                    return val
                beta = min(beta, val)
            return val

        return alphaBetaDecision(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimaxDecision(gameState):
            # Obtain Pacman actions
            actions = gameState.getLegalActions(0)
            # Initialise values
            maxVal = float('-inf')
            expectimaxAction = None
            # For each Pacman action
            for action in actions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                pacmanVal = expectedValue(pacmanState, 0, 1)
                if pacmanVal > maxVal:
                    maxVal = pacmanVal
                    expectimaxAction = action
            return expectimaxAction

        def maxValue(gameState, currentDepth):
            # If is a final state or in actual depth
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            value = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                # Generate Pacman successor given the action
                pacmanState = gameState.generateSuccessor(0, action)
                value = max(value, expectedValue(pacmanState, currentDepth, 1))
            return value

        def expectedValue(gameState, currentDepth, agentIndex):
            # If is a final state
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            value = 0
            ghostNum = gameState.getNumAgents() - 1
            # Obtain ghost actions
            actions = gameState.getLegalActions(agentIndex)
            probability = 1.0 / len(actions)
            for action in actions:
                ghostState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == ghostNum:
                    value += maxValue(ghostState, currentDepth + 1) * probability
                else:
                    value += expectedValue(ghostState, currentDepth, agentIndex + 1) * probability
            return value

        return expectimaxDecision(gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Obtain score
    score = currentGameState.getScore()
    # Obtain pacman position
    pos = currentGameState.getPacmanPosition()
    # Obtain food amount
    foodAmount = len(currentGameState.getFood().asList())
    # Calculate sum of ghost distances
    ghostStates = currentGameState.getGhostStates()
    ghostDistances = []
    for ghostState in ghostStates:
        ghostDistances.append(manhattanDistance(pos, ghostState.getPosition()))
    sumGhostDistances = sum(ghostDistances)
    return score - foodAmount + sumGhostDistances

# Abbreviation
better = betterEvaluationFunction
