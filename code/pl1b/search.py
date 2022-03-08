# search.py

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    from game import Directions

    # Define list for visited positions
    visited = []
    # Create a stack
    stack = Stack()
    # Obtain start position
    initial = problem.getStartState()
    # Push initial position and empty path to the stack
    stack.push((initial, []))
    # While there are options to search
    while not stack.isEmpty():
        # Obtain last pushed state
        (actual, path) = stack.pop()
        # Mark actual as visited
        visited.append(actual)
        # If finished return path
        if problem.isGoalState(actual):
            return path
        # Obtain all succesors of actual position
        succesors = problem.getSuccessors(actual)
        # Iterate all succesors
        for s in succesors:
            # If next position has not been visited yet
            if s[0] not in visited:
                # Push actual and path to that position
                stack.push((s[0], path + [s[1]]))
                # Uncomment to visualize the algorithm execution
                # print("Actual: ", actual, " - ", s[1], " - Next: ", s[0], "Path:", path)
    # Otherwise, there is no path
    return [Directions.STOP]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    from game import Directions

    # Define list for visited positions
    visited = []
    # Create a queue
    queue = Queue()
    # Obtain start position
    initial = problem.getStartState()
    # Push initial position and empty path to the queue
    queue.push((initial, []))
    #Add inital position to visited
    visited.append(initial)
    # While there are options to process
    while not queue.isEmpty():
        # Obtain last pushed state
        (actual, path) = queue.pop()
        # If finished return path
        if problem.isGoalState(actual):
            return path
        # Obtain all succesors of actual position
        succesors = problem.getSuccessors(actual)
        # Iterate all succesors
        for s in succesors:
            # If next position has not been visited yet
            if s[0] not in visited:
                # Mark next as visited
                visited.append(s[0])
                # Push position and the path to that position
                queue.push((s[0], path + [s[1]]))
                # Uncomment to visualize the algorithm execution
                # print("Actual: ", actual, " - ", s[1], " - Next: ", s[0], "Path:", path)
    # Otherwise, there is no path
    return [Directions.STOP]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    from game import Directions

    # Define list for visited positions
    visited = []
    # Create a priority queue
    pqueue = PriorityQueue()
    # Obtain start position
    initial = problem.getStartState()
    # Push initial position and empty path to the queue with 0 cost
    pqueue.push((initial, [], 0), 0)
    # While there are options to process
    while not pqueue.isEmpty():
        # Obtain last pushed state
        (actual, path, cost) = pqueue.pop()
        # Check that node has not been already processed
        if actual not in visited:
            # Mark actual as visited
            visited.append(actual)
            # If finished return path
            if problem.isGoalState(actual):
                return path
            # Obtain all succesors of actual position
            succesors = problem.getSuccessors(actual)
            # Iterate all succesors
            for s in succesors:
                # If next position has not been visited yet
                if s[0] not in visited:
                    # Push position and the path to that position
                    pqueue.push((s[0], path + [s[1]], cost + s[2]), cost + s[2])
                    # Uncomment to visualize the algorithm execution
                    # print("Actual: ", actual, " - ", s[1], " - Next: ", s[0], " - Cost: ", s[2], " - Path:", path, " - Path cost:", cost)
    # Otherwise, there is no path
    return [Directions.STOP]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    from game import Directions
    Comienzo = problem.getStartState()

    if problem.isGoalState(Comienzo):
        return []

    visited = []
    cola = PriorityQueue()

    cola.push((Comienzo, [], heuristic(Comienzo, problem)), heuristic(Comienzo, problem))

    while not cola.isEmpty():
        actual, path, cost = cola.pop()
        if actual not in visited:
            visited.append(actual)

            if problem.isGoalState(actual):
                return path

            sucessors=problem.getSuccessors(actual)

            for s in sucessors:
                if s[0] not in visited:
                    cola.push((s[0], path + [s[1]], cost + s[2]),  cost + s[2] + heuristic(s[0], problem))

    return [Directions.STOP]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
