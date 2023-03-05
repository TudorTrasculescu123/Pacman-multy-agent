# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from node import Node

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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


    def bfs(self, startState, goalState, gameState: GameState):
        from game import Directions
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        walls = gameState.getWalls()
        dir = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        myCollection = util.Queue()
        node = Node(startState, None, Directions.SOUTH)
        statesVisited = []
        statesVisited.append(node.getState())
        myCollection.push(node)
        while not myCollection.isEmpty():
            node = myCollection.pop()
            currentState = node.getState()
            if currentState == goalState:
                break
            for d in range(4):
                (x, y) = (dx[d] + currentState[0],  dy[d] + currentState[1])
                if not walls[x][y] and (x, y) not in statesVisited:
                    statesVisited.append((x, y))
                    myCollection.push(Node((x, y), node, dir[d]))
        actionList = []
        while node.getParent() != None:
            actionList.append(node.getAction())
            node = node.getParent()
        ##actionList.reverse()
        return len(actionList)


    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition()                                    ## tuple representing the new position for pacman
        newFood = successorGameState.getFood().asList()                                    ## list of tuples with the food coordinates
        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghostState.configuration.getPosition() for ghostState in newGhostStates] ## list of the current positions for each ghost
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]         ## list of scared times for each ghost
        "*** YOUR CODE HERE ***"
        lstLenToGhosts = [self.bfs(newPos, e, currentGameState) for e in ghostPositions]
        minDistToGhost = 1
        if len(lstLenToGhosts) > 0:
            minDistToGhost = min(lstLenToGhosts)
            indexGhost = lstLenToGhosts.index(minDistToGhost)
        minDistToFood = 1
        lstLenToFood = [self.bfs(newPos, e, currentGameState) for e in newFood]
        if len(lstLenToFood) > 0:
            minDistToFood = min(lstLenToFood)
        if minDistToGhost < 2: minDistToGhost = -30
        elif minDistToGhost < 5: minDistToGhost = -5
        elif minDistToGhost < 10: minDistToGhost = 15
        else: minDistToGhost = 25
        return successorGameState.getScore() + (5.0 / minDistToFood) + minDistToGhost ##+ (5 * newScaredTimes[indexGhost])

def scoreEvaluationFunction(currentGameState: GameState):
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

    def minimax(self, gamestate: GameState, depth, maximizingPlayer, nrGhosts):
        if depth == 0 or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate), ""
        if maximizingPlayer:
            value = float("-inf")
            bestAction = Directions.STOP
            legalActions = gamestate.getLegalActions(0)
            for child in legalActions:
                newGameState = gamestate.generateSuccessor(0, child)
                newVal, act = self.minimax(newGameState, depth, False, 1)
                if newVal > value:
                    value = newVal
                    bestAction = child
            return value, bestAction
        else:
            value = float("inf")
            bestAction = Directions.STOP
            legalActions = gamestate.getLegalActions(nrGhosts)
            for child in legalActions:
                newGameState = gamestate.generateSuccessor(nrGhosts, child)
                if nrGhosts == gamestate.getNumAgents() - 1:
                    newVal, act = self.minimax(newGameState, depth - 1, True, 0)
                else:
                    newVal, act = self.minimax(newGameState, depth, False, nrGhosts + 1)
                if newVal < value:
                    value = newVal
                    bestAction = child
            return value, bestAction


    def getAction(self, gameState: GameState):
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

        val, action = self.minimax(gameState, self.depth, True, 0)
        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta(self, gamestate: GameState, depth, maximizingPlayer, nrGhosts, alpha = float("-inf"), beta = float("inf")):
        if depth == 0 or gamestate.isWin() or gamestate.isLose():
            return self.evaluationFunction(gamestate), ""
        if maximizingPlayer:
            value = float("-inf")
            bestAction = Directions.STOP
            legalActions = gamestate.getLegalActions(0)
            for child in legalActions:
                newGameState = gamestate.generateSuccessor(0, child)
                newVal, act = self.alpha_beta(newGameState, depth, False, 1, alpha, beta)
                if newVal > value:
                    value = newVal
                    bestAction = child
                if value > beta:
                    return value, child
                alpha = max(alpha, value)
            return value, bestAction
        else:
            value = float("inf")
            bestAction = Directions.STOP
            legalActions = gamestate.getLegalActions(nrGhosts)
            for child in legalActions:
                newGameState = gamestate.generateSuccessor(nrGhosts, child)
                if nrGhosts == gamestate.getNumAgents() - 1:
                    newVal, act = self.alpha_beta(newGameState, depth - 1, True, 0, alpha, beta)
                else:
                    newVal, act = self.alpha_beta(newGameState, depth, False, nrGhosts + 1, alpha, beta)
                if newVal < value:
                    value = newVal
                    bestAction = child
                if value < alpha:
                    return value, child
                beta = min(beta, value)
            return value, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        val, act = self.alpha_beta(gameState, self.depth, True, 0)
        return act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
