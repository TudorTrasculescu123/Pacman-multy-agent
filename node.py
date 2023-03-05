class Node:
    def __init__(self, state, parent, action, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def getCost(self):
        return self.cost

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False