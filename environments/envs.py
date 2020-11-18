import numpy as np

class Environment:
    "Abstract Environment class"
    num_observations: int # number of states to pass to the representation network
    num_actions: int
    MDP: bool

    def __init__(self):
        self.state = None

    def from_state(self, state):
        self.state = np.array(state)

    def end(self):
        raise NotImplementedError()

class Game(Environment):
    num_players: int
    pass

class MDP(Environment):
    num_players = 1
    pass