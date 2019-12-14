import numpy as np
from .skeleton import Policy
from typing import Union
#np.random.seed(0)

class TabularSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._theta = np.zeros((numStates,numActions))
        self.policy = self.softmax()

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)
        self.policy = self.softmax()

    def softmax(self):
        theta = self._theta
        # #to takes care of numerical instability
        theta_exp = np.exp(theta - np.max(self._theta))
        theta_sum = np.sum(theta_exp, axis = 1)
        softmax = (theta_exp.T/theta_sum).T
        return softmax

    def policy(self) -> np.ndarray:
        pass


    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        if action == None:
            return self.getActionProbabilities(state)
        else:
            return self.policy[state][action]

    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        
        if isinstance(state, int):
            s = state
        else:
            s = int(np.where(state == 1)[0])

        return np.random.choice(np.array([0,1,2,3]), p = self.policy[s])

    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        if isinstance(state, int):
            s = state
        else:
            s = int(np.where(state == 1)[0])

        return self.policy[s]
