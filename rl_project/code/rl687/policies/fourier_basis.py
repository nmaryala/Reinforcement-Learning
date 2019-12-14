import numpy as np
from .skeleton import Policy
from typing import Union
import math
np.random.seed(0)
import itertools


class FourierBasis(Policy):
    """
    A Fourier Basis Policy (bs)


    Parameters
    ----------
    numDimensions (int): the number of dimensions of states the Fourier Basis policy has
    numActions (int): the number of actions the Fourier Basis policy has
    order (int) : order of the fourier basis
    """

    def __init__(self, numDimensions:int, numActions: int, order: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._numDimensions = numDimensions
        self._numActions = numActions
        self._order = order
        
        self._C = np.array(list( itertools.product(range(order+1), repeat=self._numDimensions) ),
                   dtype=np.int32)
        self._theta = np.zeros((numActions, self._C.shape[0]))



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

    def samplAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        # phi = np.zeros(self._C.shape[0])

        # state[0] = (state[0] + 3) / 6
        # state[1] = (2 / (1 + np.exp(-state[1]))) - 1
        # state[2] = (state[2] + np.pi / 12 - 2*np.pi) / (np.pi / 6)
        # # print("state", state[2])
        # state[3] = (2 / (1 + np.exp(-state[3]))) - 1
        phi = np.cos(np.pi * np.dot(self._C, state))       

        # print('phi=',phi)
        # # print('theta=',self._theta)
        pi = np.zeros(self._numActions)

        scores = self._theta.dot(phi)
        scores -= np.max(scores)

        pis = np.array([math.exp(score) for score in scores])
        pisum = sum(pis)
        pi = pis/pisum
        # print('pi=', pi)

        return np.random.choice(np.array([0,1]), p = pi)

    def __call__(self, state:np.ndarray, action=None)->Union[float, np.ndarray]:
        pass

    def pi(self, state:np.ndarray, action) -> float:
        phi = np.zeros(self._C.shape[0])
        # print( 'dummy', self._C.shape)
        # print('nik', state.shape)

        # state[0] = (state[0] + 3) / 6
        # state[1] = (2 / (1 + np.exp(-state[1]))) - 1
        # state[2] = (state[2] + np.pi / 12) / (np.pi / 6)
        # if state[2] > 2*np.pi:
        #     state[2] -= 2*np.pi
        # elif state[2] < -2*np.pi:
        #     state[2] += 2*np.pi
        # # print("state", state[2])
        # state[3] = (2 / (1 + np.exp(-state[3]))) - 1
        phi = np.cos(np.pi * np.dot(self._C, state))       

        # print('phi=',phi)
        # # print('theta=',self._theta)
        pi = np.zeros(self._numActions)

        scores = self._theta.dot(phi)
        scores -= np.max(scores)

        pis = np.array([math.exp(score) for score in scores])
        pisum = sum(pis)
        pi = pis/pisum
        # print('pi=', pi)

        return pi[action]
