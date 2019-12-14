import numpy as np
from .bbo_agent import BBOAgent
import matplotlib.pyplot as plt
from scipy import stats

from typing import Callable


class FCHCM(BBOAgent):
    """
    First-choice hill-climbing Modified (FCHCM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, histories, numEpisodes:int=10):
        self._name = "First_Choice_Hill_Climbing"
        self.originalTheta = theta
        self._thetaArray = theta
        self.sigma = sigma
        self.numEpisodes = numEpisodes
        self._histories = histories
        self.evaluate = evaluationFunction
        self._JBest = np.mean(self.evaluate(self.parameters, self._histories, self.numEpisodes))

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._thetaArray

    def sigma(self) -> float:
        pass

    def numEpisodes(self) -> int:
        pass

    def theta(self) -> np.ndarray:
        pass

    def evaluate(self) -> Callable:
        pass

    def originalTheta(self) -> np.ndarray:
        pass


    def train(self)->np.ndarray:

        thetaPrime = self.normalSample(self.sigma)
        jprime, std = self.evaluate(thetaPrime, self._histories, self.numEpisodes)
        condition = jprime - 2*std/np.sqrt(50)*stats.t.ppf(1-0.05, 49)
        if jprime > self._JBest:
            print(jprime, std, condition)
            self._thetaArray = thetaPrime
            self._JBest = jprime

        return self._thetaArray

    def normalSample(self, sigmaExploratory):
        identity = np.identity(len(self.parameters))
        thetaArray = np.random.multivariate_normal(self.parameters, sigmaExploratory*identity)
        return thetaArray


    def reset(self)->None:
        #TODO
        self._thetaArray = self.originalTheta
