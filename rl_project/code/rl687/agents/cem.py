import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable,  epsilon:float=0.0001):
        #TODO
        self._name = "Cross_Entropy_Method"
        self._theta = theta
        self._bestTheta = theta
        self.originalTheta = theta
        self.sigma = sigma
        self._Sigma = self.getCovMatrix(self.sigma)
        self.numEpisodes = numEpisodes
        self.popSize = popSize
        self.numElite = numElite
        self.epsilon = epsilon
        self.evaluate = evaluationFunction
        self._JBest = -1*np.inf

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._bestTheta

    def sigma(self) -> float:
        pass

    def numEpisodes(self) -> int:
        pass

    def popSize(self) -> int:
        pass

    def numElite(self) -> int:
        pass

    def epsilon(self) -> float:
        pass

    def evaluate(self) -> Callable:
        pass

    def originalTheta(self) -> np.ndarray:
        pass

    def train(self)->np.ndarray:
        # print(self._JBest)
        dic = {}
        jbest = -np.inf
        besttheta = None
        for k in range(self.popSize):
            thetaPrime = self.normalSample(self._Sigma)
            jprime = np.mean(self.evaluate(thetaPrime, self.numEpisodes))
            dic[tuple(thetaPrime)] = jprime
            if jbest < jprime:
                jbest = jprime
                besttheta = thetaPrime

        # print(np.array([k for k in sorted(dic.keys(), reverse = True)]))

        kthetas = [np.array(k) for k in sorted(dic, key = dic.get, reverse = True)[:self.numElite]]


        self._theta = np.mean(kthetas, axis = 0)
        self._Sigma = self.createCovMatrix(kthetas)

        if jbest > self._JBest:
            self._JBest = jbest
            self._bestTheta = besttheta

        # print(self._JBest)
        return besttheta

    def normalSample(self, covMatrix):
        thetaArray = np.random.multivariate_normal(self.parameters, covMatrix)
        return thetaArray

    def getCovMatrix(self, sigmaExploratory):
        identity = np.identity(len(self.parameters))
        return sigmaExploratory*identity
    
    def createCovMatrix(self, kthetas):
        kthetasRemoved = kthetas - self._theta
        identity = np.identity(len(self._theta))

        sums = kthetasRemoved.T.dot(kthetasRemoved)

        # Alternate way to find this
        # sums2 = np.zeros((len(self._theta), len(self._theta)))
        # for thetaArray_k in kthetas:
        #     sums2 +=  (thetaArray_k-self._theta).reshape(len(self._theta),1).dot((thetaArray_k-self._theta).reshape(len(self._theta),1).T)

        # print('sums shape = ',sums.shape, 'sum2.shape = ',sums2.shape)
        # print('sums = ',sums, 'sum2 = ',sums2)

        sums += self.epsilon * identity

        covMatrix = (1/(self.epsilon + self.numElite))*(sums)
        return covMatrix
    

    def reset(self)->None:
        self._theta = self.originalTheta
        self._bestTheta = self.originalTheta
        self._Sigma = self.getCovMatrix(self.sigma)
        self._JBest = -np.inf
