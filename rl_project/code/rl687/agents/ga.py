import numpy as np
from .bbo_agent import BBOAgent
import math
from typing import Callable
#np.random.seed(0)


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable,initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10, numStates:int = 25, numActions: int = 4, sigma = math.exp(0), alpha = 0.01):
        self._name = "Genetic_Algorithm"
        self._populationSize = populationSize
        self._evaluate = evaluationFunction
        self._initPopulationFunction = initPopulationFunction
        self._numElite = numElite
        # self._numParent = numParent
        self._numEpisodes = numEpisodes
        self._population = self._initPopulationFunction(populationSize)
        self._initPop = self._population
        self._sigma = sigma
        self._alpha = alpha
        self._bestTheta = self._population[0]
        self._bestJ = -np.inf

    # def normalSampleP(self, populationSize, sigma):
    #     covMatrix = (sigma) * np.identity(len(self._thetaInit))
    #     thetaArrays = [np.random.multivariate_normal(self._thetaInit, covMatrix) for x in range(populationSize)]
    #     return np.array(thetaArrays)

    @property
    def name(self)->str:
        #TODO
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._bestTheta

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        return parent + self._alpha * np.random.standard_normal(parent.size)

    def train(self)->np.ndarray:
        bestj = -10000
        bestTheta = self._bestTheta
        dic = {}
        for theta in self._population:
            jprime = np.mean(self._evaluate(theta, self._numEpisodes))
            if jprime > bestj:
                bestj = jprime
                bestTheta = theta
            dic[tuple(theta)] = jprime

        # print(dic)
        # print(np.array([dic[k] for k in sorted(dic.keys(),key = dic.get, reverse = True)]))

        eliteThetas = [np.array(k) for k in sorted(dic, key = dic.get, reverse = True)[:self._numElite]]

        # parentThetas = [np.array(k) for k in sorted(dic, key = dic.get, reverse = True)[:self._numParent]]


        for i in range(self._populationSize - self._numElite):
            pi = np.random.choice(range(self._numElite))
            eliteThetas.append(self._mutate(eliteThetas[pi]))

        if bestj > self._bestJ:
            self._bestJ = bestj
            self._bestTheta = bestTheta

        self._population = np.array(eliteThetas)
        return bestTheta

    def reset(self)->None:
        self._population = self._initPop
        self._bestTheta = self._population[0]
        self._bestJ = -np.inf
