import numpy as np
import math
from typing import Callable
from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
# import matplotlib.pyplot as plt
from rl687.agents.fchc import FCHC
from rl687.agents.fchcm import FCHCM
from rl687.agents.cem import CEM
from rl687.agents.cemm import CEMM
from rl687.agents.ga import GA
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.fourier_basis import FourierBasis
from multiprocessing import Pool
import multiprocessing as mp
import time
import pickle
import statistics 
import itertools
from scipy import stats
# import cma
from numpy import array, dot, isscalar, sum  # sum is not needed
import numpy as np
from joblib import Parallel, delayed
# from scipy.optimize import minimize, maximize
num_cores = mp.cpu_count()

class Hp():

    def __init__(self):
        self.nb_steps = 100
        self.num_episodes = 100000
        self.learning_rate = 0.08
        self.nb_directions = 5
        self.nb_best_directions = 5
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.05
        self.seed = 1729
        self.env_name = "Cartpole"
        self.dimensions = 4
        self.order = 1
        self.numActions = 2
        self.histories_c = None
        self.histories_s = None
        self.pi = None
        self.initial_theta = None
        self.tinv = stats.t.ppf(1-0.01, self.num_episodes - 1)
        self.sqrt = np.sqrt(self.num_episodes)


class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class FourierBasis():

    def __init__(self, numDimensions:int, numActions: int, order: int, hp):
        self._numDimensions = numDimensions
        self._numActions = numActions
        self._order = order
        
        self._C = np.array(list( itertools.product(range(order+1), repeat=self._numDimensions) ),
                   dtype=np.int32)
        self.theta = np.random.randn(numActions, self._C.shape[0])*0.001
        self.hp = hp
        self.dic = {}

    @property
    def parameters(self)->np.ndarray:
        """
        Return the fourierBasis parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()

    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the fourierBasis parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self.theta = p.reshape(self.theta.shape)
        self.dic = {}

    def evaluate(self, state, delta=None, direction=None):
        phi = np.cos(np.pi * np.dot(self._C, state))       
        scores = self.theta.dot(phi)

        if direction is None:
            score = self.theta.dot(phi)
        elif direction == "positive":
            score = (self.theta + self.hp.noise * delta).dot(phi)
        else:
            score = (self.theta - self.hp.noise * delta).dot(phi)

        scores -= np.max(scores)

        pis = np.array([math.exp(score) for score in scores])
        pisum = sum(pis)
        pi = pis/pisum

        return np.random.choice(np.array([0,1]), p = pi)

    def pi(self, state, action, delta=None, direction=None, base = False):
        if  base and tuple(state) in self.dic:
            return self.dic[tuple(state)][action]

        phi = np.cos(np.pi * np.dot(self._C, state))       

        # scores = None
        if not direction:
            scores = self.theta.dot(phi)  # 81 dimension
        elif direction == "positive":
            scores = (self.theta + self.hp.noise * delta).dot(phi)
        elif direction == "negative":
            scores = (self.theta - self.hp.noise * delta).dot(phi)

        scores -= np.max(scores)

        pis = np.array([math.exp(score) for score in scores])
        pisum = sum(pis)
        pi = pis/pisum

        if base:
            self.dic[tuple(state)] = pi
        return pi[action] 
   

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.hp.learning_rate / (self.hp.nb_best_directions * sigma_r) * step


#Global variable defined for usage
hp = Hp()
fourierBasis_b = FourierBasis(1,2,1, hp)
historiesGlobal = None
bestTheta = None
bestPdis = None
bestCondition = None

#Main function to run all the functionalities
def main():
    name = input("Enter problem number to solve: ")
    if name == '2':
        ReadData()
    if name == '3':
        runARSPhil()
    if name == '4':
        ttest()
    if name == '5':
        csv()
    if name == '6':
        ttestcsv()
    if name == '7':
        csvfinal()

def runARSPhil():
    hp = ReadData()
    np.random.seed(hp.seed)

    fourierBasis_e = FourierBasis(hp.dimensions,hp.numActions, hp.order, hp)
    fourierBasis_e.parameters = np.array( hp.initial_theta)
    histories_c = hp.histories_c
    print(len(histories_c))

    rewards = [sum([history[3*i+2] for i in range(len(history)//3)]) for history in histories_c ]
    c = np.mean(rewards)

    print("initial mean= ", c)


    _, reward_evaluation, std = explore(histories_c, fourierBasis_e ,c, hp, -1, dummy= True)

    condition = reward_evaluation- 2*(std/hp.sqrt) *(hp.tinv) 
    print('firsttheta:', fourierBasis_e.theta, 'Reward:', reward_evaluation, "Std:", std, "condition:", condition)

    trainParallel(hp.histories_c, fourierBasis_e, hp, c)

    # reward_evaluation, std = explore(histories, fourierBasis_e ,0,  hp)
    fourierBasis_final = FourierBasis(hp.dimensions,hp.numActions, hp.order, hp)
    fourierBasis_final.parameters = bestTheta
    _, reward_evaluation, std = explore(hp.histories_s, fourierBasis_final ,0,  hp, -1)

    print(bestTheta)
    print(fourierBasis_e.theta)

    condition = reward_evaluation- (std/hp.sqrt) *(hp.tinv) 
    print('final rewards =','Reward:', reward_evaluation, "Std:", std, "condition:", condition)


def csv():
    mylist = list(bestTheta.flatten())
    print(mylist)
    import csv
    for i in range(1,101):
        myfile=open('policies/'+str(i)+'.csv','w')
        writer=csv.writer(myfile)
        writer.writerow(mylist)

def csvfinal():
    mylist = [
        [2.3744528  , 0.48452176, -1.16195096,  0.7636471],
        [ 10.94474482  , 1.31346028, -10.40373693 ,  0.0780188 ],
        [ 10.81175035, 1.23094545, -10.31595507, 0.05160196],
        [ 9.75103712, 1.18350732,-9.40777947, -0.02202357],
        [ 2.32091001, 0.28636922, -0.99284319, 0.70911084],
        [54.19416043 ,8.78633034, 11.32974581, 4.8266031 ],
        [24.70204625, 2.3376134, 8.61086141 ,-2.88533523],
        [ 8.54861582  ,0.95241176, -8.10974945  ,0.22871519],
        [ 7.92441664  ,1.03198814, -7.66352574  , 0.21653298],
        [15.21909514 , 0.84318028 , 4.03946422,  1.12079851]
    ]
    print(mylist)
    import csv
    for i in range(10):
        for j in range(10):
            myfile=open('policies/'+str(i*10+j+1)+'.csv','w')
            writer=csv.writer(myfile)
            writer.writerow(mylist[i])



def ttest():
    import sys
    print("Give theta to test")
    args = [float(x) for x in sys.argv[1:]]
    print(args)
    args = np.array(args)

    hp = ReadData()
    np.random.seed(hp.seed)

    histories_c = hp.histories_c
    print(len(histories_c))

    rewards = [sum([history[3*i+2] for i in range(len(history)//3)]) for history in histories_c ]
    c = np.mean(rewards)

    print("initial mean= ", c)


    fourierBasis_final = FourierBasis(hp.dimensions,hp.numActions, hp.order, hp)
    fourierBasis_final.parameters = args
    _, reward_evaluation, std = explore(hp.histories_s, fourierBasis_final ,0,  hp, -1)

    print(args)

    condition = reward_evaluation- (std/hp.sqrt) *(hp.tinv) 
    print('final rewards =','Reward:', reward_evaluation, "Std:", std, "condition:", condition)


def ttestcsv():
    import sys,csv
    args = []

    for i in range(1,60):
        if i  != 31:
            fileName ='master_file/'+str(i)+'.csv' 
            with open(fileName) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                count = 0
                for row in csv_reader:
                    if count == 0:
                        args.append([float(x) for x in row])
                    count += 1

    hp = ReadData()
    np.random.seed(hp.seed)

    histories_c = hp.histories_c
    print(len(histories_c))

    rewards = [sum([history[3*i+2] for i in range(len(history)//3)]) for history in histories_c ]
    c = np.mean(rewards)

    print("initial mean= ", c)


    fourierBasis_final = FourierBasis(hp.dimensions,hp.numActions, hp.order, hp)

    # print(args)
    for arg in args:
        arg = np.array(arg)

        fourierBasis_final.parameters = arg
        _, reward_evaluation, std = explore(hp.histories_s, fourierBasis_final ,0,  hp, -1)

        print(arg)

        condition = reward_evaluation- (std/hp.sqrt) *(hp.tinv) 
        print('final rewards =','Reward:', reward_evaluation, "Std:", std, "condition:", condition)


def trainParallel(histories, fourierBasis, hp, c):
    for step in range(hp.nb_steps):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = fourierBasis.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        n = mp.cpu_count()
        lists1 = [(histories, fourierBasis,c, hp, k,  "positive", deltas[k]) for k in range(hp.nb_directions)]
        with Pool(processes= n) as pool:
            positive_rewards = pool.starmap(explore,  lists1)

        lists2 = [(histories, fourierBasis,c, hp, k, "negative", deltas[k]) for k in range(hp.nb_directions)]
        with Pool(processes= n) as pool:
            negative_rewards = pool.starmap(explore,  lists2)

        # print(positive_rewards)
        # print(negative_rewards)

        dic1 = {}
        for rew in positive_rewards:
            dic1[rew[0]] = rew[1]

        dic2 = {}
        for rew in negative_rewards:
            dic2[rew[0]] = rew[1]

        positive_rewards = [dic1[k] for k in range(hp.nb_directions)]
        negative_rewards = [dic2[k] for k in range(hp.nb_directions)]

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our fourierBasis
        fourierBasis.update(rollouts, sigma_r)

        # Printing the final reward of the fourierBasis after the update
        _, reward_evaluation, std = explore(histories, fourierBasis,c, hp, -1)

        condition = reward_evaluation- 2*(std/hp.sqrt) *(hp.tinv) 
        print('Step:', step, 'theta:', fourierBasis.theta, 'Reward:', reward_evaluation, "Std:", std, "condition:", condition)

def ReadData():
    import csv
    with open('data.csv') as f:
        reader = csv.reader(f)
        data = list(reader)

    print(len(data))
    m = int(data[0][0])
    numActions = int(data[1][0])
    order = int(data[2][0])
    thetas = data[3]
    initial_theta = []
    for theta in thetas:
        initial_theta.append(float(theta))
    totalEpisodes = int(data[4][0])
    totalHistories = []
    for i in range(totalEpisodes):
        totalHistories.append(data[5+i])
    histories = getHistoriesFromData(totalHistories)
    print("dimension=",m)
    print("numActions=",numActions)
    print("order=",order)
    print("totalEpisodes=",totalEpisodes)
    print("initial_theta=",initial_theta)
    pi = [float(i) for i in  data[-1]]
    print("pi=",pi)
    firstEpisode = histories[0]
    print("firstEpisode=",firstEpisode)
    hp = Hp()
    hp.dimensions = m
    hp.numActions = numActions
    hp.order = order
    hp.env_name = 'philWorld'
    hp.histories_c = histories[:totalEpisodes//2]
    hp.histories_s = histories[totalEpisodes//2:]
    hp.pi = pi
    hp.initial_theta = initial_theta
    hp.num_episodes = totalEpisodes//2

    rewards = [sum([history[3*i+2] for i in range(len(history)//3)]) for history in histories[-100:]]
    print("rewards=", np.mean(rewards))


    return hp

def getHistoriesFromData(totalHistories):
    finalHistories = []
    for history in totalHistories:
        finalHistory = []
        for i in range(len(history)//3):
            state = np.array([float(history[3*i])])
            action = int(history[3*i+1])
            reward = float(history[3*i+2])
            finalHistory.append(state)
            finalHistory.append(action)
            finalHistory.append(reward)
        finalHistories.append(finalHistory)
    return finalHistories


def explore(histories, fourierBasis ,c ,hp ,k ,direction=None, delta = None, dummy = True):
    global bestPdis, bestTheta, bestCondition, fourierBasis_b
    print(direction, k)
    fourierBasis_b.parameters = np.array(hp.initial_theta)
    returns = []

    for history in histories:
        ret = 0
        prod = 1
        for t in range(len(history)//3):
            state = history[3*t]
            action = history[3*t+1]
            reward = history[3*t+2]

            prod *= fourierBasis.pi(state, action, delta, direction)/fourierBasis_b.pi(state, action, base=True)
            ret += prod * reward

        returns.append(ret)

    condition = np.mean(returns)- 2*(np.std(returns)/hp.sqrt) *(hp.tinv) 
    if condition > c or dummy:
        if bestPdis == None:
            bestPdis = np.mean(returns)
            bestTheta = np.array(list(fourierBasis.theta))
            bestCondition = condition
        elif np.mean(returns) > bestPdis:
            bestPdis = np.mean(returns)
            bestTheta = np.array(list(fourierBasis.theta))
            bestCondition = condition
        return k, np.mean(returns), np.std(returns)
    else:
    	return k, (condition-c), 1e-5


#Function to calculate PDIS for a given set of fourierbasis fourierBasis
def calculateCrudePDIS(i, direction, delta, fourierBasis):
    returns = []
    factor = len(historiesGlobal)//8
    his = historiesGlobal[i*factor:i*factor + factor ]
    for history in his:
        ret = 0
        prod = 1
        for t in range(len(history)//3):
            prod = 1

            for j in range(t+1):
                state = history[3*j]
                action = history[3*j+1]
                reward = history[3*j+2]
                prod *= fourierBasis.pi(state, action, delta, direction)/fourierBasis_b.pi(state, action, base=True)

            ret += prod * reward

        returns.append(ret)
    return returns

#Function to calculate PDIS for a given set of fourierbasis fourierBasis
def calculatePDIS(i, direction, delta, fourierBasis):
    returns = []
    factor = len(historiesGlobal)//8
    his = historiesGlobal[i*factor:i*factor + factor ]
    for history in his:
        ret = 0
        prod = 1
        for t in range(len(history)//3):
            state = history[3*t]
            action = history[3*t+1]
            reward = history[3*t+2]

            prod *= fourierBasis.pi(state, action, delta, direction)/fourierBasis_b.pi(state, action, base = True)
            ret += prod * reward

        returns.append(ret)
    return returns

def runHistory(getAction, numeps=10000):
    histories = []

    cartPole = Cartpole()
    for ep in range(numeps):
        history = []
        cartPole.reset()
        history.append(cartPole.state)
        step = 0
        while not cartPole.isEnd:
            s, r, e = cartPole.step(getAction(cartPole.state))
            history.append(cartPole.action)
            history.append(cartPole.reward)
            history.append(cartPole.state)
        histories.append(history)

    return histories


	
if __name__ == '__main__':
	main()