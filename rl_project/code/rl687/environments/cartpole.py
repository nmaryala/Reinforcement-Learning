import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.
    Actions: left (0) and right (1)
    Reward: 1 always
    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        self.F = 10

        # TODO: properly define the variables below
        self._action = None
        self._reward = 1
        self._isEnd = 0
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable

    @property
    def name(self)->str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> np.ndarray:
        return np.array([self._x, self._v, self._theta, self._dtheta])

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        x = state[0]
        v = state[1]
        theta = state[2]
        dtheta = state[3]
        F = 0
        if action == 0:
            F = -self.F
        else:
            F = self.F

        ddtheta = (self._g*np.sin(theta) + np.cos(theta)*((-F-self._mp*self._l*(dtheta**2)*np.sin(theta))/(self._mp + self._mc)))/(self._l*(4/3-(self._mp * (np.cos(theta)**2)/(self._mc + self._mp))))

        dv = (F + self._mp*self._l*((dtheta**2)*np.sin(theta) - ddtheta*np.cos(theta)))/(self._mp + self._mc)

        xnew = x + self._dt * v
        vnew = v + self._dt * dv
        thetanew = theta + self._dt * dtheta
        dthetanew = dtheta + self._dt * ddtheta

        return np.array([xnew, vnew, thetanew, dthetanew])


    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        # TODO
        return 1

        # if self.terminal():
        #     self._reward = 0
        #     return 0

        # if abs(nextState[2]) > np.pi/12 or abs(nextState[0]) >= 3:
        #     self._reward = 0
        #     return 0
            
        # self._reward = 1
        # return 1


    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # TODO
        newstate = self.nextState(self.state, action)
        reward = self.R(self.state, action, newstate)

        self._t += self._dt
        self._action = action
        self._x = newstate[0]
        self._v = newstate[1]
        self._theta = newstate[2]
        self._dtheta = newstate[3]
        self._isEnd = self._t > 20.0 or abs(self._theta) > np.pi/12 or abs(self._x) >= 3
        self._reward = reward


        return (self.state, reward, self.terminal())



    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        self._name = "Cartpole"
        self.F = 10

        # TODO: properly define the variables below
        self._action = None
        self._reward = 1
        self._isEnd = 0
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        self._isEnd = self._t > 20 or abs(self._theta) > np.pi/12 or abs(self._x) >= 3
        return self._t > 20 or abs(self._theta) > np.pi/12 or abs(self._x) >= 3 

