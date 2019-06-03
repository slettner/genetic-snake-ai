# Implements the snake

import gin
import copy
from genetic_snake.util import Coordinate
from .snake_environment import Landscape, LANDSCAPE_OBJECTS


@gin.configurable
class Snake(object):
    """ Represents the Snake """

    def __init__(self,
                 actions,
                 sensors,
                 policy,
                 landscape_size=(15, 15),
                 ):
        """
        Construct Snake.

        Args:
            landscape_size(tuple): Size of the landscape
            actions(list): List of AbstractSnakeAction
            sensors(list): List of AbstractSnakeSensor
            policy(AbstractSnakePolicy): The policy.
        """
        self.body = []  # a list of board pieces
        self.is_alive = True
        self.landscape = Landscape(size=landscape_size)
        self.give_birth()

        self.actions = actions
        self.sensors = sensors
        self.policy = policy

    def give_birth(self):
        """ create a snake with size and coordinates (0, 0) (1, 0), (2, 0)"""
        for i in range(3):
            self.body.append(
                    Coordinate(x=i, y=0)
            )
            self.landscape.world[self.body[-1]] = LANDSCAPE_OBJECTS["snake"]

    @property
    def size(self):
        return len(self.body)

    def act(self, action):
        """
        Make a step.

        Args:
            action(int): One of the snake actions

        Returns:

        """
        assert self.is_alive, "A dead snake can't act"
        self.actions[action].execute(self)

    def sense(self):
        """
        Sense landscape

        Returns
            perception(dict): Maps sensor name to sensor value
        """
        perception = {}
        for sensor in self.sensors:
            perception[sensor.name] = sensor.sense(self)
        return perception

    @property
    def head(self):
        assert self.size > 0, "Snake has no head"
        return copy.copy(self.body[-1])

    def move(self):
        """ Make a move. This includes sensing the landscape, deciding on an action and executing it"""
        perception = self.sense()
        decision = self.policy.decide(reason=perception)
        self.act(action=decision)
