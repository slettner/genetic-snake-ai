# The snake sensor to perceive its environment.

import abc
import numpy as np
from ..util import Coordinate
from .snake_environment import LANDSCAPE_OBJECTS


class AbstractSnakeSensor(abc.ABC):

    """ Interface for snake sensors """

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sense(self, snake):
        """ Sense the environment """
        raise NotImplementedError


class DistanceSensor(AbstractSnakeSensor):

    """
    Gives distance to walls, snake and apple along eight directions: N, E, S, W and the bisectional directions
    The distance is 1/num_fields
    """

    def __init__(self, directions=None):
        """
        Construct Sensor. If directions is None it defaults to N, E, S, W and bisectional directions
        Args:
            directions(list): List of Coordinates indicating the direction to sense.
                              E.g. Coordinate(x=1, y=0) looks for each field in positive x direction
        """
        self.directions = directions
        if directions is None:
            self.directions = [
                Coordinate(1, 0),
                Coordinate(-1, 0),
                Coordinate(0, 1),
                Coordinate(0, -1),
                Coordinate(1, 1),
                Coordinate(-1, 1),
                Coordinate(1, -1),
                Coordinate(-1, -1)
            ]

    @property
    def name(self):
        return "DistanceSensor"

    def sense(self, snake):
        """
        Get the distance to all walls along north, east, south, west direction

        Args:
            snake(Snake): The snake
        """
        vision_array = np.zeros(3*len(self.directions))  # three values for each direction
        for i, direction in enumerate(self.directions):
            vision_array[3*i:3*i+3] = DistanceSensor.look_in_direction(snake, direction)

        return vision_array

    @classmethod
    def look_in_direction(cls, snake, direction):
        """
        Look into a direction to detect snake, apple or wall

        Args:
            snake(Snake): The snake
            direction(Coordinate): The direction vector.
        """
        current_position = snake.head
        vision = np.zeros(3)  # wall, apple, snake

        food_is_visible = False
        tail_is_visible = False
        distance = 0

        current_position += direction
        distance += 1

        while snake.landscape.size[0] > current_position.x >= 0 and snake.landscape.size[1] > current_position.y >= 0:

            if (not food_is_visible) and snake.landscape.world[current_position] == LANDSCAPE_OBJECTS["apple"]:
                vision[1] = 1/distance
                food_is_visible = True

            if (not tail_is_visible) and snake.landscape.world[current_position] == LANDSCAPE_OBJECTS["snake"]:
                vision[2] = 1/float(distance)
                tail_is_visible = True

            current_position += direction
            distance += 1

        vision[0] = 1/float(distance)
        return vision
