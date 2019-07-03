# The snake sensor to perceive its environment.

import gin
import abc
import numpy as np
from ..util import Coordinate
from .snake_environment import LANDSCAPE_OBJECTS


class AbstractSnakeSensor(abc.ABC):

    """ Interface for snake sensors """

    @property
    @abc.abstractmethod
    def size(self):
        """ Size of the sensor output """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sense(self, snake):
        """ Sense the environment """
        raise NotImplementedError


@gin.configurable
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

    @property
    def size(self):
        return len(self.directions) * 3

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


@gin.configurable
class MinimalSensor(AbstractSnakeSensor):

    """ Use fewer values to represent the state """
    def __init__(self):
        self.look_left = np.array([[0, 1], [-1, 0]])
        self.look_right = np.array([[0, -1], [1, 0]])
        self.look_straight = np.array([[1, 0], [0, 1]])

    @property
    def name(self):
        return "MinimalSensor"

    @property
    def size(self):
        return 7

    def sense(self, snake):
        """

        Args:
            snake:

        Returns:

        """
        heading = np.array([snake.heading.x, snake.heading.y])
        heading_left = np.matmul(self.look_left, heading)
        heading_right = np.matmul(self.look_right, heading)
        heading_straight = np.matmul(self.look_straight, heading)

        snake_head = snake.head

        loc_left = Coordinate(x=heading_left[0]+snake_head.x, y=heading_left[1]+snake_head.y)
        loc_right = Coordinate(x=heading_right[0]+snake_head.x, y=heading_right[1]+snake_head.y)
        loc_straight = Coordinate(x=heading_straight[0]+snake_head.x, y=heading_straight[1]+snake_head.y)

        left_blocked = snake.landscape.contains_coordinates(loc_left)
        if left_blocked:
            left_blocked = left_blocked or snake.landscape.world[loc_left] == LANDSCAPE_OBJECTS["snake"]
        left_blocked = float(not left_blocked)

        right_blocked = snake.landscape.contains_coordinates(loc_right)
        if right_blocked:
            right_blocked = right_blocked or snake.landscape.world[loc_right] == LANDSCAPE_OBJECTS["snake"]
        right_blocked = float(not right_blocked)

        straight_blocked = snake.landscape.contains_coordinates(loc_straight)
        if straight_blocked:
            straight_blocked = straight_blocked or snake.landscape.world[loc_straight] == LANDSCAPE_OBJECTS["snake"]
        straight_blocked = float(not straight_blocked)

        snake_direction_x = float(snake.heading.x)
        snake_direction_y = float(snake.heading.y)

        apple_loc = snake.landscape.apple_location

        apple_direction_vector = apple_loc - snake_head

        apple_direction_normalized = apple_direction_vector / apple_direction_vector.l2_norm
        apple_direction_x = apple_direction_normalized.x
        apple_direction_y = apple_direction_normalized.y

        return np.array(
            [
                left_blocked,
                right_blocked,
                straight_blocked,
                snake_direction_x,
                snake_direction_y,
                apple_direction_x,
                apple_direction_y
            ]
        )
