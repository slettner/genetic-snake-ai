# Implements the snake actions

import abc
import gin
import numpy as np
from ..util import Coordinate
from .snake_environment import LANDSCAPE_OBJECTS


class AbstractSnakeAction(abc.ABC):
    """ Interface for snake actions"""

    @abc.abstractmethod
    def execute(self, snake):
        """ Execute the action. Changes the snakes and env state somehow"""
        raise NotImplementedError


@gin.configurable
class MoveWest(AbstractSnakeAction):

    """ Make a movement to the left """

    def execute(self, snake):
        """
        Move the snake to the left.

        Args:
            snake(Snake): The snake
        Returns:

        """
        # update snake
        snake_head = snake.body[-1]
        snake_tail = snake.body.pop(0)  # remove the last body part
        snake.body.append(Coordinate(x=snake_head.x-1, y=snake_head.y))
        new_head = snake.body[-1]
        # if there are not only unique elements the snake bit itself
        snake.is_alive = len(snake.body) == len(set(snake.body))

        # update env
        if snake.landscape.contains_coordinates(new_head):
            if snake.landscape.world[new_head] == LANDSCAPE_OBJECTS["apple"]:
                snake.body.insert(0, snake_tail)  # append body part
            else:
                snake.landscape.world[snake_tail] = LANDSCAPE_OBJECTS["meadow"]
            snake.landscape.world[new_head] = LANDSCAPE_OBJECTS["snake"]
        else:
            snake.is_alive = False


@gin.configurable
class MoveEast(AbstractSnakeAction):

    """ Make a movement to the left """

    def execute(self, snake):
        """
        Move the snake to the left. I.e increase the x coordinate by one

        Args:
            snake(Snake): The snake
        Returns:

        """
        # update snake
        snake_head = snake.body[-1]
        snake_tail = snake.body.pop(0)  # remove the last body part
        snake.body.append(Coordinate(x=snake_head.x+1, y=snake_head.y))
        new_head = snake.body[-1]
        # if there are not only unique elements the snake bit itself
        snake.is_alive = len(snake.body) == len(set(snake.body))

        # update env
        if snake.landscape.contains_coordinates(new_head):
            if snake.landscape.world[new_head] == LANDSCAPE_OBJECTS["apple"]:
                snake.body.insert(0, snake_tail)  # append body part
            else:
                snake.landscape.world[snake_tail] = LANDSCAPE_OBJECTS["meadow"]
            snake.landscape.world[new_head] = LANDSCAPE_OBJECTS["snake"]
        else:
            snake.is_alive = False


@gin.configurable
class MoveNorth(AbstractSnakeAction):

    """ Make a movement to the left """

    def execute(self, snake):
        """
        Move the snake to the up. I.e. decrease y coordinate by one

        Args:
            snake(Snake): The snake
        Returns:

        """
        # update snake
        snake_head = snake.body[-1]
        snake_tail = snake.body.pop(0)  # remove the last body part
        snake.body.append(Coordinate(x=snake_head.x, y=snake_head.y-1))
        new_head = snake.body[-1]
        # if there are not only unique elements the snake bit itself
        snake.is_alive = len(snake.body) == len(set(snake.body))

        # update env
        if snake.landscape.contains_coordinates(new_head):
            if snake.landscape.world[new_head] == LANDSCAPE_OBJECTS["apple"]:
                snake.body.insert(0, snake_tail)  # append body part
            else:
                snake.landscape.world[snake_tail] = LANDSCAPE_OBJECTS["meadow"]
            snake.landscape.world[new_head] = LANDSCAPE_OBJECTS["snake"]
        else:
            snake.is_alive = False


@gin.configurable
class MoveSouth(AbstractSnakeAction):

    """ Make a movement to the left """

    def execute(self, snake):
        """
        Move the snake to the down. I.e increase y coordinate by one

        Args:
            snake(Snake): The snake
        Returns:

        """
        # update snake
        snake_head = snake.body[-1]
        snake_tail = snake.body.pop(0)  # remove the last body part
        snake.body.append(Coordinate(x=snake_head.x, y=snake_head.y+1))
        new_head = snake.body[-1]
        # if there are not only unique elements the snake bit itself
        snake.is_alive = len(snake.body) == len(set(snake.body))

        # update env
        if snake.landscape.contains_coordinates(new_head):
            if snake.landscape.world[new_head] == LANDSCAPE_OBJECTS["apple"]:
                snake.body.insert(0, snake_tail)  # append body part
            else:
                snake.landscape.world[snake_tail] = LANDSCAPE_OBJECTS["meadow"]
            snake.landscape.world[new_head] = LANDSCAPE_OBJECTS["snake"]
        else:
            snake.is_alive = False


#########################
# HEADING BASED ACTIONS #
#########################
@gin.configurable
class RotationAction(AbstractSnakeAction):

    """ This actions lets the snake turn based on the rotation and the snakes current heading """

    def __init__(self, rotation_matrix):
        """

        Args:
            rotation_matrix:
        """
        self.rotation_mat = rotation_matrix

    def execute(self, snake):
        """
        Performs the action. Changes the snake state accordingly

        Args:
            snake(Snake): The snake

        Returns:

        """
        heading = np.array([snake.heading.x, snake.heading.y])
        new_heading = np.matmul(self.rotation_mat, heading)
        new_heading = Coordinate(x=new_heading[0], y=new_heading[1])
        snake.heading = new_heading
        snake_tail = snake.body.pop(0)
        snake_head = snake.head
        new_head = snake_head + new_heading
        snake.body.append(new_head)

        # check if snake bit itself
        snake.is_alive = len(snake.body) == len(set(snake.body))

        # update env
        if snake.landscape.contains_coordinates(new_head):
            if snake.landscape.world[new_head] == LANDSCAPE_OBJECTS["apple"]:
                snake.body.insert(0, snake_tail)  # append body part
            else:
                snake.landscape.world[snake_tail] = LANDSCAPE_OBJECTS["meadow"]
            snake.landscape.world[new_head] = LANDSCAPE_OBJECTS["snake"]
        else:
            snake.is_alive = False


@gin.configurable
class MoveLeft(RotationAction):

    """ This action makes the snake move to the left based in its current heading """

    def __init__(self):
        super(MoveLeft, self).__init__(rotation_matrix=np.array([[0, 1], [-1, 0]]))


@gin.configurable
class MoveRight(RotationAction):

    """ This action makes the snake move to the right based in its current heading """

    def __init__(self):
        super(MoveRight, self).__init__(rotation_matrix=np.array([[0, -1], [1, 0]]))


@gin.configurable
class MoveStraight(RotationAction):

    """ This action makes the snake continue in the direction of its current heading """

    def __init__(self):
        super(MoveStraight, self).__init__(rotation_matrix=np.array([[1, 0], [0, 1]]))  # identity mat
