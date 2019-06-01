# Implements the snake actions

import abc
from ..util import Coordinate
from .snake_environment import LANDSCAPE_OBJECTS


class AbstractSnakeAction(abc.ABC):
    """ Interface for snake actions"""

    @abc.abstractmethod
    def execute(self, snake):
        """ Execute the action. Changes the snakes and env state somehow"""
        raise NotImplementedError


class MoveLeft(AbstractSnakeAction):

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


class MoveRight(AbstractSnakeAction):

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


class MoveUp(AbstractSnakeAction):

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


class MoveDown(AbstractSnakeAction):

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
