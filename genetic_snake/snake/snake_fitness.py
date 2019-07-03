# Implements fitness function for snakes to train with GA

import gin
import zmq
import multiprocessing as mp
import json
from genetic_algorithm.fitness import AbstractFitnessStrategy
from .snake_game import SnakeGame
from .snake import Snake


@gin.configurable
class SnakeFitness(AbstractFitnessStrategy):

    """ The snake fitness equals the score it can achieve in a game """

    def __init__(self, population_size, board_size, seed, max_steps_per_apple):
        """
        Construct Fitness Function.

        Args:
            population_size(int): The number of snakes to hold
            board_size(tuple): Board of the snake (x, y)
            seed(int): Seed for the game
            max_steps_per_apple(int): Steps the snake has to get the apple
        """
        self.population_size = population_size
        self.board_size = board_size
        self.seed = seed
        self.max_steps_per_apple = max_steps_per_apple

    def fitness(self, chromosomes):
        """
        Calculate fitness of chromosomes
        Args:
            chromosomes:

        Returns:

        """

        for chromosome in chromosomes:
            snake = Snake()
            snake_game = SnakeGame(
                snake=snake,
                seed=self.seed,
                max_steps_per_apple=self.max_steps_per_apple,
            )
            snake_game.snake.policy.set_from_list(chromosome.genetic_string)
            chromosome.fitness = snake_game.play() + 0.001 * snake_game.num_steps
