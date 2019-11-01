# Train Snakes with genetic algorithm

import os
import gin
from evostra import EvolutionStrategy
from evostra.models import FeedForwardNetwork
from genetic_snake.snake.snake import Snake
from genetic_snake.snake.snake_game import SnakeGame
from genetic_snake.snake.snake_sensor import MinimalSensor
from genetic_snake.snake.snake_actions import MoveLeft, MoveRight, MoveStraight
from genetic_snake.snake.snake_brain import NeuralNetwork


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def fitness(weights):
    """

    Args:
       weights(`list`):

    Returns:

    """
    snake = Snake()
    snake_game = SnakeGame(
        snake=snake,
        seed=0,
        max_steps_per_apple=100
    )
    snake_game.snake.policy.set_from_arrays(weights)
    return snake_game.play() # + 0.001 * snake_game.num_steps


def main():
    gin.parse_config_file(os.path.join(FILE_PATH, "es.gin"))
    model = NeuralNetwork()
    es = EvolutionStrategy(
        model.get_weight_as_arrays(),
        fitness,
        population_size=10,
        sigma=0.2,
        learning_rate=0.03,
        decay=0.9999,
        num_threads=4
    )
    es.run(10, print_step=1)

    model.set_from_arrays(es.get_weights())
    save = gin.query_parameter("%save")
    if save is not None:
        model.save(os.path.join(save, "snake-es"))


if __name__ == '__main__':
    main()