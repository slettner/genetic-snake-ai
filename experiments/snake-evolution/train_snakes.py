# Train Snakes with genetic algorithm

import os
import gin
import sys
import time
import logging
import genetic_algorithm
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_snake.snake.snake_fitness import SnakeFitness
from genetic_snake.snake import snake_actions, snake_sensor, snake_brain
from genetic_snake.snake.snake import Snake

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


@gin.configurable
def snake_saver_hook(ga, population, generation):
    """ hook for saving the best snake every 10th iteration"""
    if gin.query_parameter("%snake_dir") is not None:
        if not generation % 3:
            save_path = os.path.join(gin.query_parameter("%snake_dir"), "snake{}".format(generation))
            print("Saving Generation {} in {}".format(generation, save_path))
            best_snake_genetics = population.get_fittest_member().genetic_string
            snake = Snake()  # parameter are set by gin
            snake.policy.set_from_list(best_snake_genetics)
            snake.policy.save(name=os.path.join(gin.query_parameter("%snake_dir"), "snake{}".format(generation)))


def main():

    logger = logging.getLogger("SnakeGame")
    logger_ga = logging.getLogger("GeneticAlgorithm")
    # configure the command line streamer
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger_ga.addHandler(handler)

    gin.parse_config_file(os.path.join(FILE_PATH, "evolution.gin"))
    algorithm = GeneticAlgorithm()
    start = time.time()
    algorithm.train(hooks=[snake_saver_hook])
    print("Train Time: {}".format(time.time()-start))
    # weird files created by zmq
    try:
        os.remove("frontend")
        os.remove("backend")
    except FileNotFoundError:
        pass

    if isinstance(algorithm.fitness_strategy, genetic_algorithm.fitness.MultiProcessingFitnessDecorator):
        algorithm.fitness_strategy.terminate()


if __name__ == '__main__':
    main()

