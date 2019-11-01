# Play a game of snake

import logging
import sys
import os
import genetic_snake
from genetic_snake.snake.snake_game import SnakeGame
from genetic_snake.snake import snake_actions, snake_sensor, snake_brain, snake

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    logger = logging.getLogger("SnakeGame")
    # configure the command line streamer
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    snake = snake.Snake(
        actions=[
            snake_actions.MoveLeft(),
            snake_actions.MoveRight(),
            snake_actions.MoveStraight()
        ],
        sensors=[
            snake_sensor.MinimalSensor()
        ],
        policy=snake_brain.NeuralNetwork(
            action_size=3,
            state_size=7,
            hidden=tuple([15, 15, 15]),
            activations=genetic_snake.nn.relu
        )
    )
    game = SnakeGame(
        snake=snake,
        v=logging.DEBUG,
        render=True
    )
    game.snake.policy.restore(name=os.path.join(FILE_PATH, "..", "sample_snake", "snake-es"))
    score = game.play()
    print("Snake reached score {}".format(score))
