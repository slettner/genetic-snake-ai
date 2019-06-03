# Play a game of snake

import logging
import sys
from genetic_snake.snake.snake_game import SnakeGame
from genetic_snake.snake import snake_actions, snake_sensor, snake_brain, snake


if __name__ == "__main__":

    logger = logging.getLogger("SnakeGame")
    # configure the command line streamer
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    snake = snake.Snake(
        actions=[
            snake_actions.MoveLeft(),
            snake_actions.MoveRight(),
            snake_actions.MoveUp(),
            snake_actions.MoveDown()
        ],
        sensors=[
            snake_sensor.DistanceSensor()
        ],
        policy=snake_brain.BinaryNeuralNetwork(action_size=4, state_size=24, hidden=tuple([16]))
    )
    game = SnakeGame(
        snake=snake,
        v=logging.DEBUG,
        render=True
    )
    game.snake.policy.restore(name="snake-evolution/snake80")
    score = game.play()
    print("Snake reached score {}".format(score))
