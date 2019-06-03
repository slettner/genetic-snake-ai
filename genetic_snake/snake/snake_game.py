# implements the snake game

from pygame.locals import *
import numpy as np
from collections import namedtuple
from random import randint
import pygame
import time
import gin
import logging
from .snake import Snake
from .snake_environment import AppleGenerator, LANDSCAPE_OBJECTS


@gin.configurable
class SnakeGame(object):

    """ Represents the snake game. Includes planting apple, counting scores and moving the snake """

    def __init__(self,
                 snake,
                 seed=0,
                 v=logging.INFO,
                 max_steps_per_apple=1000,
                 render=False
                 ):
        """
        Construct a snake game.

        Args:
            snake(Snake): The snake
            v(int): Log level
            max_steps_per_apple(int): The max number of steps to find an apple. If apple is found counter gets reset
            render(bool): Display the game
        """
        self.snake = snake
        self.score = 0
        self.num_steps = 0
        self.seed = seed
        self.apple_generator = AppleGenerator(self.snake.landscape.size, seed=np.random.randint(0, 10000))
        self.logger = logging.getLogger("SnakeGame")
        self.logger.setLevel(level=v)
        self._steps_since_last_apple = 0
        self._max_steps_per_apple = max_steps_per_apple
        self._render = render
        if self._render:
            self.display, self.images = self._init_display()

    def play(self):
        """ Play a snake game. """

        self.plant_next_apple()
        current_snake_size = self.snake.size
        self.logger.debug(self.snake.landscape)
        if self._render:
            self.render()
            time.sleep(0.03)
        while self.snake.is_alive and self._steps_since_last_apple < self._max_steps_per_apple:

            self.snake.move()
            if current_snake_size != self.snake.size:  # snake ate apple
                self.plant_next_apple()
                self._steps_since_last_apple = 0
                current_snake_size = self.snake.size
            self.num_steps += 1
            self._steps_since_last_apple += 1
            self.score = len(self.snake.body) - 3  # three is the initial size
            self.logger.debug(self.snake.landscape)
            if self._render:
                self.render()
                time.sleep(0.03)
        self.logger.debug("Snake died after {} steps reaching a score of {}".format(self.num_steps, self.score))
        return self.score

    def plant_next_apple(self):
        """ Plants a new apple """
        successful_plant = self.snake.landscape.plant_apple(self.apple_generator.__next__())
        while not successful_plant:
            successful_plant = self.snake.landscape.plant_apple(self.apple_generator.__next__())

    def render(self):
        self.display.fill((0, 0, 0))
        for loc, code in self.snake.landscape.world.items():
            x = loc.x
            y = loc.y
            self.display.blit(self.images[LANDSCAPE_OBJECTS[code]], (32*x, 32*y))
        pygame.display.flip()

    def _init_display(self):
        pygame.init()
        window_width = 32 * self.snake.landscape.size[0]
        window_height = 32 * self.snake.landscape.size[1]
        display_surf = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE)
        path = "/Users/sebastianlettner/Desktop/genetic-snake-ai/images/"
        snake_surf = pygame.image.load(
            path + "snake.png"
        ).convert()
        apple_surf = pygame.image.load(
            path + "apple.png"
        ).convert()
        blank_surf = pygame.image.load(
            path + "blank.png"
        ).convert()

        images = {
            "meadow": blank_surf,
            "apple": apple_surf,
            "snake": snake_surf
        }

        return display_surf, images




class App:
    windowWidth = 800
    windowHeight = 600
    player = 0
    apple = 0

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.player = Player(10)
        self.apple = Apple(5, 5)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

        self._running = True
        self._image_surf = pygame.image.load("pygame.png").convert()
        self._apple_surf = pygame.image.load("apple.png").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        self.player.update()
        pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()
