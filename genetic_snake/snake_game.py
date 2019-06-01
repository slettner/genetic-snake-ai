# implements the snake game

from pygame.locals import *
import numpy as np
from collections import namedtuple
from random import randint
import pygame
import time


class Coordinate(object):
    """ Represents a x, y position"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        """
        Args:
            other(Coordinate): Other coordinates
        """
        assert isinstance(other, Coordinate), "Equality check not supported between object of type {} and {}".format(
            Coordinate.__class__.__name__,
            other.__class__.__name__
        )


BOARD_SIZE = 100
IMAGE_SIZE = 32
BLANK_ID = 0
SNAKE_ID = 1
APPLE_ID = 2


class GameBoardPiece(object):
    """ Represents a piece of the game board. Can either be blank, the snake or the apple """

    def __init__(self, coords, image_name, piece_id):
        """
        Construct Piece

        Args:
            coords(Coordinate): Position
            image_name(str): Path to the image visualizing the game object
            piece_id(int): ID for the piece. One of BLANK, SNAKE, APPLE
        """
        assert 0 <= coords.x < BOARD_SIZE, "Object placed outside board"
        assert 0 <= coords.y < BOARD_SIZE, "Object placed outside board"
        assert piece_id in [BLANK_ID, SNAKE_ID, APPLE_ID], "Invalid Game Board Piece ID"
        self.pos = coords
        self.image = pygame.image.load(image_name)

    def draw(self, surface):
        """ Draw the image on the game surface"""
        surface.blit(self.image, (self.pos.x, self.pos.y))


class Snake(object):
    """ Represents the Snake """

    SNAKE_IMAGE = "snake.png"

    def __init__(self):
        """
        Construct Snake.

        Args:
        """
        self.body = []  # a list of board pieces
        self.is_alive = True

    def init_snake(self):
        """ create a snake with size three in the upper left of the board"""
        for i in range(3):
            self.body.append(
                GameBoardPiece(
                    Coordinate(x=i, y=0),
                    image_name=Snake.SNAKE_IMAGE,
                    piece_id=SNAKE_ID
                )
            )

    @property
    def size(self):
        return len(self.body)

    def act(self, action):
        """
        Make a step.

        Args:
            action(SnakeAction):

        Returns:

        """


class GameBoard(object):
    """ Represents the board state. (0, 0) is the upper left of the board """
    def __init__(self):
        """
        Construct Board
        """
        self.board = np.repeat(BLANK_ID, BOARD_SIZE*BOARD_SIZE).reshape(BOARD_SIZE, BOARD_SIZE)  # board as 2D array
        self.board_pieces =

    def set_game_piece(self, piece):
        """
        Set the game piece onto the board
        Args:
            piece:

        Returns:

        """

class Game:
    def isCollision(self, x1, y1, x2, y2, bsize):
        if x1 >= x2 and x1 <= x2 + bsize:
            if y1 >= y2 and y1 <= y2 + bsize:
                return True
        return False


class App(object):
    """ Represent the  """

    WINDOW_SIZE = 800

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.game = Game()
        self.player = Player(3)
        self.apple = Apple(5, 5)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('Pygame pythonspot.com example')
        self._running = True
        self._image_surf = pygame.image.load("block.png").convert()
        self._apple_surf = pygame.image.load("block.png").convert()
        return True

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        self.player.update()

        # does snake eat apple?
        for i in range(0, self.player.length):
            if self.game.isCollision(self.apple.x, self.apple.y, self.player.x[i], self.player.y[i], 44):
                self.apple.x = randint(2, 9) * 44
                self.apple.y = randint(2, 9) * 44
                self.player.length = self.player.length + 1

        # does snake collide with itself?
        for i in range(2, self.player.length):
            if self.game.isCollision(self.player.x[0], self.player.y[0], self.player.x[i], self.player.y[i], 40):
                print("You lose! Collision: ")
                print("x[0] (" + str(self.player.x[0]) + "," + str(self.player.y[0]) + ")")
                print("x[" + str(i) + "] (" + str(self.player.x[i]) + "," + str(self.player.y[i]) + ")")
                exit(0)

        pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.player.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if not self.on_init():
            self._running = False

        while self._running:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[K_RIGHT]:
                self.player.moveRight()

            if keys[K_LEFT]:
                self.player.moveLeft()

            if keys[K_UP]:
                self.player.moveUp()

            if keys[K_DOWN]:
                self.player.moveDown()

            if keys[K_ESCAPE]:
                self._running = False

            self.on_loop()
            self.on_render()

            time.sleep(50.0 / 1000.0)
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()