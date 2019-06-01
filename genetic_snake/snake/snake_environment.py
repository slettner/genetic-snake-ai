# The environment in which the snake lives

import logging
from genetic_snake.util import Coordinate

LANDSCAPE_OBJECTS = {
    "meadow": 0,
    "snake": 1,
    "apple": 2,
    0: "meadow",  # mostly for easy logs
    1: "snake",  # mostly for easy logs
    2: "apple"  # mostly for easy logs
}


class Landscape(object):
    """ The area in which the snake moves around """
    def __init__(self, size):
        """
        Creates Landscape

        Args:
            size(tuple): Size for x and y
        """
        assert len(size) == 2, "invalid landscape size"
        assert size[0] > 0 and size[1] > 0
        self.size = size

        self.world = self._create_world()
        self.logger = logging.getLogger("LandScape")

    def _create_world(self):
        """ Create the coordinates af the world"""
        points = [(x, y) for x in range(self.size[0]) for y in range(self.size[1])]
        world = dict()
        for x, y in points:
            world[Coordinate(x=x, y=y)] = LANDSCAPE_OBJECTS["meadow"]
        return world

    def plant_apple(self, coordinates):
        """
        Plant an apple at position. Returns True if planting was successful and False if not (i.e. loc was occupied)

        Args:
            coordinates(Coordinate): Position

        Returns:
            bool

        Raises:
            KeyError if the coordinates are not contained within this world
        """
        if self.world[coordinates] != 0:
            self.logger.debug("Trying to plant apple at {}, {} but spot is occupied by {}".format(
                coordinates.x, coordinates.y, LANDSCAPE_OBJECTS[self.world[coordinates]]
            ))
            return False

        else:
            self.world[coordinates] = LANDSCAPE_OBJECTS["apple"]
            return True

    def contains_coordinates(self, coordinates):
        """ Checks if coordinates are contained in the world """
        x_valid = 0 <= coordinates.x < self.size[0]
        y_valid = 0 <= coordinates.y < self.size[1]
        return x_valid and y_valid

