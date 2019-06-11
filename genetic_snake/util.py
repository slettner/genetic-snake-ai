# Utilities

import numpy as np


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
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        assert isinstance(other, Coordinate), "Equality check not supported between object of type {} and {}".format(
            Coordinate.__class__.__name__,
            other.__class__.__name__
        )
        result = Coordinate(x=0, y=0)
        result.x = self.x + other.x
        result.y = self.y + other.y
        return result

    def __sub__(self, other):
        assert isinstance(other, Coordinate), "Equality check not supported between object of type {} and {}".format(
            Coordinate.__class__.__name__,
            other.__class__.__name__
        )
        result = Coordinate(x=0, y=0)
        result.x = self.x - other.x
        result.y = self.y - other.y
        return result

    @property
    def l2_norm(self):
        return np.sqrt(self.x**2 + self.y**2)

    def __truediv__(self, other):
        result = Coordinate(x=0, y=0)
        if isinstance(other, Coordinate):
            result.x = self.x / other.x
            result.y = self.y / other.y
        else:
            result.x = self.x / other
            result.y = self.y / other
        return result
