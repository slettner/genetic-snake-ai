# Utilities


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
        self.x += other.x
        self.y += other.y
        return self
