import math

eps = 1e-10


def gini_imp(p):
    return 1 - (p**2 + (1 - p) ** 2)


def entropy(p):
    return -(p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))


class Node:
    def __init__(self, dx=None, sx=None, dec_fn=None):
        self.dx = dx
        self.sx = sx
        self.dec_fn = dec_fn

        self.is_leaf = self._update_leaf_status()  # creates / updates the leaf status
        self.target = None

    def _update_leaf_status(self):
        return self.dx is None and self.sx is None

    def add_dx(self, node_dx):
        self.dx = node_dx
        self.is_leaf = self._update_leaf_status()

    def add_sx(self, node_dx):
        self.sx = node_dx
        self.is_leaf = self._update_leaf_status()

    def expand_node(self, dec_fn):
        self.dec_fn = dec_fn

        self.add_sx(Node())
        self.add_dx(Node())


class BinaryTreePredictor:
    def __init__(self, dec_fn=None, split_fn=None, stop_fn=None):
        self.dec_fn = dec_fn
        self.split_fn = split_fn
        self.stop_fn = stop_fn

    def fit(self, X, y):
        pass


def in_order(node):
    it = node
    if it is None:
        return

    in_order(it.sx)
    print(it.target)
    in_order(it.dx)


if __name__ == "__main__":
    # print(gini_imp(0.5))

    print(entropy(0))
