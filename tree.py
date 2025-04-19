import math
import pandas as pd


# ------------------------
# Gini Impurity & Entropy function
# ------------------------

# A small epsilon to avoid log(0)
eps = 1e-10

# Gini Impurity function
def gini_imp(p):
    return 1 - (p**2 + (1 - p) ** 2)

# Entropy function
def entropy(p):
    return -(p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))


# ------------------------
# structure for the nodes
# ------------------------

# Node class
class Node:
    def __init__(self, dx=None, sx=None, dec_fn=None):
        self.dx = dx
        self.sx = sx
        self.dec_fn = dec_fn  

        self.is_leaf = self._update_leaf_status() 
        self.target = None

    def _update_leaf_status(self):
        return self.dx is None and self.sx is None

    def add_dx(self, node_dx):
        self.dx = node_dx
        self.is_leaf = self._update_leaf_status()

    def add_sx(self, node_sx):
        self.sx = node_sx
        self.is_leaf = self._update_leaf_status()

    def expand_node(self, dec_fn):
        self.dec_fn = dec_fn
        self.add_sx(Node())
        self.add_dx(Node())



    def evaluate(self, data_point):
        if self.is_leaf:
            return self.target
        else:
            if self.dec_fn(data_point): 
                return self.sx.evaluate(data_point)
            else:
                return self.dx.evaluate(data_point)
                
# Binary Tree Predictor
class BinaryTreePredictor:
    def __init__(self, dec_fn=None, split_fn=None, stop_fn=None):
        self.dec_fn = dec_fn
        self.split_fn = split_fn
        self.stop_fn = stop_fn

    def fit(self, X, y):
        pass

def in_order(node):
    if node is None:
        return

    in_order(node.sx)
    print(node.target)
    in_order(node.dx)


# Entropy
print(f"Entropy at p=0: {entropy(0)}")
print(f"Entropy at p=0.5: {entropy(0.5)}")
print(f"Entropy at p=1: {entropy(1)}")

# Gini Impurity - p=0.5
print(f"Gini Impurity at p=0.5: {gini_imp(0.5)}")
