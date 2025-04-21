# ---------------------------------
# Encode features & labels
# ---------------------------------
from sklearn.preprocessing import LabelEncoder

X_encoded = dataset_dict["X"].apply(LabelEncoder().fit_transform)
y_raw = dataset_dict["y"].values.ravel()
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# Check if it's binary
print("Encoded target classes:", np.unique(y_encoded))

# ------------------------
# Structure for the nodes
# ------------------------

import math
import pandas as pd

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

# ------------------------
# Decision Tree node
# ------------------------              

class TreeNode:
    def __init__(self, depth=0):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None
        self.depth = depth

# ------------------------
# Gini Impurity & Entropy function & Missclassification
# ------------------------

# A small epsilon to avoid log(0)
eps = 1e-10

def gini(y):
    probs = np.bincount(y) / len(y)
    return 1 - np.sum(probs ** 2)

def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log(p + 1e-10) for p in probs if p > 0])

def misclassification(y):
    if len(y) == 0:
        return 0
    most_common = Counter(y).most_common(1)[0][1]
    return 1 - (most_common / len(y))

# ---------------------------------
# Tree structure
# ---------------------------------

class TreeNode:
    def __init__(self, depth=0):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None
        self.depth = depth

class BinaryTreePredictor:
    def __init__(self, criterion="gini", max_depth=5, min_samples_split=2):
        if criterion == "gini":
            self.criterion = gini
        elif criterion == "entropy":
            self.criterion = entropy
        elif criterion == "misclassification":
            self.criterion = misclassification
        else:
            raise ValueError("Unsupported criterion.")
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _best_split(self, X, y):
        best_feature, best_thresh, best_gain = None, None, -float('inf')
        current_impurity = self.criterion(y)
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                weighted_impurity = (
                    len(y_left)/len(y)*self.criterion(y_left) +
                    len(y_right)/len(y)*self.criterion(y_right)
                )
                gain = current_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_thresh = t

        return best_feature, best_thresh

    def _build_tree(self, X, y, depth=0):
        node = TreeNode(depth=depth)
        if (depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split):
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        node.feature_index = feature
        node.threshold = threshold
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        node.left = self._build_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth+1)
        return node

    def fit(self, X, y):
        self.root = self._build_tree(np.array(X), np.array(y))

    def _predict_one(self, x, node):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.array(X)])

# ---------------------------------
# Fit & predict
# ---------------------------------

from collections import Counter

tree = BinaryTreePredictor(criterion="misclassification", max_depth=6)
tree.fit(X_encoded, y_encoded)

# Predict first 10 samples
preds = tree.predict(X_encoded[:10])
print("\nPredictions (first 10):", preds)

# ---------------------------------
# in_order traversal
# ---------------------------------
def in_order(node):
    if node is None:
        return
    in_order(node.left)
    if node.prediction is not None:
        print(f"Leaf → Class: {node.prediction}")
    in_order(node.right)

print("\nTree traversal (in-order):")
in_order(tree.root)

# ---------------------------------
# Training Error (0-1 Loss)
# ---------------------------------

def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

train_loss = zero_one_loss(y_encoded, tree.predict(X_encoded))
print("\nTraining Error (0-1 Loss):", train_loss)

def print_tree(node, depth=0):
    prefix = "  " * depth
    if node.prediction is not None:
        print(f"{prefix}Predict: {node.prediction}")
    else:
        print(f"{prefix}X[{node.feature_index}] <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

print_tree(tree.root)

# ---------------------------------
# Accuracy
# ---------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
