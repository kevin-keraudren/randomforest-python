import numpy as np

from .regression_tree import RegressionTree
from .forest import Forest


class RegressionForest(Forest):
    tree_class = RegressionTree

    def fit(self, points, responses):
        for i in range(self.ntrees):
            self.trees.append(RegressionTree(self.tree_params))
            self.trees[i].fit(points, responses)

    def predict(self, point):
        response = []
        for i in range(self.ntrees):
            response.append(self.trees[i].predict(point))
        return np.mean(response, axis=0)
