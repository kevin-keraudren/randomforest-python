import numpy as np

from .tree import Tree


class RegressionTree(Tree):
    def MSE(self, responses):
        mean = np.mean(responses, axis=0)
        return np.mean((responses - mean) ** 2)

    def split_points(self, points, responses, test):
        left = []
        right = []
        for p, r in zip(points, responses):
            if self.params['test_class'].run(p, test):
                right.append(r)
            else:
                left.append(r)
        return left, right

    def make_leaf(self, responses):
        self.leaf = np.mean(responses, axis=0)

    def fit(self, points, responses, depth=0):

        print("Number of points:", len(points))

        error = self.MSE(responses)
        print("Current MSE:", error)

        if (depth == self.params['max_depth']
            or len(points) <= self.params['min_sample_count']
            or error == 0):
            self.make_leaf(responses)
            return

        all_tests = self.params['test_class'].generate_all(points, self.params[
            'test_count'])

        best_error = np.inf
        best_i = None
        for i, test in enumerate(all_tests):
            left, right = self.split_points(points, responses, test)
            error = (len(left) / len(points) * self.MSE(left)
                     + len(right) / len(points) * self.MSE(right))
            if error < best_error:
                best_error = error
                best_i = i

        print("Best error:", best_error)

        if best_i is None:
            print("no best split found: creating a leaf")
            self.make_leaf(responses)
            return

        self.test = all_tests[best_i]
        print("TEST:", self.test)
        left_points = []
        left_responses = []
        right_points = []
        right_responses = []
        for p, r in zip(points, responses):
            if self.params['test_class'].run(p, self.test):
                right_points.append(p)
                right_responses.append(r)
            else:
                left_points.append(p)
                left_responses.append(r)
        self.left = RegressionTree(self.params)
        self.right = RegressionTree(self.params)

        self.left.fit(np.array(left_points), left_responses, depth + 1)
        self.right.fit(np.array(right_points), right_responses, depth + 1)
