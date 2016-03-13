import numpy as np
from numpy.random import uniform, random_integers

__all__ = ["AxisAligned",
           "Linear",
           "Conic",
           "Parabola"]


class WeakLearner(object):
    def generate_all(self, points, count):
        return None

    def __str__(self):
        return None

    def run(self, point, test):
        return None


class AxisAligned(WeakLearner):
    """Axis aligned"""

    def __str__(self):
        return "AxisAligned"

    def generate_all(self, points, count):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend(zip(np.zeros(count, dtype=int),
                         uniform(x_min, x_max, count)))
        tests.extend(zip(np.ones(count, dtype=int),
                         uniform(y_min, y_max, count)))
        return np.array(tests)

    def run(self, point, test):
        return point[int(test[0])] > test[1]

    def run_all(self, points, test):
        return np.array(list(map(lambda test: points[:, test[0]] > test[1],
                                 test))).T


class Linear(WeakLearner):
    """Linear"""

    def __str__(self):
        return "Linear"

    def generate_all(self, points, count):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend(zip(uniform(x_min, x_max, count),
                         uniform(y_min, y_max, count),
                         uniform(0, 360, count)))
        return tests

    def run(self, point, test):
        theta = test[2] * np.pi / 180
        return (np.cos(theta) * (point[0] - test[0]) +
                np.sin(theta) * (point[1] - test[1])) > 0

    def run_all(self, points, tests):
        def _run(test):
            theta = test[2] * np.pi / 180
            return (np.cos(theta) * (points[:, 0] - test[0]) +
                    np.sin(theta) * (points[:, 1] - test[1])) > 0

        return np.array(list(map(_run, tests))).T


class Conic(WeakLearner):
    """Non-linear: conic"""

    def __str__(self):
        return "Conic"

    def generate_all(self, points, count):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = max(points.max(), abs(points.min()))
        tests = []
        tests.extend(zip(uniform(x_min, x_max, count),
                         uniform(y_min, y_max, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count),
                         uniform(-scale, scale,
                                 count) * random_integers(0, 1, count)))

        return tests

    def run(self, point, test):
        x = (point[0] - test[0])
        y = (point[1] - test[1])
        A, B, C, D, E, F = test[2:]
        return (A * x * x + B * y * y + C * x * x + D * x + E * y + F) > 0

    def run_all(self, points, tests):
        def _run(test):
            x = (points[:, 0] - test[0])
            y = (points[:, 1] - test[1])
            A, B, C, D, E, F = test[2:]
            return (A * x * x + B * y * y + C * x * x + D * x + E * y + F) > 0

        return np.array(list(map(_run, tests))).T


class Parabola(WeakLearner):
    """Non-linear: parabola"""

    def __str__(self):
        return "Parabola"

    def generate_all(self, points, count):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = abs(points.max() - points.min())
        tests = []
        tests.extend(zip(uniform(2 * x_min, 2 * x_max, count),
                         uniform(2 * y_min, 2 * y_max, count),
                         uniform(-scale, scale, count),
                         random_integers(0, 1, count)))

        return tests

    def run(self, point, test):
        x = (point[0] - test[0])
        y = (point[1] - test[1])
        p, axis = test[2:]
        if axis == 0:
            return x * x < p * y
        else:
            return y * y < p * x

    def run_all(self, points, tests):
        def _run(test):
            x = (points[:, 0] - test[0])
            y = (points[:, 1] - test[1])
            p, axis = test[2:]
            if axis == 0:
                return x * x < p * y
            else:
                return y * y < p * x

        return np.array(list(map(_run, tests))).T


class FeatureExtractor(object):
    def __init__(self, learner, n_features):
        self.learner = learner
        self.n_features = n_features
        self.tests = []

    def fit_transform(self, points):
        self.tests = self.learner.generate_all(points, self.n_features)
        return self.apply_all(points)

    def apply(self, point):
        return np.array(list(map(lambda t: self.learner.run(point, t),
                                 self.tests)))

    def apply_all(self, points):
        return self.learner.run_all(points, self.tests)
