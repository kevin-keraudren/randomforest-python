import numpy as np

from .tree import Tree


class ClassificationTree(Tree):
    labels = []

    def entropy(self, d):
        E = 0.0
        for r in d.keys():
            if d['count'] > 0 and d[r] > 0:
                proba = float(d[r]) / d['count']
                E -= proba * np.log(proba)
        return E

    def split_points(self, points, responses, test):
        left = {'count': 0.0}
        right = {'count': 0.0}
        for c in self.labels:
            right[c] = 0.0
            left[c] = 0.0
        for p, r in zip(points, responses):
            if self.params['test_class'].run(p, test):
                right[r] += 1
                right['count'] += 1
            else:
                left[r] += 1
                left['count'] += 1
        return left, right

    def make_leaf(self, all_points):
        response = None
        max_count = -1
        for c in self.labels:
            if all_points[c] > max_count:
                response = c
                max_count = all_points[c]
        self.leaf = response

    def fit(self, points, responses, labels=None, depth=0):
        if labels is None:
            self.labels = []
            for r in responses:
                if r not in self.labels:
                    self.labels.append(r)
        else:
            self.labels = labels

        print("Number of points:", len(points))

        all_points = {'count': len(points)}
        for c in self.labels:
            all_points[c] = 0.0
        for p, r in zip(points, responses):
            all_points[r] += 1

        H = self.entropy(all_points)
        print("Current entropy:", H)

        if (depth == self.params['max_depth']
            or len(points) <= self.params['min_sample_count']
            or H == 0):
            self.make_leaf(all_points)
            return

        all_tests = self.params['test_class'].generate_all(points, self.params[
            'test_count'])

        best_gain = 0
        best_i = None
        for i, test in enumerate(all_tests):
            left, right = self.split_points(points, responses, test)
            I = H - (left['count'] / all_points['count'] * self.entropy(left)
                     + right['count'] / all_points['count'] * self.entropy(
                right))
            if I > best_gain:
                best_gain = I
                best_i = i

        print("Information gain:", best_gain)

        if best_i is None:
            print("no best split found: creating a leaf")
            self.make_leaf(all_points)
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
        self.left = ClassificationTree(self.params)
        self.right = ClassificationTree(self.params)

        self.left.fit(np.array(left_points), left_responses, self.labels,
                      depth + 1)
        self.right.fit(np.array(right_points), right_responses, self.labels,
                       depth + 1)
