from .forest import Forest
from .classification_tree import ClassificationTree


class ClassificationForest(Forest):
    tree_class = ClassificationTree
    labels = []

    def fit(self, points, responses):
        self.labels = []
        for r in responses:
            if r not in self.labels:
                self.labels.append(r)
        for i in range(self.ntrees):
            self.trees.append(ClassificationTree(self.tree_params))
            self.trees[i].fit(points, responses, self.labels)

    def predict(self, point):
        r = {}
        for c in self.labels:
            r[c] = 0.0
        for i in range(self.ntrees):
            response = int(self.trees[i].predict(point))
            r[response] += 1

        response = None
        max_count = -1
        for c in self.labels:
            if r[c] > max_count:
                response = c
                max_count = r[c]
        return response

    def predict_proba(self, point):
        r = {}
        for c in self.labels:
            r[c] = 0.0
        for i in range(self.ntrees):
            response = int(self.trees[i].predict(point))
            r[response] += 1

        for c in self.labels:
            r[c] /= self.ntrees
        return r
