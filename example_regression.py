#!/usr/bin/python

import numpy as np
import cv2
import itertools

from randomforest.regression_tree import RegressionTree
from randomforest.regression_forest import RegressionForest
from randomforest import weakLearner


def img_test(tree, points, colors, filename, size=512, radius=3):
    img = np.zeros((size, size, 3), dtype='float')
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min) / img.shape[0]
    grid = np.arange(v_min, v_max, step)

    xy = np.array(list(itertools.product(grid, grid)))

    for x in grid:
        for y in grid:
            prediction = np.array([x, y]) + tree.predict([x, y])
            x0, y0 = np.round((prediction - v_min) / step).astype('int32')
            if 0 <= x0 < size and 0 <= y0 < size:
                img[y0, x0, :] += 1

    img *= 255 / img.max()

    points = ((points - v_min) / step).astype('int')
    for p, r in zip(points, responses):
        cv2.circle(img, tuple(p), radius + 1, (0, 0, 0), thickness=-1)
        cv2.circle(img, tuple(p), radius, (0, 255, 0), thickness=-1)

    cv2.imwrite(filename, img.astype('uint8'))


t = np.linspace(0, 2 * np.pi, num=50)

radius = [30, 60]
colors = np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255]], dtype='float32')

points = np.zeros((len(t) * len(radius), 2))
for r in range(len(radius)):
    points[r * len(t):(r + 1) * len(t), 0] = radius[r] * np.cos(t)  # x
    points[r * len(t):(r + 1) * len(t), 1] = radius[r] * np.sin(t)  # y
center = points.mean(axis=0) + 45 * np.ones((2)) / np.sqrt(2)
responses = center[np.newaxis, ...] - points

for learner in weakLearner.__all__:
    print(learner)
    params = {'max_depth': 10,
              'min_sample_count': 5,
              'test_count': 100,
              'test_class': getattr(weakLearner, learner)()}
    tree = RegressionTree(params)
    tree.fit(points, responses)

    # save tree to a text file
    tree.save('tree.txt')
    tree = RegressionTree()
    tree.load('tree.txt', test=params['test_class'])

    img_test(tree, points, colors,
             'img/regression_tree_' + str(learner) + '.png')

    forest = RegressionForest(10, params)
    forest.fit(points, responses)

    # save forest to a directory of text files
    forest.save('saved_model')
    forest = RegressionForest()
    forest.load('saved_model', test=params['test_class'])

    img_test(forest, points, colors,
             'img/regression_forest_' + str(learner) + '.png')
