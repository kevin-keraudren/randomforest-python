#!/usr/bin/python

import numpy as np
import cv2  # OpenCV
import itertools

from randomforest import weakLearner
from randomforest.weakLearner import FeatureExtractor

from sklearn import ensemble


def img_test(forest, feature_extractor, points, colors, filename, size=512,
             radius=3, proba=True):
    img = np.zeros((size, size, 3))
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min) / img.shape[0]
    grid = np.arange(v_min, v_max, step)

    xy = np.array(list(itertools.product(grid, grid)))
    features = feature_extractor.apply_all(xy)

    if proba:
        r = forest.predict_proba(features)
        col = np.dot(r, colors)
    else:
        r = forest.predict(features).astype('int32')
        col = colors[r]
    img[((xy[:, 1] - v_min) / step).astype('int32'),
        ((xy[:, 0] - v_min) / step).astype('int32')] = col

    points = ((points - v_min) / step).astype('int')
    for p, r in zip(points, responses):
        col = tuple(colors[int(r)])
        cv2.circle(img, tuple(p), radius + 1, (0, 0, 0), thickness=-1)
        cv2.circle(img, tuple(p), radius, col, thickness=-1)

    cv2.imwrite(filename, img)


t = np.arange(0, 10, 0.1)

theta = [0, 30, 60]
colors = np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255]], dtype='float')

points = np.zeros((len(t) * len(theta), 2))
responses = np.zeros(len(t) * len(theta))
for c in range(len(theta)):
    points[c * len(t):(c + 1) * len(t), 0] = t ** 2 * np.cos(t + theta[c])  # x
    points[c * len(t):(c + 1) * len(t), 1] = t ** 2 * np.sin(t + theta[c])  # y
    responses[c * len(t):(c + 1) * len(t)] = c

for learner in weakLearner.__all__:
    test_class = getattr(weakLearner, learner)()
    params = {'max_depth': None,
              'min_samples_split': 2,
              'n_jobs': 1,
              'n_estimators': 100}

    print(str(learner))

    forest = ensemble.RandomForestClassifier(**params)
    feature_extractor = FeatureExtractor(test_class, n_features=1000)
    features = feature_extractor.fit_transform(points)
    forest.fit(features, responses)

    img_test(forest, feature_extractor, points, colors,
             'img/classification_forest_sklearn_' + str(learner) + '_soft.png',
             proba=True)
    img_test(forest, feature_extractor, points, colors,
             'img/classification_forest_sklearn_' + str(learner) + '_hard.png',
             proba=False)
