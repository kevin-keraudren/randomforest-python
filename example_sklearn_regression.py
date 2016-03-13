#!/usr/bin/python

import numpy as np
import cv2  # OpenCV
import itertools

from randomforest import weakLearner
from randomforest.weakLearner import FeatureExtractor

from sklearn import ensemble


def img_test(tree, feature_extractor, points, colors, filename, size=512,
             radius=3):
    img = np.zeros((size, size, 3), dtype='float')
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min) / img.shape[0]
    grid = np.arange(v_min, v_max, step)

    xy = np.array(list(itertools.product(grid, grid)))
    features = feature_extractor.apply_all(xy)

    predictions = xy + tree.predict(features)
    predictions = np.round((predictions - v_min) / step).astype('int32')

    flat_indices = np.ravel_multi_index(np.transpose(predictions),
                                        img.shape[:2], mode='clip')
    bins = np.bincount(flat_indices, minlength=np.prod(img.shape[:2]))
    img += bins.reshape(img.shape[:2])[..., np.newaxis]

    # artefacts of clipping
    img[0] = 0
    img[-1] = 0
    img[:, 0] = 0
    img[:, -1] = 0

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
    test_class = getattr(weakLearner, learner)()
    params = {'max_depth': None,
              'min_samples_split': 2,
              'n_jobs': 1,
              'n_estimators': 100}

    print(str(learner))

    forest = ensemble.RandomForestRegressor(**params)
    feature_extractor = FeatureExtractor(test_class, n_features=1000)
    features = feature_extractor.fit_transform(points)
    forest.fit(features, responses)

    img_test(forest, feature_extractor, points, colors,
             'img/regression_forest_sklearn_' + str(learner) + '.png')
