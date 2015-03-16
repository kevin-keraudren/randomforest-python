#!/usr/bin/python

import numpy as np
import cv2 # OpenCV

from randomforest import *
from randomforest import weakLearner

from sklearn import ensemble

class FeatureExtractor(object):
    def __init__(self,learner,n_features):
        self.learner = learner
        self.n_features = n_features
        self.tests = []

    def fit_transform(self,points):
        self.tests = self.learner.generate_all( points, self.n_features )
        return self.apply_all(points)

    def apply(self,point):
        return np.array( map( lambda t: self.learner.run(point,t), self.tests ) )

    def apply_all(self,points):
        return map(self.apply,points)

    
def img_test(forest, feature_extractor, points, colors, filename, size=512, radius=3, soft=True):
    img = np.zeros((512,512,3))
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min)/img.shape[0]
    grid = np.arange( v_min, v_max, step )

    for x in grid:
        for y in grid:
            if soft:
                r = forest.predict_proba(feature_extractor.apply([x,y])).flatten()
                col = np.zeros(3,dtype=float)
                for c in range(forest.n_classes_):
                    col += r[int(c)]*np.array(colors[int(c)])
                    col = tuple(col.astype('int'))
            else:
                r = forest.predict(feature_extractor.apply([x,y]))
                col = colors[int(r)]
            img[int((y-v_min)/step),
                int((x-v_min)/step),:] = col

    points = ((points - v_min)/step).astype('int')
    for p,r in zip(points,responses):
        cv2.circle(img, tuple(p), radius+1, (0,0,0), thickness=-1 )
        cv2.circle(img, tuple(p), radius, colors[int(r)], thickness=-1 )

    cv2.imwrite(filename,img)
            

t = np.arange(0,10,0.1)

theta = [0,30,60]
colors = [(255,0,0),
          (0,255,0),
          (0,0,255)]


points = np.zeros((len(t)*len(theta),2))
responses = np.zeros(len(t)*len(theta))
for c in range(len(theta)):
    points[c*len(t):(c+1)*len(t),0] = t**2*np.cos(t+theta[c]) # x
    points[c*len(t):(c+1)*len(t),1] = t**2*np.sin(t+theta[c]) # y
    responses[c*len(t):(c+1)*len(t)] = c

for learner in weakLearner.__all__:
    test_class = getattr( weakLearner, learner)()
    params={ 'max_depth' : None,
             'min_samples_split' : 2,
             'n_estimators' : 100 }
    
    forest = ensemble.RandomForestClassifier( **params )
    print points
    feature_extractor = FeatureExtractor( test_class, n_features=1000 )
    features = feature_extractor.fit_transform(points)
    forest.fit( features, responses )

    for i in range(len(points)):
        print responses[i], forest.predict_proba(features[i])

    img_test( forest, feature_extractor, points, colors, 'forest_sklearn_'+str(learner)+'_soft.png',soft=True)
    img_test( forest, feature_extractor, points, colors, 'forest_sklearn_'+str(learner)+'_hard.png',soft=False)
