#!/usr/bin/python

import time
start_time = time.time()

import numpy as np
from PIL import Image

from randomforest import *
from randomforest import weakLearner

def img_test( tree, points, colors, filename, size=512, radius=3):
    img = np.zeros((size,size,3), dtype='uint8')
    v_min = points.min()
    v_max = points.max()
    step = float(v_max - v_min)/img.shape[0]
    grid = np.arange( v_min, v_max, step )

    for x in grid:
        for y in grid:
            label = int( tree.predict([x,y]) )
            img[int((y-v_min)/step),
                int((x-v_min)/step),:] = colors[label]

    #img = Image.fromarray(img)
    img = Image.fromstring('RGB',img.shape[:2], img.tostring())
    img.save(filename)

        
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
    params={ 'max_depth' : 10,
             'min_sample_count' : 5,
             'test_count' : 100,
             'test_class' : getattr( weakLearner, learner)() }
    
    tree = Tree( params )
    tree.grow( points, np.random.RandomState(0), responses )

    for i in range(len(points)):
        print responses[i], tree.predict(points[i])

    img_test( tree, points, colors, 'tree_'+str(learner)+'.png' )

print("--- %s seconds ---" % (time.time() - start_time))
            
