import numpy as np
import os
from glob import glob
import shutil

from .weakLearner import WeakLearner, AxisAligned


class Forest(object):
    def __init__(self,
                 ntrees=20,
                 tree_params={'max_depth': 10,
                              'min_sample_count': 5,
                              'test_count': 100,
                              'test_class': AxisAligned()}):
        self.ntrees = ntrees
        self.tree_params = tree_params
        self.trees = []

    def __len__(self):
        return self.ntrees

    def save(self, folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        template = '%0' + str(int(np.log10(self.ntrees))) + 'd.data'
        for i in range(self.ntrees):
            filename = template % i
            self.trees[i].save(folder + '/' + filename)

    def load(self, folder, test=WeakLearner()):
        self.trees = []
        for f in glob(folder + '/*'):
            self.trees.append(self.tree_class())
            self.trees[-1].load(f, test)
        self.ntrees = len(self.trees)
        if 'labels' in dir(self):
            self.labels = self.trees[0].labels
