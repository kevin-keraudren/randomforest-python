# Format for saving tree:
# for an internal node:
# left right test
# for a leaf
# -1 -1 class

import numpy as np
from .weakLearner import WeakLearner, AxisAligned


class Tree(object):
    def __init__(self,
                 params={'max_depth': 10,
                         'min_sample_count': 5,
                         'test_count': 100,
                         'test_class': AxisAligned()}):
        self.params = params
        self.leaf = None
        self.left = None
        self.right = None
        self.test = None

    def __len__(self):
        if self.leaf is not None:
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def write_node(self, line_index, file_buffer, ref_next_available_id):
        if self.leaf is not None:
            file_buffer[line_index] = '-1\t-1\t'
            if isinstance(self.leaf, np.ndarray):
                file_buffer[line_index] += '\t'.join(map(str, self.leaf))
            else:
                file_buffer[line_index] += str(self.leaf)
            return

        else:
            left_index = ref_next_available_id[0]
            right_index = ref_next_available_id[0] + 1
            ref_next_available_id[0] += 2

            file_buffer[line_index] += (str(left_index)
                                        + '\t'
                                        + str(right_index)
                                        + '\t'
                                        + '\t'.join(map(str, self.test))
                                        )

            self.left.write_node(left_index, file_buffer, ref_next_available_id)
            self.right.write_node(right_index, file_buffer,
                                  ref_next_available_id)

    def save(self, filename):
        file_buffer = ['' for i in range(2 + len(self))]

        # save params
        keys = list(self.params.keys())
        for i in range(len(keys)):
            file_buffer[0] += keys[i] + '\t' + str(self.params[keys[i]])
            if i < len(keys) - 1:
                file_buffer[0] += '\t'

        # save labels
        if 'labels' in dir(self):
            for i in range(len(self.labels)):
                file_buffer[1] += str(int(self.labels[i]))
                if i < len(self.labels) - 1:
                    file_buffer[1] += '\t'

        # save nodes
        line_index = 2
        ref_next_available_id = [line_index + 1]
        self.write_node(line_index, file_buffer, ref_next_available_id)
        f = open(filename, 'wt')
        for line in file_buffer:
            f.write(line + '\n')
        f.close()

    def load_node(self, id, lines):
        line = lines[id].split('\t')
        line = list(map(float, line))
        if line[0] == -1:
            if 'labels' in dir(self):
                self.leaf = line[2]
            else:
                self.leaf = np.array(line[2:])
            return
        else:
            self.left = Tree(self.params)
            self.right = Tree(self.params)
            self.test = line[2:]
            self.left.load_node(int(line[0]), lines)
            self.right.load_node(int(line[1]), lines)
            return

    def load(self, filename, test=WeakLearner()):
        f = open(filename, 'rt')

        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip(), lines))

        # read params
        params = lines[0].split('\t')
        self.params = dict(zip(params[::2], params[1::2]))
        self.params['max_depth'] = int(self.params['max_depth'])
        self.params['min_sample_count'] = int(self.params['min_sample_count'])
        self.params['test_count'] = int(self.params['test_count'])

        assert self.params['test_class'] == str(test), "expected %s, got %s" % (
            self.params['test_class'],
            str(test))

        self.params['test_class'] = test

        # read labels
        if 'labels' in dir(self):
            self.labels = list(map(int, lines[1].split('\t')))

        self.load_node(2, lines)

    def predict(self, point):
        if self.leaf is not None:
            return self.leaf
        else:
            if self.params['test_class'].run(point, self.test):
                return self.right.predict(point)
            else:
                return self.left.predict(point)
