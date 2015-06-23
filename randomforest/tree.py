import numpy as np
import string

# Format for saving tree:
# for an internal node:
# left right test
# for a leaf
# -1 -1 class

from weakLearner import WeakLearner, AxisAligned
    
class Tree:
    def __init__( self,
                  params={ 'max_depth' : 10,
                           'min_sample_count' : 5,
                           'test_count' : 100,
                           'test_class' : AxisAligned() } ):
        self.params = params
        self.labels = None
        self.leaf = None
        self.left = None
        self.right = None
        self.test = None

    def __len__(self):
        if self.leaf is not None:
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def write_node(self,line_index, file_buffer,ref_next_available_id):
        if self.leaf is not None:
            file_buffer[line_index] = '-1\t-1\t'+str(self.leaf)
            return
        
        else:
            left_index = ref_next_available_id[0]
            right_index = ref_next_available_id[0] +1
            ref_next_available_id[0] += 2
            
            file_buffer[line_index] += ( str(left_index)
                                         +'\t'
                                         + str(right_index)
                                         + '\t'
                                         +'\t'.join( map(str,self.test) )
                                         )
            
            self.left.write_node(left_index, file_buffer, ref_next_available_id)
            self.right.write_node(right_index, file_buffer, ref_next_available_id)
            return
        
    def save(self,filename):
        file_buffer = [ '' for i in range(2+len(self))]

        # save params
        keys = self.params.keys()
        for i in range(len(keys)):
            file_buffer[0] += keys[i] +'\t'+str(self.params[keys[i]])
            if i < len(keys)-1:
                file_buffer[0] += '\t'

        # save labels
        for i in range(len(self.labels)):
            file_buffer[1] += str(self.labels[i])
            if i < len(self.labels)-1:
                file_buffer[1] += '\t'

        # save nodes
        line_index = 2
        ref_next_available_id = [line_index+1]
        self.write_node( line_index, file_buffer,ref_next_available_id)
        f = open(filename, 'wb')
        for line in file_buffer:
            f.write( line + '\n' )
        f.close()
        return

    def load_node(self,id,lines):
        line = lines[id].split('\t')
        line = map( float, line )
        if line[0] == -1:
            self.leaf = line[2]
            return
        else:
            self.left= Tree(self.params)
            self.right=Tree(self.params)
            self.test = line[2:]
            self.left.load_node(int(line[0]),lines)
            self.right.load_node(int(line[1]),lines)
            return

    def load(self,filename, test=WeakLearner()):
        f = open(filename, 'rb')

        lines = f.readlines()
        lines = map(string.rstrip,lines)
        
        # read params
        params = lines[0].split('\t')
        #self.params = dict( zip( params[::2], params[1::2]) )
        self.params['max_depth'] = int(params[0])
        self.params['min_sample_count'] = int(params[1])
        self.params['test_count'] = int(params[2])
        self.params['test_class'] = params[3]

        assert self.params['test_class'] == str(test), "expected %s, got %s" % (
            self.params['test_class'],
            str(test) )

        self.params['test_class'] = test

        # read labels
        self.labels = map( float, lines[1].split('\t') )
        
        self.load_node(0,lines[2:])
        
        return

    def entropy(self,d):
        E = 0.0
        for r in d.keys():
            if d['count'] > 0 and d[r] > 0:
                proba = float(d[r])/d['count']
                E -= proba*np.log(proba)
        return E 

    def split_points( self,points, responses, test ):
        left = {'count':0.0}
        right = {'count':0.0}
        for c in self.labels:
            right[c] = 0.0
            left[c] = 0.0
        for p,r in zip(points, responses): 
            if self.params['test_class'].run(p, test):
                    right[r] +=1
                    right['count'] +=1
            else:
                left[r] += 1
                left['count'] +=1
        return left, right
        
    def make_leaf(self, all_points):
        response = None
        max_count = -1
        for c in self.labels:
            if all_points[c] > max_count:
                response = c
                max_count = all_points[c]
        self.leaf = response
        return
    
    def grow( self,points, random_state, responses, labels=None, depth=0 ):
        if labels is None:
            self.labels = []
            for r in responses:
                if r not in self.labels:
                    self.labels.append(r)
        else:
            self.labels = labels
        
        print "Number of points:", len(points)
        
        all_points = {'count':len(points)}
        for c in self.labels:
            all_points[c] = 0.0
        for p,r in zip(points, responses): 
            all_points[r] += 1

        H = self.entropy( all_points )
        print "Current entropy:", H            
        
        if ( depth == self.params['max_depth']
             or len(points) <= self.params['min_sample_count']
             or H == 0 ):
            self.make_leaf( all_points )
            return self
        
        all_tests = self.params['test_class'].generate_all( points, random_state, self.params['test_count'] )

        H = self.entropy( all_points )
        print "Current entropy:", H
        best_gain = 0
        best_i = None
        for i, test in enumerate(all_tests):
            left,right = self.split_points( points, responses, test )
            I = H - ( left['count']/all_points['count']*self.entropy(left)
                      + right['count']/all_points['count']*self.entropy(right) )
            if I > best_gain:
                best_gain = I
                best_i = i

        print "Information gain:", best_gain

        if best_i is None:
            print "no best split found: creating a leaf"
            self.make_leaf( all_points )
            return self

        self.test = all_tests[best_i]
        print "TEST:", self.test
        left_points = []
        left_responses = []
        right_points = []
        right_responses = []
        for p,r in zip(points, responses):
            if self.params['test_class'].run(p, self.test):
                right_points.append(p)
                right_responses.append(r)
            else:
                left_points.append(p)
                left_responses.append(r)
        self.left= Tree(self.params)
        self.right=Tree(self.params)

        self.left.grow( np.array(left_points), random_state, left_responses, self.labels, depth+1)
        self.right.grow( np.array(right_points), random_state, right_responses, self.labels, depth+1)

        return self

    def predict(self,point):
        if self.leaf is not None:
            return self.leaf
        else:
            if self.params['test_class'].run(point, self.test):
                return self.right.predict(point)
            else:
                return self.left.predict(point)
        

