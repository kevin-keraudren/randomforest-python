import numpy as np

__all__ = [ "AxisAligned",
            "Linear",
            "Conic",
            "Parabola" ]

class WeakLearner:
    def generate_all(self, points, count ):
        return None

    def __str__(self):
        return None

    def run(self, point, test):
        return None

class AxisAligned(WeakLearner):
    """Axis aligned"""
    def __str__(self):
        return "AxisAligned"

    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        print tests
        print  x_min,x_max,count
        tests.extend( zip(np.zeros(count,dtype=int), np.random.uniform(x_min,x_max,count)))
        tests.extend( zip(np.ones(count,dtype=int), np.random.uniform(y_min,y_max,count)))
        return tests

    def run(self, point, test):
        return point[int(test[0])] > test[1]

class Linear(WeakLearner):
    """Linear"""
    def __str__(self):
        return "Linear"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        print tests
        print  x_min,x_max,count
        tests.extend( zip(np.random.uniform(x_min,x_max,count),
                          np.random.uniform(y_min,y_max,count),
                          np.random.uniform(0,360,count)))
        return tests

    def run(self, point, test):
        theta = test[2]*np.pi/180
        return ( np.cos(theta)*(point[0]-test[0])
                 + np.sin(theta)*(point[1]-test[1]) ) > 0

class Conic(WeakLearner):
    """Non-linear: conic"""
    def __str__(self):
        return "Conic"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = max( points.max(),abs(points.min()) )
        tests = []
        tests.extend( zip( np.random.uniform(x_min,x_max,count),
                           np.random.uniform(y_min,y_max,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count),
                           np.random.uniform(-scale,scale,count)*np.random.random_integers(0,1,count)
                           )
                      )
        
        return tests

    def run(self, point, test):
        x = (point[0]-test[0])
        y = (point[1]-test[1])
        A,B,C,D,E,F = test[2:]
        return ( A*x*x + B*y*y + C*x*x + D*x + E*y + F) > 0

class Parabola(WeakLearner):
    """Non-linear: parabola"""
    def __str__(self):
        return "Parabola"
    
    def generate_all(self, points, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = abs( points.max()-points.min() )
        tests = []
        tests.extend( zip( np.random.uniform(2*x_min,2*x_max,count),
                           np.random.uniform(2*y_min,2*y_max,count),
                           np.random.uniform(-scale,scale,count),
                           np.random.random_integers(0,1,count)
                           )
                      )
        
        return tests

    def run(self, point, test):
        x = (point[0]-test[0])
        y = (point[1]-test[1])
        p,axis = test[2:]
        if axis == 0:
            return x*x < p*y
        else:
            return y*y < p*x
