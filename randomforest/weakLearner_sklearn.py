import numpy as np

__all__ = [ "AxisAligned",
            "Linear",
            "Conic",
            "Parabola" ]

class WeakLearner:
    def generate_all(self, points, random_state, count ):
        return None

    def __str__(self):
        return None

    def run(self, point, test):
        return None

class AxisAligned(WeakLearner):
    """Axis aligned"""
    def __str__(self):
        return "AxisAligned"

    def generate_all(self, points, random_state, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend( zip(np.zeros(count,dtype=int), random_state.uniform(x_min,x_max,count)))
        tests.extend( zip(np.ones(count,dtype=int), random_state.uniform(y_min,y_max,count)))
        return np.array(tests)

    def run(self, points, tests):
        return np.transpose([points[:,tests[i,0]] > tests[i,1] for i in range(len(tests))])

class Linear(WeakLearner):
    """Linear"""
    def __str__(self):
        return "Linear"
    
    def generate_all(self, points, random_state, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        tests = []
        tests.extend( zip(random_state.uniform(x_min,x_max,count),
                          random_state.uniform(y_min,y_max,count),
                          random_state.uniform(0,360,count)))
        return np.array(tests)

    def run(self, points, tests):
        return np.transpose([( np.cos(tests[i,2]*np.pi/180)*(points[:,0]-tests[i,0])
                 + np.sin(tests[i,2]*np.pi/180)*(points[:,1]-tests[i,1]) ) > 0
                             for  i in range(len(tests))])

class Conic(WeakLearner):
    """Non-linear: conic"""
    def __str__(self):
        return "Conic"
    
    def generate_all(self, points, random_state, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = max( points.max(),abs(points.min()) )
        tests = []
        tests.extend( zip( random_state.uniform(x_min,x_max,count),
                           random_state.uniform(y_min,y_max,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count),
                           random_state.uniform(-scale,scale,count)*random_state.random_integers(0,1,count)
                           )
                      )
        
        return np.array(tests)

    def _run(self, points, test):
        x = (points[:,0]-test[0])
        y = (points[:,1]-test[1])
        A,B,C,D,E,F = test[2:]
        return ( A*x*x + B*y*y + C*x*x + D*x + E*y + F) > 0

    def run(self, points, tests):
        return np.transpose(map(lambda t : self._run(points,t), tests))

class Parabola(WeakLearner):
    """Non-linear: parabola"""
    def __str__(self):
        return "Parabola"
    
    def generate_all(self, points, random_state, count ):
        x_min = points.min(0)[0]
        y_min = points.min(0)[1]
        x_max = points.max(0)[0]
        y_max = points.max(0)[1]
        scale = abs( points.max()-points.min() )
        tests = []
        tests.extend( zip( random_state.uniform(2*x_min,2*x_max,count),
                           random_state.uniform(2*y_min,2*y_max,count),
                           random_state.uniform(-scale,scale,count),
                           random_state.random_integers(0,1,count)
                           )
                      )
        
        return np.array(tests)

    def _run(self, points, test):
        x = (points[:,0]-test[0])
        y = (points[:,1]-test[1])
        p,axis = test[2:]
        if axis == 0:
            return x*x < p*y
        else:
            return y*y < p*x

    def run(self, points, tests):
        return np.transpose(map(lambda t : self._run(points,t), tests))
