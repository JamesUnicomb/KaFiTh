import numpy as np

import theano
import theano.tensor as T
import theano.gradient as G


class BrownianFilter:
    def __init__(self,
                 state       = 'x',
                 measurement = 'z',
                 motion_transition      = None,
                 measurement_transition = None):

       self.N = len(state.split(' '))
       self.M = len(measurement.split(' '))


       self.X, self.Z         = T.fvectors('X','Z')
       self.P, self.Q, self.R = T.fmatrices('P','Q','R')
       self.F, self.H         = T.matrices('F','H')
       self.dt                = T.scalar('dt')


       self.X_  = T.dot(self.F, self.X)
       self.fX_ = G.jacobian(T.flatten(self.X_), self.X)
       self.P_  = T.dot(T.dot(self.fX_, self.P), T.transpose(self.fX_)) + self.dt * self.Q

       self.h = T.dot(self.H, self.X_)
       self.y = self.Z - self.h

       self.hX_ = G.jacobian(self.h, self.X_)

       self.matrix_inv = T.nlinalg.MatrixInverse()

       self.S = T.dot(T.dot(self.hX_, self.P_), T.transpose(self.hX_)) + self.R
       self.K = T.dot(T.dot(self.P_, T.transpose(self.hX_)), self.matrix_inv(self.S))

       self.X__ = self.X_ + T.dot(self.K, self.y)
       self.P__ = T.dot(T.identity_like(self.P) - self.K * self.hX_, self.P_)


       self.prediction = theano.function(inputs  = [self.X,
                                                    self.P,
                                                    self.Q,
                                                    self.F,
                                                    self.dt],
                                         outputs = [self.X_,
                                                    self.P_],
                                         allow_input_downcast = True)

       self.update = theano.function(inputs  = [self.X,
                                                self.Z,
                                                self.P,
                                                self.Q,
                                                self.R,
                                                self.F,
                                                self.H,
                                                self.dt],
                                     outputs = [self.X__,
                                                self.P__],
                                     allow_input_downcast = True)

       if motion_transition == None:
           self.motion_transition = np.eye(self.N)
       else:
           self.motion_transition = np.array(motion_transition)

       if measurement_transition == None:
           self.measurement_transition = np.eye(self.M)
       else:
           self.measurement_transition = np.array(motion_transition)


    def __call__(self,
                 X, P, Z, Q, R, dt):
        if np.all(np.isfinite(np.array(Z, dtype=np.float32))):
            x_, p_ = self.update(X,
                                 Z,
                                 P,
                                 Q,
                                 R,
                                 self.motion_transition,
                                 self.measurement_transition,
                                 dt)
        else:
            x_, p_ = self.prediction(X,
                                     P,
                                     Q,
                                     self.motion_transition,
                                     dt)

        return x_, p_
