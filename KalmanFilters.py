import numpy as np

import theano
import theano.tensor as T
import theano.gradient as G

import lasagne
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, \
                           RecurrentLayer, LSTMLayer, \
                           get_output, get_all_params, \
                           get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify, linear, tanh
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_network_params, l2 as L2
from lasagne.updates import adam

from sklearn.model_selection import train_test_split


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
        self.P__ = T.dot(T.identity_like(self.P) - T.dot(self.K, self.hX_), self.P_)


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


class AutoRegressiveModel:
    def __init__(self,
                 steps        = 1,
                 num_layers   = 2,
                 num_units    = 32,
                 eps          = 1e-2,
                 recurrent    = False,
                 nonlinearity = tanh,
                 ):
        self.steps = steps

        self.X = T.fmatrix()
        self.Y = T.fmatrix()

        def network(l):
            if recurrent:
                l = ReshapeLayer(l,
                                 shape = (-1, steps, 1))
                l = LSTMLayer(l, num_units)

            for k in range(num_layers):
                l = DenseLayer(l,
                               num_units    = num_units,
                               nonlinearity = nonlinearity)
            l = DenseLayer(l,
                           num_units    = 1,
                           nonlinearity = linear)

            return l

        self.network = network

        l = InputLayer(input_var = self.X,
                       shape     = (None, steps))
        l = self.network(l)

        self.l_ = l
        self.x_ = get_output(self.l_)

        self.f  = theano.function([self.X],
                                  self.x_,
                                  allow_input_downcast=True)

        l2_penalty = regularize_network_params(l,L2)
        error = squared_error(self.x_, self.Y).mean()
        loss = error + eps * l2_penalty
        params = get_all_params(l)
        updates = adam(loss,
                       params)

        self.error = theano.function([self.X,self.Y],
                                     error,
                                     allow_input_downcast=True)

        self.train = theano.function([self.X,self.Y],
                                     loss,
                                     updates=updates,
                                     allow_input_downcast=True)


    def fit(self,
            time_series,
            test_size = 0.4):
        X = np.array([time_series[i:i-self.steps] for i in range(self.steps)]).T
        Y = np.array([time_series[i+self.steps:] for i in range(1)]).T

        trX, teX, trY, teY = train_test_split(X, Y, test_size=test_size)

        for k in range(1000):
            self.train(trX,trY)
            train_error = self.error(trX,trY)
            test_error  = self.error(teX,teY)

            print k, train_error, test_error



class AutoRegressiveExtendedKalmanFilter:
    def __init__(self,
                 steps      = 1,
                 num_layers = 2,
                 num_units  = 32,
                 eps        = 1e-2):

        self.X, self.Z         = T.fvectors('X','Z')
        self.P, self.Q, self.R = T.fmatrices('P','Q','R')
        self.dt                = T.scalar('dt')

        self.matrix_inv = T.nlinalg.MatrixInverse()

        self.ar = AutoRegressiveModel(steps      = steps,
                                      num_layers = num_layers,
                                      num_units  = num_units,
                                      eps        = eps)

        l = InputLayer(input_var = self.X,
                       shape     = (steps,))
        l = ReshapeLayer(l, shape = (1,steps,))
        l = self.ar.network(l)
        l = ReshapeLayer(l, shape=(1,))

        self.l_ = l
        self.f_ = get_output(self.l_)

        self.X_  = T.concatenate([self.f_, T.dot(T.eye(steps)[:-1], self.X)], axis=0)
        self.fX_ = G.jacobian(self.X_.flatten(), self.X)
        self.P_  = T.dot(T.dot(self.fX_, self.P), T.transpose(self.fX_)) + \
                    T.dot(T.dot(T.eye(steps)[:,0:1], self.dt * self.Q), T.eye(steps)[0:1,:])

        self.h = T.dot(T.eye(steps)[0:1], self.X_)
        self.y = self.Z - self.h

        self.hX_ = G.jacobian(self.h, self.X_)

        self.S = T.dot(T.dot(self.hX_, self.P_), T.transpose(self.hX_)) + self.R
        self.K = T.dot(T.dot(self.P_, T.transpose(self.hX_)), self.matrix_inv(self.S))

        self.X__ = self.X_ + T.dot(self.K, self.y)
        self.P__ = T.dot(T.identity_like(self.P) - T.dot(self.K, self.hX_), self.P_)


        self.prediction = theano.function(inputs  = [self.X,
                                                     self.P,
                                                     self.Q,
                                                     self.dt],
                                          outputs = [self.X_,
                                                     self.P_],
                                          allow_input_downcast = True)

        self.update = theano.function(inputs  = [self.X,
                                                 self.Z,
                                                 self.P,
                                                 self.Q,
                                                 self.R,
                                                 self.dt],
                                      outputs = [self.X__,
                                                 self.P__],
                                      allow_input_downcast = True)


    def fit(self,
            time_series,
            test_size = 0.4):

        self.ar.fit(time_series,
                    test_size = test_size)
        set_all_param_values(self.l_, get_all_param_values(self.ar.l_))



    def __call__(self,
                 X, P, Z, Q, R, dt):
        if np.all(np.isfinite(np.array(Z, dtype=np.float32))):
            x_, p_ = self.update(X,
                                 Z,
                                 P,
                                 Q,
                                 R,
                                 dt)
        else:
            x_, p_ = self.prediction(X,
                                     P,
                                     Q,
                                     dt)

        return x_, p_



class AutoRegressiveUnscentedKalmanFilter:
    def __init__(self,
                 steps      = 1,
                 num_layers = 2,
                 num_units  = 32,
                 eps        = 1e-2,
                 alpha      = 1e-3,
                 beta       = 1.0,
                 kappa      = 0.1):

        lam = alpha * alpha * (steps + kappa) - steps + beta

        self.X, self.Z         = T.fvectors('X','Z')
        self.P, self.Q, self.R = T.fmatrices('P','Q','R')
        self.dt                = T.scalar('dt')

        sqrtm = MatrixSqrt()
        self.matrix_inv = T.nlinalg.MatrixInverse()

        self.ar = AutoRegressiveModel(steps      = steps,
                                      num_layers = num_layers,
                                      num_units  = num_units,
                                      eps        = eps)

        def weighted_mean(A,w):
            mu = T.zeros((steps, 1))
            for i in range(2 * steps + 1):
                mu += w[i] * A[:,i:i+1]
            return mu

        def weighted_covariance(A,B,a,b,w):
            sigma = T.zeros((steps,steps))
            for i in range(2 * steps + 1):
                sigma += w[i] * T.dot((A[:,i:i+1] - a), (B[:,i:i+1] - b).T)
            return sigma


        self.sqrtP = sqrtm(self.P)
        self.XB = T.dot(T.stack(self.X).T, T.ones((1, 2 * steps +1))) + T.concatenate([T.zeros((steps,1)),
                                                                                       T.sqrt(steps + lam) * self.sqrtP,
                                                                                       -T.sqrt(steps + lam) * self.sqrtP], axis=1)

        l = InputLayer(input_var = self.XB.T,
                       shape     = (2 * steps + 1, steps))
        l = self.ar.network(l)
        l = ReshapeLayer(l, shape=(1, 2 * steps + 1))

        self.l_ = l
        self.f_ = get_output(self.l_)

        self.XC = T.concatenate([self.f_, T.dot(T.eye(steps)[:-1], self.XB)], axis=0)

        W_m = T.concatenate([(lam / (steps + lam)) * T.ones(1),
                             (1.0 / (2.0 * (steps + lam))) * T.ones(2 * steps)], axis=0)
        W_c = T.concatenate([(lam / (steps + lam) + (1.0 - alpha * alpha + beta)) * T.ones(1),
                             (1.0 / (2.0 * (steps + lam))) * T.ones(2 * steps)], axis=0)

        self.X_ = weighted_mean(self.XC, W_m)
        self.P_ = weighted_covariance(self.XC, self.XC, self.X_, self.X_, W_c) + \
                                      T.dot(T.dot(T.eye(steps)[:,0:1], self.dt * self.Q), T.eye(steps)[0:1,:])

        self.ZB = T.dot(T.eye(steps)[0:1,:], self.XC)

        self.Z_  = weighted_mean(self.ZB, W_m)
        self.S   = weighted_covariance(self.ZB, self.ZB, self.Z_, self.Z_, W_c) + self.R

        self.K  = T.dot(weighted_covariance(self.XC, self.ZB, self.X_, self.Z_, W_c),
                        self.matrix_inv(self.S))

        self.X__ = self.X_ + T.dot(self.K, self.Z - self.Z_)
        self.P__ = self.P_ - T.dot(T.dot(self.K, self.S), self.K.T)

        # self.test_f = theano.function([self.X, self.P],
        #                               [self.X_, self.K, self.P_],
        #                               allow_input_downcast=True,
        #                               on_unused_input='ignore')

        self.prediction = theano.function(inputs  = [self.X,
                                                     self.P,
                                                     self.Q,
                                                     self.dt],
                                          outputs = [self.X_,
                                                     self.P_],
                                          allow_input_downcast = True)

        self.update = theano.function(inputs  = [self.X,
                                                 self.Z,
                                                 self.P,
                                                 self.Q,
                                                 self.R,
                                                 self.dt],
                                      outputs = [self.X__,
                                                 self.P__],
                                      allow_input_downcast = True)


    def fit(self,
            time_series,
            test_size = 0.4):

        self.ar.fit(time_series,
                    test_size = test_size)
        set_all_param_values(self.l_, get_all_param_values(self.ar.l_))



class MatrixSqrt:
    def __init__(self):
        E = T.nlinalg.Eigh()
        I = T.nlinalg.MatrixInverse()

        def SqrtM(X):
            D, P = E(X)
            D = T.nlinalg.diag(D)
            P_ = I(P)

            return T.dot(P,T.dot(T.sqrt(D),P_))

        self.SqrtM = SqrtM

    def __call__(self, X):
        return self.SqrtM(X)



class sqrtm:
    def __init__(self):
        self.X = T.fmatrix()

        sqrtm = MatrixSqrt()

        self.sqrtm = theano.function([self.X],
                                     sqrtm(self.X),
                                     allow_input_downcast=True)

    def __call__(self, X):
        return self.sqrtm(X)
