from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
import matplotlib.pyplot as plt
import abc
import copy
from warnings import warn

# Added by Enrique Noriega
from scipy.stats import norm, invgamma, gamma

from pyhsmm.basic.abstractions import Distribution, GibbsSampling



class Probit(GibbsSampling, Distribution):
    ''' Probit model for the UA Context model of the REACH team

        The cumulative Gaussian distribution has zero mean and identity covariance matrix

        Author: Enrique Noriega
        Email: enoriega@email.arizona.edu
    '''

    def __init__(self, W, l):
        ''' W is the weight matrix and l is the state vector

            l is a binary vector and W a real matrix
        '''

        self.W = np.matrix(W)
        self.l = np.matrix(l)

        # Enforce l to be a column vector
        if l.shape[1] > 1:
            l = l.T

        assert W.shape[1] == l.shape[0], "The weight matrix and the state vector should be equivalent"
        assert ((l <= 1) & (l >= 0)).all(), "l should be a binary vector"

        # TODO: I think this computation may be wrong. Double check with the team.
        # Compute the weights vector
        self.w = W*l #These should be numpy's matrix objects so this works

    @np.vectorize
    def _threshold(s, p):
        ''' Internal method to check wether an element is above a threshold or not '''
        return 0. if s>= p else 1.

    def rvs(self, size=1):
        ''' Generates a random variate (sample)

            It samples from a bernoulli with parameters as self.w
        '''

        if type(size) in (list, tuple):
            size = size[0] if len(size) > 0 else 1

        variate = np.matrix(np.zeros((size, self.w.shape[0])))

        for j in xrange(self.w.shape[0]):
            variate[:, j] = self._threshold(stats.norm.rvs(size=(size, 1)), self.w[j])

        return variate


    def log_likelihood(self, X):
        ''' Computes the log likelihood according to the following formula:

        p(\mathbf{y} | \mathbf{w}) = \log \left[ \prod_{k} \Phi(w_k)^{y_k} (1 - \Phi(w_k))^{1-y_k} \right]

        where \mathbf{w} is the product \mathbf{W} \times \mathbf{l}
        '''

        X = X.astype(np.bool)

        if X.shape[1] < X.shape[0]:
            X = X.T

        # for i in xrange(X.shape[0]):
        #     x = X[i,:]
        #
        #     # Now compute the joint log probability
        #     if x.shape[1] > 1:
        #         x = x.T
        #
        #     ret[i] = np.log(norm.cdf(self.w[x == 1])).sum() + np.log(np.ones([1, x.shape[0] - x.sum()]) - norm.cdf(self.w[x == 0])).sum()
        #
        # return ret

        # import ipdb
        # ipdb.set_trace()

        w = np.tile(self.w, X.shape[1])

        im = np.multiply(w, X)
        t1 = np.log(norm.cdf(im)).sum(axis=0)

        iX = np.invert(X)
        im = np.ones(w.shape) - norm.cdf(w)
        t2 = np.multiply(np.log(im), iX).sum(axis=0)

        return t1 + t2

    # TODO: Implement this to actually do something
    def resample(self,data=[]):
        pass

class LinearGaussian(GibbsSampling, Distribution):
    ''' Linear-gaussian  model for the UA Context model of the REACH team

        The Gaussian distributions has and identity covariance matrix

        Author: Enrique Noriega
        Email: enoriega@email.arizona.edu
    '''

    @property
    def others(self):
        return self._others

    @others.setter
    def others(self, ot):
        self._others = ot

    @others.deleter
    def others(self):
        del self._others
        self._others = np.matrix(np.zeros(self.l.shape))

    def __init__(self, W, l, ah=.1, bh=.1, bayesian=True):
        ''' W is the weight matrix and l is the state vector

            l is a binary vector and W a real matrix
        '''

        self.bayesian = bayesian
        # Hyper parameters for the covariance
        self.ah = ah
        self.bh = bh


        self.W = np.matrix(W)

        # Enforce l to be a column vector
        if l.shape[1] > 1:
            l = l.T

        # Initial identity covariance matrix
        self.variances = np.ones(W.shape[0])

        self.l = np.concatenate([np.matrix(l), np.matrix([1])]) # Added the bias term
        self.others = np.matrix(np.zeros(self.l.shape))


        l = self.l # To avoid any further confusion

        assert W.shape[1] == l.shape[0], "The weight matrix and the state vector should be equivalent"
        assert ((l <= 1) & (l >= 0)).all(), "l should be a binary vector"


    def rvs(self, size=1):
        ''' Generates a random variate (sample)
        '''

        # Compute the state vector given the other chains
        global_state = self.l + self.others

        # Correct the overflow in the bias term
        global_state[-1, :] = 1.

        # Compute the weights vector, which becomes the mean vector
        w = self.W*global_state #These should be numpy's matrix objects so this works

        if type(size) in (list, tuple):
            size = size[0] if len(size) > 0 else 1

        variate = np.matrix(np.zeros((size, w.shape[0])))

        for j in xrange(w.shape[0]):
            #TODO: Rewrite this to avoid the loop
            variate[:, j] = stats.norm.rvs(size=(size, 1), loc=w[j], scale=np.sqrt(self.variances))

        return variate


    def log_likelihood(self, X):
        ''' Computes the log likelihood according to the following formula:
        '''

        # Compute the state vector given the other chains
        global_state = self.l + self.others

        # Correct the overflow in the bias term
        global_state[-1, :] = 1.

        # Sanity check
        assert global_state[global_state > 1].any() == False, "There is a problem rebuilding the global state matrix"

        # Compute the weights vector, which becomes the mean vector
        w = self.W*global_state #These should be numpy's matrix objects so this works

        # if X.shape[1] < X.shape[0]:
        #     X = X.T
        # import ipdb; ipdb.set_trace()

        return np.sum(norm.logpdf(X, loc=w.T, scale=np.sqrt(self.variances)), axis=1) # w is a column vector so we transpose it to leverage numpy's broadcast.


    # TODO: Implement this to actually do something, right now all parameters are fixed.
    def resample(self,data=[]):

        return
        if self.bayesian:
            # Compute the states matrix of the whole sequence given the other chains' states
            global_state = self.l + self.others

            # Correct the overflow in the bias term
            global_state[-1, :] = 1.

            # Sanity check
            assert global_state[global_state > 1].any() == False, "There is a problem rebuilding the global state matrix"
            w = self.W*global_state #These should be numpy's matrix objects so this works

            # Reconstruct the indices of the data
            data = data[0]

            idx = np.zeros(data.shape[0], dtype=int)
            for i in xrange(data.shape[0]):
                idx[i] = np.where(np.all(self.global_data == data[i, :], axis=1))[0]

            # Compute the sum of squared errors
            centered = np.power(data - w.T[idx, :], 2)

            sse = np.sum(centered, axis=0)

            variances = np.ones(sse.shape[1])
            # Resample the variances
            for i in xrange(sse.shape[1]):
                alpha = self.ah + (self.global_data.shape[0]/2.)
                beta = 1. /(self.bh + (sse[0, i]/2.))

                sample = invgamma.rvs(alpha, beta)
                variances[i] = sample

            self.variances = variances
