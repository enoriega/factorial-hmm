from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pyhsmm
from pyhsmm.util.text import progprint_xrange

from arizona import LinearGaussian


# Read data
observations = np.loadtxt('obs.txt')
real_states = np.loadtxt('states.txt')
weights = np.loadtxt('weights.txt')

# Add the bias term to the observations
observations = np.concatenate([observations, np.ones((observations.shape[0], 1))], axis=1)

T = observations.shape[0]
COMP = 7
Nmax = 2 # Two states per component

# observation distributions
obs_distns = [LinearGaussian(W, np.matrix(l)) for l in [ [int(i) for i in reversed(list('{0:5b}'.format(state).replace(' ', '0')))] for state in xrange(Nmax)]]


### construct posterior model
posteriormodel = pyhsmm.models.HMM(obs_distns)

posteriormodel.add_data(data=observations)

# This is Gibbs sampling
nsubiter=25
for itr in progprint_xrange(10):
    posteriormodel.resample_model()

plt.figure(); plt.plot(posteriormodel.states_list[0].museqs);
plt.title('sampled after %d iterations' % ((itr+1)))

plt.show()
