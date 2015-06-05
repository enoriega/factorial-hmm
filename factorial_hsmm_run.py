from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pandas as pd
import pyhsmm
from pyhsmm.util.text import progprint_xrange

from arizona import LinearGaussian
from scipy.stats import invgamma

from utils import *


plt.ion()

# Read data
observations = np.loadtxt('obs.txt')
real_states = np.loadtxt('states.txt')
weights = np.loadtxt('weights.txt').T

# Add the bias term to the observations
#observations = np.concatenate([observations, np.ones((observations.shape[0], 1))], axis=1)

ITER = 1000 # Number of iterations of MCMC
T = observations.shape[0]
COMP = 7
Nmax = 2 # Two states per component


### construct posterior model

dur_hypparams = {'alpha_0':10,
                 'beta_0':2}
dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(2)]

# Instantiate an HMM per factor
components = []
for i in xrange(COMP):
    component = pyhsmm.models.HSMM(alpha=6., init_state_concentration=.5, obs_distns = [
        LinearGaussian(weights, np.matrix(get_binary_vector(l, i, COMP))) for l in xrange(2)
    ], dur_distns = dur_distns)

    component.add_data(data=observations, trunc=60)
    components.append(component)



# Data structure to collect results
results = []#pd.DataFrame(columns=['Log-Likelihood', 'Accuracy', 'Precision', 'Recall', 'F1'])

# Collect stats into the dataframe
stats = collect_stats(components, real_states, weights)
results.append(list(stats))


# This is Gibbs sampling
for itr in progprint_xrange(ITER):
    # In each resamle, we are going to fix all the chains except one, then resample
    # that chain given all the other fixed ones. After doing this, a single
    # resample of the factorial model is done

    # Resample the variances
    states = np.matrix(np.zeros((COMP+1, observations.shape[0])))
    states[-1, :] = 1

    for i, component in enumerate(components):
        states[i, :] = component.states_list[0].stateseq

    # Now compute the means
    means = np.matrix(weights)*states
    # Squared summed error
    sse = np.power(observations - means.T, 2).sum(axis=0)



    new_variances = np.zeros((1, sse.shape[1]))

    for i in xrange(sse.shape[1]):
        alpha = .05 + (observations.shape[0]/ 2.)
        beta = (.05 + (sse[0, i]/2.))
        new_variances[0, i] = invgamma.rvs(alpha, scale=beta)

    del states

    for component in components:

        # The number of components plus a bias term
        global_states = np.zeros((COMP+1, observations.shape[0]))
        global_states[-1, :] = 1. # The bias term

        for i, other in enumerate(components):
            # Only if this is another chain
            if other != component:
                seq = other.states_list[0].stateseq
                global_states[i, :] = seq

        for obs_dist in component.obs_distns:
            obs_dist.others = global_states
            obs_dist.variances = new_variances
            #obs_dist.global_data = observations

        component.resample_model()

        del global_states # To avoid memory leaks





    # Do something to collect the hidden states and the likelihood of the model if
    # this is the appropriate iteration
    if (itr % 1) == 0:
        # Collect stats into the dataframe
        stats = collect_stats(components, real_states, test_obs, weights)
        results.append(list(stats))


frame = pd.DataFrame(results, columns=['llj', 'acc', 'pr', 'rc', 'f1'])

frame.llj.plot()
plt.show()

print "Done!!"
