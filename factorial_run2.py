from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from matplotlib import pyplot as plt

import pandas as pd
import pyhsmm
from pyhsmm.util.text import progprint_xrange

from arizona import LinearGaussian


# Read data
observations = np.loadtxt('obs.txt')
real_states = np.loadtxt('states.txt')
weights = np.loadtxt('weights.txt').T

# Add the bias term to the observations
#observations = np.concatenate([observations, np.ones((observations.shape[0], 1))], axis=1)

BURNIN = 100
ITER = 5000 # Number of iterations of MCMC
T = observations.shape[0]
COMP = 7
Nmax = 2 # Two states per component


def get_binary_vector(state, comp_no):
    ''' Returns a binary vector where the state is set in the current component '''

    ret = np.zeros((COMP, 1))
    ret[comp_no, :] = state

    return ret

def collect_stats(components):
    ''' Returns stats about the current approx of the posterior '''


    llj = sum([c.log_likelihood() for c in components])
    #llj = components[0].log_likelihood()

    predicted = np.array([c.states_list[0].stateseq for c in components]).T

    assert predicted.shape == real_states.shape, "Error in the dimensions of the predictions"


    return llj,0,0,0,0

### construct posterior model

# Instantiate an HMM per factor
components = []
for i in xrange(COMP):
    component = pyhsmm.models.HMM(alpha=6.,init_state_concentration=.5, obs_distns = [
        LinearGaussian(weights, np.matrix(get_binary_vector(l, i))) for l in xrange(2)
    ])

    component.add_data(data=observations)
    components.append(component)


# Data structure to collect results
results = []#pd.DataFrame(columns=['Log-Likelihood', 'Accuracy', 'Precision', 'Recall', 'F1'])


# This is Gibbs sampling
for itr in progprint_xrange(ITER):
    # In each resamle, we are going to fix all the chains except one, then resample
    # that chain given all the other fixed ones. After doing this, a single
    # resample of the factorial model is done

    # The number of components plus a bias term
    global_states = np.zeros((COMP+1, observations.shape[0]))
    global_states[-1, :] = 1. # The bias term

    for i, other in enumerate(components):
        seq = other.states_list[0].stateseq
        global_states[i, :] = seq

    for i, component in enumerate(components):

        global2 = global_states.copy()
        global2[i, :] = 0.

        for obs_dist in component.obs_distns:
            obs_dist.others = global2

        component.resample_model()

    del global_states # To avoid memory leaks


    # Do something to collect the hidden states and the likelihood of the model if
    # this is the appropriate iteration
    if itr >= BURNIN and (itr % 10) == 0:
        # Collect stats into the dataframe
        stats = collect_stats(components)
        results.append(list(stats))

frame = pd.DataFrame(results, columns=['llj', 'acc', 'pr', 'rc', 'f1'])
print "Done!!"
