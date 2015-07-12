import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import norm
def get_binary_vector(state, comp_no, COMP):
    ''' Returns a binary vector where the state is set in the current component '''

    ret = np.zeros((COMP, 1))
    ret[comp_no, :] = state

    return ret

def collect_stats(components, real_states, real_obs, testing_obs, weights, vars, Model):
    ''' Returns stats about the current approx of the posterior '''

    weights = np.matrix(weights)

    # Training log-lokelihood
    training_states = get_states_matrix(components)
    llj = compute_likelihood(real_obs, training_states, weights, vars)

    # Testing log-likelihood
    # Generate a new sample
    testing_states = sample_testing_states(components, testing_obs, Model)
    tll = np.array([compute_likelihood(testing_obs, tc, weights, vars) for tc in testing_states]).mean()


    predicted = np.array([c.states_list[0].stateseq for c in components]).T

    assert predicted.shape == real_states.shape, "Error in the dimensions of the predictions"

    acc = accuracy_score(predicted.flatten(), real_states.flatten())
    pr, rc, f1, _ = precision_recall_fscore_support(predicted.flatten(), real_states.flatten())

    return llj/400.,acc,pr[1],rc[1],f1[1], tll/400.


def sample_testing_states(components, testing_obs, Model):


    ret = []
    testing_components = []
    for component in components:

        if hasattr(component, 'dur_distns'):
            tc = Model(trans_distn=component.trans_distn, dur_distns=component.dur_distns, obs_distns=component.obs_distns, init_state_distn=component.init_state_distn)
        else:
            tc = Model(trans_distn=component.trans_distn, obs_distns=component.obs_distns, init_state_distn=component.init_state_distn)

        tc.add_data(testing_obs)

        testing_components.append(tc)

    for i in xrange(100):

        for tc in testing_components:

            # Now sample states for the testing observations
            testing_states = np.zeros((len(components)+1, testing_obs.shape[0]))
            testing_states[-1, :] = 1. # The bias term


            for i, other in enumerate(testing_components):
                # Only if this is another chain
                if other != tc:
                    seq = other.states_list[0].stateseq
                    testing_states[i, :] = seq

            for obs_dist in tc.obs_distns:
                obs_dist.others = testing_states
                # Set the new variances as part of resampling the emission model

            # Resample the rest of the model:
            tc.resample_states()

            del testing_states

        ret.append(get_states_matrix(testing_components))


    return ret


def get_states_matrix(components):

    chains = [c.states_list[0].stateseq for c in components]
    chains.append(np.array([1 for i in xrange(components[0].states_list[0].stateseq.shape[0])]))

    return np.matrix(chains)

def compute_likelihood(data, states, weights, variances):

    W = weights*states
    centered = data.T - W
    sse = np.power(centered.sum(axis=1), 2)

    ll = np.array([-(0.5*(1./variances[0, k])*sse[k, 0] + np.log(2*np.pi*variances[0, k])) for k in xrange(sse.shape[0])]).sum()

    return ll


def write_files(frame):
    ''' Writes the scores using Colin's format '''

    for fname, series in zip(['train_log_likelihood.txt', 'accuracy.txt', 'precision.txt', 'recall.txt', 'f1.txt', 'test_log_likelihood.txt'],
        [frame.llj, frame.acc, frame.pr, frame.rc, frame.f1, frame.tll]):

        with open(fname, 'w') as f:
            f.write('iteration value\n')
            series.to_csv(f, sep=' ', header=False)
