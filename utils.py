import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import norm
def get_binary_vector(state, comp_no, COMP):
    ''' Returns a binary vector where the state is set in the current component '''

    ret = np.zeros((COMP, 1))
    ret[comp_no, :] = state

    return ret

def collect_stats(components, real_states, testing_obs, weights, vars):
    ''' Returns stats about the current approx of the posterior '''


    llj = sum([c.log_likelihood() for c in components])


    # Observed log-likelihood
    weights = np.matrix(weights)

    state_seqs = np.matrix(np.ones((8, testing_obs.shape[0])))
    state_seqs[:7, :] = [c.heldout_viterbi(testing_obs) for c in components]

    # import ipdb; ipdb.set_trace()

    means = state_seqs.T * weights.T

    tll = norm.logpdf(testing_obs, loc=means, scale=np.sqrt(vars)).sum()
    ########################


    predicted = np.array([c.states_list[0].stateseq for c in components]).T

    assert predicted.shape == real_states.shape, "Error in the dimensions of the predictions"

    acc = accuracy_score(predicted.flatten(), real_states.flatten())
    pr, rc, f1, _ = precision_recall_fscore_support(predicted.flatten(), real_states.flatten())

    return llj,acc,pr[1],rc[1],f1[1], tll


def write_files(frame):
    ''' Writes the scores using Colin's format '''

    for fname, series in zip(['train_log_likelihood.txt', 'accuracy.txt', 'precision.txt', 'recall.txt', 'f1.txt', 'test_log_likelihood.txt'],
        [frame.llj, frame.acc, frame.pr, frame.rc, frame.f1, frame.tll]):

        with open(fname, 'w') as f:
            f.write('iteration value\n')
            series.to_csv(f, sep=' ', header=False)
