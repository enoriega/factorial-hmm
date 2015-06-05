import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.spatial.distance import hamming
def get_binary_vector(state, comp_no, COMP):
    ''' Returns a binary vector where the state is set in the current component '''

    ret = np.zeros((COMP, 1))
    ret[comp_no, :] = state

    return ret

def collect_stats(components, real_states):
    ''' Returns stats about the current approx of the posterior '''


    llj = sum([c.log_likelihood() for c in components])
    #llj = components[0].log_likelihood()

    predicted = np.array([c.states_list[0].stateseq for c in components]).T

    assert predicted.shape == real_states.shape, "Error in the dimensions of the predictions"

    acc = accuracy_score(predicted.flatten(), real_states.flatten())
    pr, rc, f1, _ = precision_recall_fscore_support(predicted.flatten(), real_states.flatten())

    return llj,acc,pr[1],rc[1],f1[1]
