
import numpy as np


def adaboost(diff, n_pos, n_neg, n_tot, T=1000):
    """
    Runs Adaboost algorithm. Adaboost is a very simple algorithm to combine several weak
    learners to one strong classifier. It works by iteratively selecting a feature using min error,
    obtaining predictions and errors, then re-weighting the samples so that misclassified samples
    have higher weights and repeating this process for T iterations.
    ---------
    Arguments
    ---------
    diff [np.array]: Numpy array of shape (m,d) with 0 where prediction is
                     correct and 1 at other places.
    n_pos [int]: Number of positive samples
    n_neg [int]: Number of negative samples
    n_tot [int]: Total number of samples (n_tot=n_pos+n_neg)
    T [int]: Number of iterations of Adaboost
    -------
    Returns
    -------
    chosen_feature_indices [list(int)]: Indices of chosen features
    betas [list(float)]: Beta values at each iteration
    """
    weights = np.zeros((n_tot,))
    weights[:n_pos] = 1/(2*n_pos)
    weights[n_pos:] = 1/(2*n_neg)
    weights = weights/weights.sum()
    chosen_feature_indices, betas = [], []
    for t in range(T):
        errors = (weights.reshape(-1,1) * diff.astype(float)).sum(0)
        chosen_feature_idx = errors.argmin()
        min_error = errors[chosen_feature_idx]
        beta = min_error/(1-min_error)
        incorrectly_clasified = diff[:,chosen_feature_idx]
        incorrectly_clasified = incorrectly_clasified.astype(float)
        weights = weights * ( beta ** (1-incorrectly_clasified) )
        weights = weights/weights.sum()
        chosen_feature_indices.append(chosen_feature_idx)
        betas.append(beta)
        print(t, chosen_feature_idx, beta, min_error)
    return chosen_feature_indices, betas

def adaboost_prediction(test_data_, chosen_feature_indices_, betas_, theta_vector_, parity_vector_):
    """
    Returns adaboost prediction
    ---------
    Arguments
    ---------
    test_data_ [np.array]: Numpy array of shape (m,d) denoting test data
    chosen_feature_indices_ [list(int)]: Indices of chosen features
    betas_ [list(float)]: Beta values at each iteration of adaboost   
    theta_vector_ [np.array]: Numpy array of shape (d,) denoting thresholds
    parity_vector_ [np.array]: Numpy array of shape (d,) with (1,-1) denoting sign of inequality
    -------
    Returns
    -------
    pred_fin [np.array]: Numpy array of shape (m,) denoting final predictions
    """
    test_data_sub = test_data_[:,chosen_feature_indices_]
    theta_vector_sub = theta_vector_[chosen_feature_indices_]
    parity_vector_sub = parity_vector_[chosen_feature_indices_]
    preds = predict_full_data(test_data_sub,theta_vector_sub, parity_vector_sub)
    alphas = -np.log(np.array(betas_))
    pred_fin = np.zeros((len(test_data),))
    pred_fin[(preds * alphas.reshape(1,-1)).sum(1) >= .5*alphas.sum()] = 1
    return pred_fin

