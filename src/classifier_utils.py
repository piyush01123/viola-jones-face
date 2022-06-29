
import numpy as np


def train_weak_classifiers(data,labels):
    """
    Trains weak classifiers.    
    ---------
    Arguments
    ---------
    data [np.array]: Numpy array of shape (m,162336)
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    -------
    Returns
    -------
    theta_vector [np.array]: Numpy array of shape (162336,) denoting thresholds
    parity_vector [np.array]: Numpy array of shape (162336,) with (1,-1) denoting sign of inequality
    """
    num_samples, num_features = data.shape
    theta_vector = np.zeros((num_features,))
    parity_vector = np.zeros((num_features,))
    for feat_id in range(num_features):
        features = data[:,feat_id]
        indices = features.argsort()
        labels_sorted = labels[indices]
        num_corrects = np.zeros((2,num_samples))
        for t in range(num_samples):
            num_corrects[0,t] = t-labels_sorted[:t].sum()+labels_sorted[t:].sum()
            num_corrects[1,t] = num_samples - num_corrects[0,t]
        idx = num_corrects.argmax()
        parity = idx//num_samples
        theta = features[indices][idx%num_samples]
        theta_vector[feat_id] = theta
        parity_vector[feat_id] = parity
        # if feat_id==50:
        #     break
    parity_vector = 1-parity_vector
    parity_vector[parity_vector==0] = -1
    return theta_vector, parity_vector

def predict(feature_vec,theta,parity):
    """
    Returns predictions for m samples with a specific feature.
    ---------
    Arguments
    ---------
    feature_vec [np.array]: Numpy array of shape (m,) denoting a particular feature of m samples
    theta [float]: Threshold
    parity [parity]: Sign of inequality
    -------
    Returns
    -------
    pred [np.array]: Numpy array of shape (m,) denoting predictions of m samples in (1,-1) notation
    """
    return parity*np.sign(feature_vec-theta)

def predict_full_data(data,theta_vec,parity_vec, majority_label=0):
    """
    Returns predictions for m samples with multiple features. Each sample x feature combination has a prediction
    ---------
    Arguments
    ---------
    data [np.array]: Numpy array of shape (m,162336)
    theta_vec [np.array]: Numpy array of shape (162336,) denoting thresholds
    parity_vec [np.array]: Numpy array of shape (162336,) with (1,-1) denoting sign of inequality
    majority_label [int]: Majority label. In case feature==threshold, majority label is assigned
    -------
    Returns
    -------
    preds [np.array]: Numpy array of shape (m,162336) denoting predictions in (1,0) notation
    """
    num_samples, num_features = data.shape
    preds = np.zeros((num_samples,num_features))
    for feat_id in range(num_features):
        feature_vec = data[:,feat_id]
        preds[:,feat_id] = predict(feature_vec,theta_vec[feat_id],parity_vec[feat_id])
        # if feat_id==50:
        #     break
    preds[preds==0] = majority_label
    preds[preds==-1] = 0
    preds = preds.astype(int)
    return preds

def accumulate_accs(labels,preds):
    """
    Measure performance of weak classifiers
    ---------
    Arguments
    ---------
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    preds [np.array]: Numpy array of shape (m,162336) denoting predictions in (1,0) notation
    -------
    Returns
    -------
    accs [np.array]: Numpy array of shape (162336,) denoting accuracies of all features
    """
    num_samples, num_features = preds.shape
    accs = np.zeros((num_features,))
    for feat_id in range(num_features):
        preds_vec = preds[:,feat_id]
        assert labels.shape == preds_vec.shape
        acc = (labels==preds_vec).sum()/num_samples
        accs[feat_id] = acc
    return accs


def print_accs(labels,preds):
    """
    Prints performance of weak classifiers
    ---------
    Arguments
    ---------
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    preds [np.array]: Numpy array of shape (m,162336) denoting predictions in (1,0) notation
    -------
    Returns
    -------
    None
    """
    labels[labels==0] = -1
    num_samples, num_features = preds.shape
    for feat_id in range(num_features):
        preds_vec = preds[:,feat_id]
        assert labels.shape == preds_vec.shape
        acc = (labels==preds_vec).sum()/num_samples
        print(feat_id, acc)
        # if feat_id==50:
        #     break

