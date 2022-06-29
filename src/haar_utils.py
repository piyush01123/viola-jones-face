
import os,glob
import cv2
import numpy as np


"""
Haar features

    Type A: Horizontal 2-strip (eye type)
    Type B: Horizontal 3-strip bridge
    Type C: Vertical 2-strip
    Type D: Vertical 3-strip
    Type E: Cross
"""


def haar_features_A(patch_integral, patch_size=24):
    """
    Obtain Haar features of type A
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,patch_size-i,1):
                for w in range(0,(patch_size-j)//2,1):
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2)
    return np.array(features)


def haar_features_B(patch_integral, patch_size=24):
    """
    Obtain Haar features of type B
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,patch_size-i,1):
                for w in range(0,(patch_size-j)//3,1):
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+2*w,j+3*w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2+s3)
    return np.array(features)


def haar_features_C(patch_integral, patch_size=24):
    """
    Obtain Haar features of type C
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//2,1):
                for w in range(0,patch_size-j,1):
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2)
    return np.array(features)


def haar_features_D(patch_integral, patch_size=24):
    """
    Obtain Haar features of type D
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//3,1):
                for w in range(0,patch_size-j,1):
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+2*h,i+3*h), (j,j+w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2+s3)
    return np.array(features)


def haar_features_E(patch_integral, patch_size=24):
    """
    Obtain Haar features of type E
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//2,1):
                for w in range(0,(patch_size-j)//2,1):
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j+w,j+2*w)
                    s4 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2-s3+s4)
    return np.array(features)

def get_selected_haar_features(patch_integral, feature_indices, patch_size=24):
    """
    Obtain Haar features of all types (A,B,C,D,E) with selected indices only
    ---------
    Arguments
    ---------
    patch_integral [np.array]: Numpy array of shape (h,w) denoting integral image/patch
    feature_indices [list(int)]: List containing feature indices that we want to get
    patch_size [int]: Size of image/patch
    -------
    Returns
    -------
    features: Numpy array denoting features
    """
    features = []
    ctr = 0
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,patch_size-i,1):
                for w in range(0,(patch_size-j)//2,1):
                    if ctr not in feature_indices:
                        ctr += 1
                        continue
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2)
                    ctr += 1
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,patch_size-i,1):
                for w in range(0,(patch_size-j)//3,1):
                    if ctr not in feature_indices:
                        ctr += 1
                        continue
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+2*w,j+3*w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2+s3)
                    ctr += 1
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//2,1):
                for w in range(0,patch_size-j,1):
                    if ctr not in feature_indices:
                        ctr += 1
                        continue
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2)
                    ctr += 1
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//3,1):
                for w in range(0,patch_size-j,1):
                    if ctr not in feature_indices:
                        ctr += 1
                        continue
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+2*h,i+3*h), (j,j+w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2+s3)
                    ctr += 1
    for i in range(patch_size):
        for j in range(patch_size):
            for h in range(0,(patch_size-i)//2,1):
                for w in range(0,(patch_size-j)//2,1):
                    if ctr not in feature_indices:
                        ctr += 1
                        continue
                    tl, br = (i,i+h), (j,j+w)
                    s1 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j,j+w)
                    s2 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i,i+h), (j+w,j+2*w)
                    s3 = get_sum_pixels(patch_integral, tl, br)
                    tl, br = (i+h,i+2*h), (j+w,j+2*w)
                    s4 = get_sum_pixels(patch_integral, tl, br)
                    features.append(s1-s2-s3+s4)
                    ctr += 1
    return np.array(features)


def get_dataset_fast(face_glob, non_face_glob, feature_indices):
    """
    Gets selected haar features fast.
    ---------
    Arguments
    ---------
    face_glob [list(str)]: Glob of image paths containing faces
    non_face_glob [list(str)]: Glob of image paths not containing faces
    feature_indices [list(int)]: List containing feature indices that we want to get   
    -------
    Returns
    -------
    data [np.array]: Numpy array of shape (m,162336)
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    n_pos [int]: Number of face images
    n_neg [int]: Number of non-face images
    n_tot [int]: Total number of images
    """
    face_imgs = sorted(glob.glob(face_glob))
    non_face_imgs = sorted(glob.glob(non_face_glob))
    n_pos, n_neg = len(face_imgs), len(non_face_imgs)
    n_tot = n_pos+n_neg
    labels = np.zeros((n_tot,), dtype=int)
    labels[:n_pos] = 1

    data = np.zeros((n_tot, len(feature_indices)))

    for i, fp in enumerate(face_imgs+non_face_imgs):
        print("Processing img ", i)
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(24,24))
        img = (img-img.mean())/img.std()
        int_img = get_integral_image(img)
        data[i] = get_selected_haar_features(int_img, feature_indices)

    return data, labels, n_pos, n_neg, n_tot

def get_dataset(face_glob, non_face_glob):
    """
    Gets selected haar features fast.
    ---------
    Arguments
    ---------
    face_glob [list(str)]: Glob of image paths containing faces
    non_face_glob [list(str)]: Glob of image paths not containing faces
    -------
    Returns
    -------
    data [np.array]: Numpy array of shape (m,162336)
    labels [np.array]: Numpy array of shape (m,) with ones for faces and zeros for non-faces
    n_pos [int]: Number of face images
    n_neg [int]: Number of non-face images
    n_tot [int]: Total number of images
    """
    face_imgs = sorted(glob.glob(face_glob))
    non_face_imgs = sorted(glob.glob(non_face_glob))
    n_pos, n_neg = len(face_imgs), len(non_face_imgs)
    n_tot = n_pos+n_neg
    labels = np.zeros((n_tot,), dtype=int)
    labels[:n_pos] = 1

    a,b,c,d,e = 43200,27600,43200,27600,20736
    data = np.zeros((n_tot, a+b+c+d+e))

    for i, fp in enumerate(face_imgs+non_face_imgs):
        print("Processing img ", i)
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(24,24))
        img = (img-img.mean())/img.std()
        int_img = get_integral_image(img)
        data[i,:a] = haar_features_A(int_img)
        data[i,a:a+b] = haar_features_B(int_img)
        data[i,a+b:a+b+c] = haar_features_C(int_img)
        data[i,a+b+c:a+b+c+d] = haar_features_D(int_img)
        data[i,a+b+c+d:a+b+c+d+e] = haar_features_E(int_img)


    return data, labels, n_pos, n_neg, n_tot

