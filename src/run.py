
from utils import save_cropped_rotated_faces, select_faces,augment_horizontally_inverted
from haar_utils import get_dataset
from classifier_utils import train_weak_classifiers
from adaboost import adaboost
import os, time
import numpy as np


def prepare_dataset_imgs():
    # Prepare data/face_detection_dataset
    os.makedirs("data/cropped_faces",exist_ok=True)
    os.makedirs("data/selected_faces",exist_ok=True)
    save_cropped_rotated_faces("data/FDDB", "data/cropped_faces")
    select_faces("data/cropped_faces", "data/selected_faces", min_size=110)
    augment_horizontally_inverted("data/SUN_data/*/*/*.jpg")

def get_haar_feat_ds():
    # Haar features
    t1 = time.time()
    train_data, train_labels, n_pos, n_neg, n_tot = get_dataset("face_detection_dataset/train/face/*.jpg", \
                                                "face_detection_dataset/train/nonface/*.jpg"
                                                )
    np.save("train_data.npy", train_data)
    np.save("train_labels.npy", train_labels)
    t2 = time.time()
    print("Time taken:{} secs".format(t2-t1))

    t1 = time.time()
    test_data, test_labels, _,_,_ = get_dataset("face_detection_dataset/test/face/*.jpg", \
                                                "face_detection_dataset/test/nonface/*/*.jpg"
                                                )
    np.save("test_data.npy", test_data)
    np.save("test_labels.npy", test_labels)
    t2 = time.time()
    print("Time taken:{} secs".format(t2-t1))
    return train_data, train_labels, test_data, test_labels


def main():
    # prepare_dataset_imgs()
    train_data, train_labels, test_data, test_labels = get_haar_feat_ds()

    t1 = time.time()
    theta_vector, parity_vector = train_weak_classifiers(train_data,train_labels)
    np.save("theta_vector.npy", theta_vector)
    np.save("parity_vector.npy", parity_vector)
    t2 = time.time()
    print("Time taken:{} secs".format(t2-t1))

    preds = predict_full_data(train_data,theta_vector, parity_vector).astype(int)
    preds = preds.astype(int)
    np.save("preds.npy", preds)

    diff = np.abs(train_labels.reshape(-1,1)-preds)
    np.save("diff.npy", diff)

    n_tot = len(train_labels)
    n_pos = int(train_labels.sum())
    n_neg = n_tot-n_pos

    chosen_feature_indices, betas = adaboost(diff, n_pos, n_neg, n_tot, T=50)

    pred_final = adaboost_prediction(test_data, chosen_feature_indices, betas, theta_vector, parity_vector)

    acc = sum(pred_final==test_labels)/len(test_labels)
    print("Accuracy:", acc)
    # 90.64%

    

