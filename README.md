
# Viola Jones Face Detection

This repo is an implementation of Viola Jones face detection framework. This method from 2001 is widely used even now in many devices for face detection.

Link to paper: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

## Main components
- Haar features are used for the classifier.
- To quickly calculate Haar features, integral image is calculated
- Weak classifiers are trained for each feature resulting in one decision stump ([wiki](https://en.wikipedia.org/wiki/Decision_stump)) per feature. Hence we essentially get a threshold (`float`) and a parity (`bool`, which side is +ve) for each features.
- These weak classifiers are combined using adaboost ([wiki](https://en.wikipedia.org/wiki/AdaBoost)). Adaboost works through multiple rounds of feature ranking and sample weighting. In first round, all samples have equal weightage and with this weightage we select the best feature. In next round, the misclassified samples from previous round have higher weights. Thus we get a strong classifier from several weak classifiers. In this way we get a feature ranking at the end. 
- For inference, a cascaded classifier is created using the feature ranking obtained. Thus non-faces are quickly rejected whereas faces go through all the gates and finally return True.
