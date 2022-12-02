
# Viola Jones Face Detection
<p align="center">
<img src="https://user-images.githubusercontent.com/19518507/205269807-fae84c58-0ff7-4cce-beb2-995b2e5788c9.png" width="500">
</p>

This repo is an implementation of Viola Jones face detection framework. This method from 2001 is widely used even now in many devices for face detection.

Link to paper: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

## How it works
### Step 1: Calculate integral image
Integral image allows us to quickly compute the feature values. Here is a demo of an integral image. Basically each pixel contains its own value added with all the cells in its upper left portion of grid

![integral_image_b](https://user-images.githubusercontent.com/19518507/205270652-61eaca26-1016-4bfa-ac10-5553d07c084b.png)

### Step 2: Calculate Haar feature values
There are five types of Haar features:

![haar](https://user-images.githubusercontent.com/19518507/205271250-c653b965-ba15-4039-b71a-f55d49d143f3.png)

We calculate these at every possible scale ie each feature is used several thousand times. We are going to scan each feature across the image to get the features.

### Step 3: Train weak classifiers
Weak classifiers are trained for each feature resulting in one [decision stump](https://en.wikipedia.org/wiki/Decision_stump) per feature. Hence we essentially get a threshold (`float`) and a parity (`bool`, which side is +ve) for each features.

### Step 4: AdaBoost
These weak classifiers are combined using [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost). AdaBoost works through multiple rounds of feature ranking and sample weighting. In first round, all samples have equal weightage and with this weightage we select the best feature. In next round, the misclassified samples from previous round have higher weights. Thus we get a strong classifier from several weak classifiers. In this way we get a feature ranking at the end. 

### Step 5: Inference
For inference, a cascaded classifier is created using the feature ranking obtained. Thus non-faces are quickly rejected whereas faces go through all the gates and finally return True.
