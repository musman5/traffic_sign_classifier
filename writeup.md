# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_data_set.png "Training Data"
[image2]: ./examples/valid_data_set.png "Validity Data"
[image3]: ./examples/test_data_set.png "Test Data"
[image4]: ./sample_images/bumpy_road.jfif "Bumpy Road Sign"
[image5]: ./sample_images/no_entry.jfif "No Entry"
[image6]: ./sample_images/Right_of_way_at_next_intersection.jpg "Right of way at next intersection"
[image7]: ./sample_images/road_work.jfif "Road Work"
[image8]: ./sample_images/speed_limit_70.jpg "Speed Limit 70"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is = 34799
* The size of the validation set is = 4410
* The size of test set is = 12630
* The shape of a traffic sign image is = (32, 32, 3)
* The number of unique classes/labels in the data set is = 43

#### 2. Include an exploratory visualization of the dataset.

My visualization of the data set is as below. 
These bar charts plots number of classes on x-axis and number of images present in each class on y-axis.
There are 43 classes. The validation, test and training set data is spread over each class not evenly. Some classes have high number of images compared with other classes.

Training Data Set:
![alt text][examples/training_data_set.png]
Valid Data Set:
![alt text][examples/valid_data_set.png]
Test Data Set:
![alt text][examples/test_data_set.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because traffic signs are differentiable from the contents and shapes and color is not important. So we can ignore color here and convert image to gray scale.

Gray scaling is not necessary, but gray scaling was has benifits:
- Gray scale image use small memory as it has only one channel instead of three in color images
- Gray scale images are of small size so processing can be much faster than color images.

As a last step, I normalized the image data because image data should have mean zero and have equal variance.

Below is the code snippet for gray scale conversion and normalization:
def convertToGrayscale(color_images):
    return np.sum(color_images/3, axis = 3, keepdims = True)
def normalize(grayscaled_images):
    return (grayscaled_images -128) / 128

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Relu is used as activation function.
Max pooling operation is performed after each convolution with 2x2 window and stride of 2.

1. First Convolution : input 32x32x1 -> output 28x28x6 -> relu -> drop out-> pooling output 14x14x6
2. Second Convolution : input 14x14x6 -> output 10x10x16 -> relu -> pooling output 5x5x16
3. Flatten : Input 5x5x16 output 400
4. Fully Connected : input 400 output 120
5, Drop out
6. Relu.
7. Fully Connected : input 120 output 84
8. Relu.
9. Drop out
10. Fully Connected: input 84 output 43



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 64 epochs (iterations) with batch size of 128 for each epoch. Also i used Adam Optimizer with learning rate of 0.001

EPOCHS = 64
BATCH_SIZE = 128
mu = 0
sigma = 0.1
rate = 0.001
Drop out at FC0 = 0.6
Drop out at FC2 = 0.6
Drop out at Conv1 = 0.7


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


I used model is similar to the LeNet model introduced in the course.
My model consists of two convolution layers and three fully connected layers.
I used Relu as activation function.
(First i implemented my model without drop out)
Input image -> First Convolution -> Relu and Max Pool -> Second Convolution -> Relu and Max Pool -> Flatten -> Fully Connected -> Relu -> Fully Connected -> Relu -> Fully Connected

For max pool operation i used 2x2 windows.

Since LeNet model is based on Yan Lecun model of 1998. This model was for reconizing the hand written number in an image. For traffic sign project we have to perform similar task to detect and recognize traffice sign model inside an image. So i believe LeNet model is a good starting point for this project. 

Then based on this model i tried to modify the model by adding drop out function and also tried to tune different hyper parameters to get maximum validation accuracy.

I tried to tune several hyper parameters such as learning rate, epoch, batch size and initial weights.

With LeNet architecture and convertion to grayscale the accuracy of model was very low.

EPOCH 1 ...
Validation Accuracy = 0.056
EPOCH 2 ...
Validation Accuracy = 0.061
EPOCH 3 ...
Validation Accuracy = 0.061
EPOCH 4 ...
Validation Accuracy = 0.061
EPOCH 5 ...
Validation Accuracy = 0.054
EPOCH 6 ...
Validation Accuracy = 0.056
EPOCH 7 ...
Validation Accuracy = 0.056
EPOCH 8 ...
Validation Accuracy = 0.054
EPOCH 9 ...
Validation Accuracy = 0.061
EPOCH 10 ...
Validation Accuracy = 0.056


Tune EPOCHS to 10 and batch size to 64. Accuracy improved alot.

EPOCH 1 ...
Validation Accuracy = 0.789
EPOCH 2 ...
Validation Accuracy = 0.903
EPOCH 3 ...
Validation Accuracy = 0.929
EPOCH 4 ...
Validation Accuracy = 0.954
EPOCH 5 ...
Validation Accuracy = 0.953
EPOCH 6 ...
Validation Accuracy = 0.974
EPOCH 7 ...
Validation Accuracy = 0.971
EPOCH 8 ...
Validation Accuracy = 0.962
EPOCH 9 ...
Validation Accuracy = 0.971
EPOCH 10 ...
Validation Accuracy = 0.977

Increase learning rate to 0.01, accuracy decreased slightly.
EPOCH 1 ...
Validation Accuracy = 0.844
EPOCH 2 ...
Validation Accuracy = 0.914
EPOCH 3 ...
Validation Accuracy = 0.937
EPOCH 4 ...
Validation Accuracy = 0.923
EPOCH 5 ...
Validation Accuracy = 0.930
EPOCH 6 ...
Validation Accuracy = 0.945
EPOCH 7 ...
Validation Accuracy = 0.925
EPOCH 8 ...
Validation Accuracy = 0.931
EPOCH 9 ...
Validation Accuracy = 0.948
EPOCH 10 ...
Validation Accuracy = 0.940

Introduce drop ratio as 0.5, accuracy did not imporve.
EPOCH 1 ...
Validation Accuracy = 0.402
EPOCH 2 ...
Validation Accuracy = 0.638
EPOCH 3 ...
Validation Accuracy = 0.730
EPOCH 4 ...
Validation Accuracy = 0.768
EPOCH 5 ...
Validation Accuracy = 0.812
EPOCH 6 ...
Validation Accuracy = 0.823
EPOCH 7 ...
Validation Accuracy = 0.842
EPOCH 8 ...
Validation Accuracy = 0.851
EPOCH 9 ...
Validation Accuracy = 0.860
EPOCH 10 ...
Validation Accuracy = 0.873

Decrease learning rate to 0.0001, accuracy was still low

EPOCH 1 ...
Validation Accuracy = 0.200
EPOCH 2 ...
Validation Accuracy = 0.402
EPOCH 3 ...
Validation Accuracy = 0.543
EPOCH 4 ...
Validation Accuracy = 0.642
EPOCH 5 ...
Validation Accuracy = 0.717
EPOCH 6 ...
Validation Accuracy = 0.755
EPOCH 7 ...
Validation Accuracy = 0.779
EPOCH 8 ...
Validation Accuracy = 0.812
EPOCH 9 ...
Validation Accuracy = 0.826
EPOCH 10 ...
Validation Accuracy = 0.847

Modify initial weights value sigma from 0.1 to 0.01, accuracy decreased slightly.

EPOCH 1 ...
Validation Accuracy = 0.054
EPOCH 2 ...
Validation Accuracy = 0.246
EPOCH 3 ...
Validation Accuracy = 0.552
EPOCH 4 ...
Validation Accuracy = 0.724
EPOCH 5 ...
Validation Accuracy = 0.817
EPOCH 6 ...
Validation Accuracy = 0.862
EPOCH 7 ...
Validation Accuracy = 0.881
EPOCH 8 ...
Validation Accuracy = 0.902
EPOCH 9 ...
Validation Accuracy = 0.921
EPOCH 10 ...
Validation Accuracy = 0.936

Increase initial weight value sigma from 0.1 to 0.2, no big effect on accuracy. Accuracy didnot improve instead dropped minutely.
EPOCH 1 ...
Validation Accuracy = 0.721
EPOCH 2 ...
Validation Accuracy = 0.885
EPOCH 3 ...
Validation Accuracy = 0.901
EPOCH 4 ...
Validation Accuracy = 0.929
EPOCH 5 ...
Validation Accuracy = 0.952
EPOCH 6 ...
Validation Accuracy = 0.954
EPOCH 7 ...
Validation Accuracy = 0.961
EPOCH 8 ...
Validation Accuracy = 0.950
EPOCH 9 ...
Validation Accuracy = 0.967
EPOCH 10 ...
Validation Accuracy = 0.969

Finally i settled down with following parameters:
Convert to graysacle, shuffle and remove drop out layer.

EPOCH = 10
BATCH_SIZE = 64
rate = 0.001
sigma = 0.1

Validation Accuracy = 0.973
Training Accuracy = 0.990
Testing Accuracy = 0.894

This model with gray scale and normalization with batch size 64 and 10 iterations was initial model. This model failed miserably on new images.

Then i introduced drop out in the model. Still model was not performing well on new images.

Next i introduced image tranformation steps, which include image rotation, image translation, image shear. 

Next i found maxium number of images in class data. To make predication smooth i added images to the classes so that each class will contain equal number of images. These image are tranformed images. These will greatly increase the training accuracy of the model.
I saved all the data in pickle file for later use. 

Next i loaded saved pickle data which will include orignal images plus transformed images.
I separated 20% of training images for validation.

Next i tried to use different iterations and settle down with 64 iterations and batch size of 128.

Finally my model accuracy was around 99.8%. This is shown below:

EPOCH 1 ...
Validation Accuracy = 0.793
Training Accuracy = 0.800
Testing Accuracy = 0.457

EPOCH 2 ...
Validation Accuracy = 0.875
Training Accuracy = 0.882
Testing Accuracy = 0.621

EPOCH 3 ...
Validation Accuracy = 0.913
Training Accuracy = 0.920
Testing Accuracy = 0.709

EPOCH 4 ...
Validation Accuracy = 0.938
Training Accuracy = 0.942
Testing Accuracy = 0.763

EPOCH 5 ...
Validation Accuracy = 0.954
Training Accuracy = 0.957
Testing Accuracy = 0.810

EPOCH 6 ...
Validation Accuracy = 0.953
Training Accuracy = 0.959
Testing Accuracy = 0.806

EPOCH 7 ...
Validation Accuracy = 0.967
Training Accuracy = 0.972
Testing Accuracy = 0.835

EPOCH 8 ...
Validation Accuracy = 0.965
Training Accuracy = 0.971
Testing Accuracy = 0.836

EPOCH 9 ...
Validation Accuracy = 0.970
Training Accuracy = 0.976
Testing Accuracy = 0.850

EPOCH 10 ...
Validation Accuracy = 0.971
Training Accuracy = 0.977
Testing Accuracy = 0.851

EPOCH 11 ...
Validation Accuracy = 0.977
Training Accuracy = 0.982
Testing Accuracy = 0.855

EPOCH 12 ...
Validation Accuracy = 0.979
Training Accuracy = 0.984
Testing Accuracy = 0.862

EPOCH 13 ...
Validation Accuracy = 0.980
Training Accuracy = 0.985
Testing Accuracy = 0.859

EPOCH 14 ...
Validation Accuracy = 0.983
Training Accuracy = 0.988
Testing Accuracy = 0.865

EPOCH 15 ...
Validation Accuracy = 0.980
Training Accuracy = 0.987
Testing Accuracy = 0.875

EPOCH 16 ...
Validation Accuracy = 0.982
Training Accuracy = 0.988
Testing Accuracy = 0.872

EPOCH 17 ...
Validation Accuracy = 0.986
Training Accuracy = 0.991
Testing Accuracy = 0.885

EPOCH 18 ...
Validation Accuracy = 0.984
Training Accuracy = 0.989
Testing Accuracy = 0.876

EPOCH 19 ...
Validation Accuracy = 0.987
Training Accuracy = 0.992
Testing Accuracy = 0.882

EPOCH 20 ...
Validation Accuracy = 0.986
Training Accuracy = 0.992
Testing Accuracy = 0.876

EPOCH 21 ...
Validation Accuracy = 0.987
Training Accuracy = 0.992
Testing Accuracy = 0.886

EPOCH 22 ...
Validation Accuracy = 0.986
Training Accuracy = 0.991
Testing Accuracy = 0.885

EPOCH 23 ...
Validation Accuracy = 0.988
Training Accuracy = 0.993
Testing Accuracy = 0.885

EPOCH 24 ...
Validation Accuracy = 0.990
Training Accuracy = 0.994
Testing Accuracy = 0.889

EPOCH 25 ...
Validation Accuracy = 0.989
Training Accuracy = 0.994
Testing Accuracy = 0.883

EPOCH 26 ...
Validation Accuracy = 0.986
Training Accuracy = 0.992
Testing Accuracy = 0.889

EPOCH 27 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.899

EPOCH 28 ...
Validation Accuracy = 0.989
Training Accuracy = 0.994
Testing Accuracy = 0.897

EPOCH 29 ...
Validation Accuracy = 0.991
Training Accuracy = 0.995
Testing Accuracy = 0.891

EPOCH 30 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.898

EPOCH 31 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.896

EPOCH 32 ...
Validation Accuracy = 0.991
Training Accuracy = 0.995
Testing Accuracy = 0.894

EPOCH 33 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.890

EPOCH 34 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.897

EPOCH 35 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.896

EPOCH 36 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.896

EPOCH 37 ...
Validation Accuracy = 0.993
Training Accuracy = 0.997
Testing Accuracy = 0.904

EPOCH 38 ...
Validation Accuracy = 0.991
Training Accuracy = 0.996
Testing Accuracy = 0.894

EPOCH 39 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.896

EPOCH 40 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.907

EPOCH 41 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.909

EPOCH 42 ...
Validation Accuracy = 0.991
Training Accuracy = 0.995
Testing Accuracy = 0.881

EPOCH 43 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.905

EPOCH 44 ...
Validation Accuracy = 0.993
Training Accuracy = 0.997
Testing Accuracy = 0.908

EPOCH 45 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.907

EPOCH 46 ...
Validation Accuracy = 0.994
Training Accuracy = 0.997
Testing Accuracy = 0.908

EPOCH 47 ...
Validation Accuracy = 0.993
Training Accuracy = 0.996
Testing Accuracy = 0.905

EPOCH 48 ...
Validation Accuracy = 0.993
Training Accuracy = 0.997
Testing Accuracy = 0.902

EPOCH 49 ...
Validation Accuracy = 0.993
Training Accuracy = 0.998
Testing Accuracy = 0.898

EPOCH 50 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.902

EPOCH 51 ...
Validation Accuracy = 0.993
Training Accuracy = 0.997
Testing Accuracy = 0.900

EPOCH 52 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.913

EPOCH 53 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.905

EPOCH 54 ...
Validation Accuracy = 0.993
Training Accuracy = 0.998
Testing Accuracy = 0.906

EPOCH 55 ...
Validation Accuracy = 0.992
Training Accuracy = 0.996
Testing Accuracy = 0.907

EPOCH 56 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.904

EPOCH 57 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.912

EPOCH 58 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.909

EPOCH 59 ...
Validation Accuracy = 0.995
Training Accuracy = 0.999
Testing Accuracy = 0.912

EPOCH 60 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.910

EPOCH 61 ...
Validation Accuracy = 0.995
Training Accuracy = 0.998
Testing Accuracy = 0.910

EPOCH 62 ...
Validation Accuracy = 0.995
Training Accuracy = 0.998
Testing Accuracy = 0.914

EPOCH 63 ...
Validation Accuracy = 0.993
Training Accuracy = 0.998
Testing Accuracy = 0.908

EPOCH 64 ...
Validation Accuracy = 0.994
Training Accuracy = 0.998
Testing Accuracy = 0.905

Model saved

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][sample_images/bumpy_road.jfif]
![alt text][sample_images/no_entry.jfif]
![alt text][sample_images/Right_of_way_at_next_intersection.jpg]
![alt text][sample_images/road_work.jfif]
![alt text][sample_images/speed_limit_70.jpg]

The first image might be difficult to classify because sign is on a pole with trees on the back groun.

Second image might be difficult to classify because there is threre is traffic in the back ground. We can see bus and trees in background at the corners.

Third image has clouds in back ground it might be difficult to classify as clouds have different colors light and dark in the back ground.

Fourth image also has clouds and blue sky in the back ground. 

Fifth image has trees and clounds in the back ground so it might be challenging to classify as we shold know exact boundry of the traffic sign to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| No Entry     			| No Entry 										|
| Right of the was at 	| Right of the was at next intersection			|
next intersection
| Road  work	     	| Road  work					 				|
| Speed limit 70		| Speed limit 70     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

For the first image, the model is sure that this is a right of the way at intersection sign (probability of 99.97%), and the image does contain a stop sign. The top five soft max probabilities were

| Probability (%)         	|     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 99.97         		| Right of the way at intersection				| 
| .03     				| Beware of ice/snow							|
| .00					| Double curve									|
| .00	      			| Road work 					 				|
| .00				    | Priority road      							|


For the second image the model is pretty sure that this is bumpy road image with 100% probability.

| Probability (%)         	|     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| Bumpy road                    				| 
| .00     				| Bicycles crossing 							|
| .00					| Wild animals crossing							|
| .00	      			| Children crossing				 				|
| .00				    | General caution      							|

For the third image model correctly identifed no entry image with 100% probability.

| Probability (%)         	|     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| No entry                      				| 
| .00     				| No passing        							|
| .00					| Stop      									|
| .00	      			| Turn left ahead				 				|
| .00				    | Roundabout mandatory 							|

For fourth image model detected road work with 83.99% probabilty. The other possible sign was wild animal corssing with 13.99% probablity.

| Probability (%)         	|     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 83.99         		| Road work                     				| 
| 13.99     			| Wild animals crossing							|
| 1.35					| Slippery road									|
| .29	      			| Bumpy road 					 				|
| .12				    | Keep right         							|

For fifth image model was sure this is speed limit 70 sign with 100% probablity.

| Probability (%)         	|     Prediction	       					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00         		| Speed limit (70km/h)          				| 
| .00     				| Speed limit (30km/h)							|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (80km/h) 			 				|
| .00				    | Keep left         							|

Obersvartion:
Probabilties of are decreases as number of epochs(iterations) are increased in the model. Also the success rate of correctly detecting the image increased in comparison with low iterations. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


