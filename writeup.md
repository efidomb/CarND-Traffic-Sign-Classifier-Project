
# **Traffic Sign Recognition** 

## Writeup

---

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

[//]: # (Image References)
[image1]: ./examples/Comparing_relative_frequency_between_data_sets.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./keeprightsign.jpg "Traffic Sign 1"
[image5]: ./noentrysign.jpg "Traffic Sign 2"
[image6]: ./roadworksign.jpg "Traffic Sign 3"
[image7]: ./speedlimit30sign.jpg "Traffic Sign 4"
[image8]: ./stopsign.jpg "Traffic Sign 5"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted

#### 1. Submission Files: The project submission includes all required files.

all of the required files are accesble from [here](https://github.com/efidomb/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Dataset Exploration

#### Dataset Summary: The submission includes a basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of test set is 12,630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory Visualization: The submission includes an exploratory visualization on the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data classes is split between the training set, test set and validation set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing: The submission describes the preprocessing techniques used and why these techniques were chosen.

As a first step, I decided to convert the images to grayscale because most of the information in the image is still preserved, but it reduce the amount of computation and memory needed for training.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalize the image data because it prevent the weights from exploding (and therefore not trainable).

#### 2. Model Architecture: The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

I used the *LeNet architecture* and my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 	|
| RELU					| activation of the convolution layer			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 		     		|
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16	|
| RELU					| activation of the last convolution layer		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 		     		|
| Flatten   	      	| 400 neurons               		     		|
| Fully connected		| 400 neurons to 120 neurons					|
| RELU					| activation of the fully connected layer		|
| Fully connected		| 120 neurons to 84 neurons			    		|
| RELU					| activation of the last fully connected layer	|
| Fully connected		| 84 neurons to 43 neurons			    		|
 


#### 3. Model Training: The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

To train the model, I used the following parameters:

- optimizer = AdamOptimizer
- batch size = 128
- number of epochs = 30
- learning rate = 0.001
- mu = 0
- sigma = 0.1


#### 4. Solution Approach: The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

My final model results were:
* training set accuracy of 1.0.
* validation set accuracy of 0.9451.
* test set accuracy of 0.9303.

about the procedure:
* I chose the *LeNet* architecture.
* the MNist and German traffic sign datasets here are a bit similar in some ways. they both around 30x30 pixels, with grayscale and not-too-complex shapes for classes. so I thought is a good point to start.
* The accuracy of the train model (1.0) is very high, so it means that the model is very good at classifying itself, but not if it can generalize to data he hasn't see before.
* the test score (0.9303) keeps getting better each epoch, but it slows down so match that it seems that more epochs will not improve the accuracy significantly more than that.
* The accuracy of the validation set is 0.9451. for this project, we consider this score as a success.


### Test a Model on New Images

#### 1. Acquiring New Images: The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

- The first and the second images might be difficult to classify because the signs are turned sideways.
- the third image might be difficult because he has a lot of areas that are irrelevant to sign's class.
- the fourth image might be difficult because he is very similar to other speed limit signs that are different only in the first number in the speed limit sign.
- the fifth image is pretty straightforward, and he is the last image that I'll think that the NN will misclassify.

#### 2. Performance on New Images: The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop          		| Stop   	    								| 
| Road work    			| Road work 									|
| No entry				| No entry										|
| Speed limit (30km/h)	| Speed limit (50km/h)			 				|
| Keep right			| Keep right        							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is less than the accuracy of the test and validation sets, but this is probably because this web-downloaded set has only 5 images, so every misclassification has an effect of 20 percent on the underline score.

#### 3. Model Certainty - Softmax Probabilities: The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all of the images, the model was very sure about their classes, with a score of (nearly) 1.0 for the class chosen, and (nearly) 0.0 for the other classes. the only exception is the speed limit (30km/h) sign that the second option is a bit more than the others, (but still extremely low), which makes sence since this image was the only image that was misclassified (although the right class appears only as the fourh option).
