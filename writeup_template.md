#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualizations/training.png "Training Data"
[image2]: ./visualizations/valid.png "Validation Data"
[image3]: ./visualizations/test.png "Test Data"
[image4]: ./web-images/_13_1.png "web image 1"
[image5]: ./web-images/_22.png "web image 2"
[image6]: ./web-images/_0.jpg "web image 3"
[image7]: ./web-images/_25.jpg "web image 4"
[image8]: ./web-images/_3.jpg "web image 5"
[image9]: ./generated_images/13.png "generate image"
[image10]: ./generated_images/22.png "generate image"
[image11]: ./generated_images/0.png "generate image"
[image12]: ./generated_images/25.png "generate image"
[image13]: ./generated_images/3.png "generate image"
[image14]: ./visualizations/feature_maps/13.png "feature maps"
[image15]: ./visualizations/feature_maps/22.png "feature maps"
[image16]: ./visualizations/feature_maps/0.png "feature maps"
[image17]: ./visualizations/feature_maps/25.png "feature maps"
[image18]: ./visualizations/feature_maps/3.png "feature maps"



## Rubric Points
###My answer to the [rubric points](https://review.udacity.com/#!/rubrics/481/view) are below.  

---
###Writeup / README

####1. Provide a Writeup / README 

This is it! and here is a link to my [project code](https://github.com/ajaffer/traffic-signs/blob/master/my_project.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. It is a bar chart that shows the number of data points per class.

![alt text][image1]

And here is the same for validation data.

![alt text][image2]


And again for test data.

![alt text][image3]

The above images show that the data set is unbalanced.


###Design and Test a Model Architecture

####1. Preprocessing of the image data.
// What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose not to convert the images to gray-scale images since I was getting good results with color images. 

I normalized the data by making it zero mean and equal standard deviation, that will help converging quicker.


####2. Model architecture description

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten   		|  outputs 400       									|
| Fully connected		|  outputs 120       									|
| RELU					|												|
| Fully connected		|  outputs 84       									|
| RELU					|												|
| Fully connected		|  outputs 43       									|
| RELU					|												|
| Softmax				| outputs 43        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer, I used a batch size of 128, 20 epochs and learning rate of 0.003

I tried with a few learning rates, the different results are below:

| Learning Rate         		|     Validate Accuracy	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.001         		| 91%  							| 
| 0.003         		| 93.7%  							| 
| 0.009         		| 90.8%  							| 
 
 
####4.Approach taken 

My final model results were:
* training set accuracy of 99.0
* validation set accuracy of 93.7 
* test set accuracy of 90.2

* What architecture was chosen? LeNet Architecture
* Why did you believe it would be relevant to the traffic sign application? LeNet was originally designed for handwriting and machine-printed character recognition. Many traffic signs contains features that have machine-printed characters and symbols. LeNet seems like a logical choice for these kind of images.    
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Since the model's accuracy for all training, validation and test are high, we do not see any evidence of over or under fitting. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third and fourth images might be difficult to classify because the training data for these classes of images are lower in number. 

####2. Discuss the model's predictions on these new traffic signs and 
compare the results to predicting on the test set. 
At a minimum, discuss what the predictions were, 
the accuracy on these new predictions, 
and compare the accuracy to the accuracy on the test set 
(OPTIONAL: Discuss the results in more detail as described in the 
"Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Bumpy road     			| Bumpy road 										|
| 20 km/h					| General Caution											|
| Road work	      		| Go straight or left					 				|
| 60 km/h			| 60 km/h      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Description of how certain the model is when predicting on each of the five new images 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield   									| 
| 1.00     				| Bumpy road 										|
| 0.00					| 20 km/h											|
| 0.00	      			| Road work					 				|
| 0.98				    | 60 km/h      							|

Below is a visualization of the 5 softmax probabilities

Yield
![alt text][image9]
Bumpy road
![alt text][image10]
20 km/h
![alt text][image11]
Road work
![alt text][image12]
60 km/h
![alt text][image13]

### Visualizing the Neural Network 

What characteristics did the neural network use to make classifications?

The neural network has the following activations for these images:

Network correctly has activations for inverted triangular shape 

| Visualization |  Image         	|     Characteristics	        					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
|![alt text][image14]         			| Yield   									| activations on the inverted triangle
|![alt text][image15]         			| Bumpy Road   									| activations on triangle + bump symbols 
|![alt text][image16]         			| 20 km/h   									| wrong activations on triangle shape
|![alt text][image17]         			| Road work   									| wrong activations Traffic_Sign_Classifier.ipynbon triangle shape
|![alt text][image18]         			| 60 km/h   									| activations on circle + characters for number '60'

