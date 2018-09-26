# **Traffic Sign Recognition** 

## Writeup
## Author: Pengmian Yan


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

[image1]: ./result/data_distribution.png "Visualization"
[image2]: ./result/img_example_compare.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/3.jpg "Traffic Sign 1"
[image5]: ./test_images/4.jpg "Traffic Sign 2"
[image6]: ./test_images/5.jpg "Traffic Sign 3"
[image7]: ./test_images/1.jpg "Traffic Sign 4"
[image8]: ./test_images/2.jpg "Traffic Sign 5"
[image9]: ./result/Top5_posibilities.png "Top5_posibilities"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because my experimental results showed that classification with grayscale images resulted in higher accuracy classification than with RGB images. This can also be confirmed by Hieu Minh Bui across the different types of classifiers,  in his paper "Using grayscale images for object recognition with convolutional-recursive neural network". 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to make all the inputs are at a comparable range and to increase the stability of a neural network. 
 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

 
| Layer         		|     Description	        					| 	
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
|modified Input         | 32x32x1 grayscaled and normalized image       |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 5x5x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 		    		|
| Flatten	          	|  outputs 1600 		                 		|
| Fully connected		| outputs  480    								|
| Fully connected		| outputs 120   								|
| Fully connected		| outputs 43   									|
| Softmax				| outputs 43        							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a mini batch in tensorflow with a batch size of 120 images. I found 18 epochs can get the best fitting performance. Learning rate was chosed with 0.0001 to get a better converce. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.936
* test set accuracy of 0.927
 
At first, I started with a The LeNet-5 archtecture from Yann Lucun, which has two convolution-layers and two fully connected layers. LeNet-5 is a convolutional network designed for handwritten and machine-printed character recognition. So it should also be working on traffic sign. To improve the imput quality, I grayscaled and normalized the imput images. This architecture get a accuracy of about 0.88.

To increase the accuracy I added one convolution layer and one fully connected layer to handle the big data set. As the fully connected layers have many parameters, I added for each fully connected layer a dropout layer to avoid over-fitting. I tued the hyperparameters like the learning rate, epochs and batch size to get the best prediction accuracy. In the end the accuracy of validation set reached 0.936.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because of the high lightness deviance. This picture was token against the sun.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  						| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| No passing					| No passing											|
| Beware of ice/snow	      		| Children crossing					 				|
| Stop			| Stop      							|


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.927.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th code cell of the Ipython notebook.

The top 5 softmax probabilities for each image along with the sign type of each probability were visualizied in folling lateral bar chart.

![alt text][image9]

For the first two images, the model is relatively sure that they are a 'Speed limit (30km/h)' sign (probability of  0.998848) and a 'Right-of-way at the next intersection' sign (probability of 0.999859), and the pridictions are correct. 

For the last three images, the model has a dominant correct prediction but it remains also some small possibilities for other signs. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


