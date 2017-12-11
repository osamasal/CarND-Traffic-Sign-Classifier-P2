# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[image1]:  ./Images/Visualization.png "Visualization"
[image2]:  ./Images/Visualization_pie.png "Visualization pie chart"
[image3]:  ./Images/Color.png "Color image"
[image4]:  ./Images/Grayscale.png "Grayscaled image"
[image5]:  ./Images/original.png "Original image"
[image6]:  ./Images/rotated.png "Rotated image"
[image7]:  ./Dataset/keep_right.png "Traffic Sign: keep right"
[image8]:  ./Dataset/no_entry.png "Traffic Sign: no entry"
[image9]:  ./Dataset/priority_road.png "Traffic Sign: priority road"
[image10]:  ./Dataset/right_turn.png "Traffic Sign: right turn"
[image11]: ./Dataset/speed_limit_50.png "Traffic Sign: speed limit 50"
[image12]: ./Dataset/stop.png "Traffic Sign: stop"
[image13]: ./Dataset/yield.png "Traffic Sign: yield"

Here is a link to my [project code](https://github.com/osamasal/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I some statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed across the various classes.
As can be seen in this graph, some traffic signs have many training data while other have very few.

![alt text][image1]

This image shows the distribution of the signs in the input training set (i.e. bigger slices of the pie correspond to more training data).

![alt text][image2]

This image shows the same distribution only viewed as a pie chart.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this reduces the complexity (and dimensions) of the network. For example of having input of shape (32 x 32 x 3), it is now (32 x 32 x 1) when being fed through the network.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]

As a last step, I normalized the image data because I wanted to avoid having the values involved in the calculations to become too big or too small (to avoid accumulating errors). I did this by subtracting the mean value for each image from the image pixel values, then I divide each pixel value by the maximum value found in the image pixel. I did this to avoid having image pixels values being too small when most of the image values are around 128. This why I decided to avoid using the provided "(pixel - 128)/ 128" formula.

I decided to generate additional data because I wanted to allow my network to be more flexible in recognizing traffic signs which do not appear perfectly in the image (e.g. rotated images).

To add more data to the the data set, I doubled the size of the input data set by randomly introducing a rotated version of each input image to the input set. The image is randomly rotated 90, 180, or 270 degrees before being added to the input set.

Here is an example of an original image and an augmented image:

![alt text][image5]
![alt text][image6]

The difference between the original data set and the augmented data set is that the augmented data set concludes copies of input data which have been randomly rotated 90, 180, or 270 degrees.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model was largely based on the LeNet network. It included

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32 x 32 x 1 grayscale images     				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the provided mu and sigma values (mu = 0 and sigma = 0.1) as changing these values did not seem to improve the performance of my network.

I also stuck with a batch size of 128 and a learning rate of 0.001 as these values proved optimal for my network (given the input). I increased the epoch to 30 since I noticed the accuracy continues to improve beyond epoch = 20, but then it over-fits when epoch is > 30.

I also noticed that there are some fluctuations in the accuracy of the classification for epoch roughly > 10. Because of this, it was difficult to come up with a single epoch value which I could claim with confidence to be an optimal value. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **?????????????????????????????????????????????**
* validation set accuracy of 0.935
* test set accuracy of **????????????????????????????????????????????**

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| no entry     			| no entry 										|
| speed limit 50		| speed limit 50      							|
| keep right      		| keep right   									|
| priority road			| priority road									|
| stop sign	      		| stop sign			       		 				|
| right turn	   		| right turn					 				|
| yield     			| yield             							|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the seven new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 32nd cell of the Ipython notebook.

** Please note that all probabilities are given to three decimal places, so very small probabilities show a value of 0.000. ***

For the first image, the model is very sure that this is a "no entry" sign (probability of 0.988), and the image does contain a "no entry" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .988         			| no entry   									|
| .006     				| go straight or right							|
| .004					| ahead only									|
| .002	      			| stop sign 					 				|
| .002				    | road work         							|

For the second image, the model is very sure that this is a "speed limit 50" sign (probability of 0.916), and the image does contain a "speed limit 50" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .916         			| speed limit 50								|
| .045     				| speed limit 30			      				|
| .014					| speed limit 60								|
| .012	      			| speed limit 80 				 				|
| .006				    | speed limit 70       							|

For the third image, the model is relatively sure that this is a "keep right" sign (probability of 0.802), and the image does contain a "keep right" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .802         			| keep right      								|
| .188     				| keep left		        	      				|
| .008					| turn right ahead								|
| .000	      			| turn left ahead				 				|
| .000				    | road work          							|

For the forth image, the model is very sure that this is a "priority road" sign (probability of 1.000), and the image does contain a "priority road" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.000        			| priority road	     							|
| 0.000     			| roundabout mandatory     	      				|
| 0.000					| no vehicles   								|
| 0.000	      			| yield				 				            |
| 0.000				    | end of all speed and passing limits     		|

For the fifth image, the model is very sure that this is a "stop" sign (probability of 0.999), and the image does contain a "stop" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.999        			| stop    	     	    						|
| 0.000     			| speed limit 60          	      				|
| 0.000					| speed limit 20   								|
| 0.000	      			| speed limit 50					            |
| 0.000				    | speed limit 70    		                    |

For the sixth image, the model is very sure that this is a "right turn ahead" sign (probability of 0.999), and the image does contain a "right turn ahead" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.999        			| right turn ahead	     	    				|
| 0.000     			| keep right          	      				    |
| 0.000					| go straight or right   						|
| 0.000	      			| keep left					                    |
| 0.000				    | right-of-way at the next intersection         |

For the seventh image, the model is very sure that this is a "yield" sign (probability of 0.999), and the image does contain a "yield" sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.999        			| yield	     	                				|
| 0.000     			| bumpy road          	      				    |
| 0.000					| traffic signals   	       					|
| 0.000	      			| no vehicles				                    |
| 0.000				    | priority road                                 |
