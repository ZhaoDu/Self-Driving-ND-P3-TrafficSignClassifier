# Project-03: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview 
In this project, a convolutional neural networks was built to classify traffic signs. [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) was adopted to train and validate this model.

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results

## Step 0: Load the Data Set
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (31,31,3)
* The number of unique classes/labels in the data set is 43

## Step 1: Explore, Summarize and Visualize the Data Set
Here is an exploratory visualization of the data set. It is a bar chart showing how the traffic sign data distribued.

![training data][image1]

The average number of training examples per class is 809, the minimum is 180 and the maximum 2010, hence some labels are one order of magnitude more abundant than others.

Most common signs:

- Speed limit (50km/h) train samples: 2010
- Speed limit (30km/h) train samples: 1980
- Yield train samples: 1920
- Priority road train samples: 1890
- Keep right train samples: 1860

Here is an visualization of some 8 randomly picked training examples for each class. As can be found, within each class there is a high variability in appearance due to different weather conditions, time of the day and image angle.
![training img][image2]

## Step2: Design, train and test a model architecture
### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:


The difference between the original data set and the augmented data set is the following ... 


### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers:

| LAYER         		|     DESCRIPTION	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected		| inputs 400, outputs: 120                      |
| RELU					|												|
| Fully connected		| inputs 120, outputs: 84                       |
| RELU					|												|
| Fully connected		| inputs 84, outputs: 43                        |
 


### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tuned the epochs, batch_size and learning rate. Traning accuracy under different combinations of above parameters can be found in floowing table.

|TIME      |ACCURACY  |EPOCHS    |BATCH_SIZE|LEARNING RATE|Time         |
|:--------:|:---------|----------|----------|-------------|------------:|
|2019-12-7 |92.7%     |100       |128       |0.001        |223.922      |
|2019-12-8 |92.4%     |100       |64        |0.001        |357.654      |
|2019-12-8 |87.3%     |100       |128       |0.0001       |222.185      |
|2019-12-8 |93.4%     |200       |128       |0.001        |449.605      |

Training error:
![alt text][image3]
Training loss:
![alt text][image4]

### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of 0.92?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

## Step3: Use the model to make predictions on new images
### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

 ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...


## Step4: Analyze the softmax probabilities of the new images

### Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
## Step5: Summarize the results

## (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
### Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

## Discussion 


[//]: # (Image References)

[image1]: ./img/training_set_counts.png "Visualization"
[image2]: ./img/random_examples.png "Visualization_imgs"
[image3]: ./img/learning_curve_error.png "Error"
[image4]: ./img/learning_curve_loss.png "Loss"

[image5]: ./img/placeholder.png "Traffic Sign 2"
[image6]: ./img/placeholder.png "Traffic Sign 3"
[image7]: ./img/placeholder.png "Traffic Sign 4"
[image8]: ./img/placeholder.png "Traffic Sign 5"
img\