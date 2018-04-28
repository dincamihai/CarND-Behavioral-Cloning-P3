# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/track1-center.gif "Center driving"
[image3]: ./examples/track1-recovery.gif "Recovery driving"
[image4]: ./examples/track1-center-reverse.gif "Reverse lap track1"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5-18-04-26-10-32-00-0.9
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of:

 - input: 3x160x320
 
 - a pre-processing section: [model.py:L31-L37](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L31-L37)
   - remove sky and car front from the image
   - normalize the images values to mean 0
   - use maxpooling to reduce the model size (this reduces the number of model weights and speeds up the training)
   - output: 3x21x106
   
 - a convolution neural network: [model.py:L39-L53](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L39-L53)
   - two layers:
     - 36x5x5 strides 2x2 + max pooling 2x2 + relu -> 36x4x25
     - 64x3x3 strides 1x1 + max pooling 2x2 + relu -> 64x1x11
   - I use batch normalization after each layer to help the optimizer (same effect as normalizing in the pre-processing) and 0.5 dropout to reduce overfitting

- a fully connected section: [model.py:L57-L64](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L57-L64)
  - two layers (50 and 20 nodes)
  - I use L2 normalization to reduce overfitting with lambda 0.01
  - I use relu as activation function

- one output

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, the model contains dropout layers in the convolutional section
  - [model.py:L44](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L44)
  - [model.py:L52](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L52)
  
and L2 regularization in the fully connected network
  - [model.py:L57](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L57)
  - [model.py:L60](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L60)
  

The model was trained and validated on different data sets to ensure that the model was not overfitting [model.py:L148](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L148). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually [model.py:L70](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L70).

I tuned the dropout and the L2 regularization but also the correction applied to left and right images: [model.py:L75](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L75)

I've also treated the training data as a parameter and tuned it in order to get the best performance. [model.py:L95-L118](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L95-L118)

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road and driving the track in reverse.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was:

 - implement a simple model to make sure the training works and to establish the workflow
 - implement an architecture that already proved good results (eg: lenet or nvidia)
 - tune and collect the right data to make it drive on track1
 - check it is not completely bad on track2
 - reduce the model but retain the performance
 - collect data to drive on track2
 - bonus:
   - make it drive on both tracks

My first step was to use a convolution neural network model similar to the nvidia model described in this paper: [self-driven car - nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

I chose this model since it was implemented exactly for this purpose, to be used on a self-driven car.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 160, 320, 3)   0                                            
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           input_1[0][0]                    
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 21, 106, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 9, 51, 36)     2736        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 4, 25, 36)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 4, 25, 36)     0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 4, 25, 36)     144         activation_1[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4, 25, 36)     0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 2, 23, 64)     20800       dropout_1[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 1, 11, 64)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 1, 11, 64)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 1, 11, 64)     256         activation_2[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1, 11, 64)     0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 704)           0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 50)            35250       flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 50)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 20)            1020        activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 20)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             21          activation_4[0][0]               
====================================================================================================
Total params: 60,227
Trainable params: 60,027
Non-trainable params: 200
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

I started by recording track1 at full speed without taking care to drive on the center. Thin didn't give good results.
The next step was to record another lap focusing on center driving. This didn't give good results in curves and difficult sections (unclear road edge).
I proceeded with recording another center driving lap, driving with slow speed and very slow speed on curves and difficult sections (bridge, unclear edges).
This yelded good results but still had problems.

![alt text][image2]

I then recorded a lap driving from the left of the road to center and then to the right of the road and back.
After recording this lap I removed the images going from center to edges and kept only the images going from edge to center.
I was interested in teaching the network how to recover from edge and go back to center. Going from center to edge wouldn't be something useful.

![alt text][image3]

In order to get more variation to the data I have also recorded a lap driving on track1 counter-clock wise:

![alt text][image4]

I also augmented the dataset by appending the images captured with the left and right cameras, using a correction for the recorded angles: [model.py:L130-L145](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L130-L145)
In addition to this, I also horizontally flipped all the images obtained in the previous step: [model.py:L169-L170](https://github.com/dincamihai/CarND-Behavioral-Cloning-P3/blob/master/model.py#L169-L170)

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
