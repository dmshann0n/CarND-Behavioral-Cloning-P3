# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_arch.png "Nvidia Architecture"
[image2]: ./examples/center.jpg "Center driving"
[image3]: ./examples/recovery_1.jpg "Recovery Image"
[image4]: ./examples/recovery_2.jpg "Recovery Image"
[image5]: ./examples/recovery_3.jpg "Recovery Image"
[image6]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md -- this document!

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 

I encapsulated the CSV file in the DrivingEntry class (driving_entry.py) to make it clearer how that data was being used in augmented data downstream (rather than relying on the indices from the CSV row). Augmentations are implemented as simple strategies that return a tuple containing the image, which has been potentialy modified, and a steering angle. These permutations are used in the training and validation generators.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The data is normalized in the model using a Keras lambda layer (code lines 65-66). I followed the network from the Nvidia autonomous vehicle architecture.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving in reverse.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed a similar path to the lesson. I started with a very simple network to prove out the integration between my model, model.h5, sample data loading, and driving the vehicle autonomously using the model.

Afterward, I switched immediately from this simple model to the Nvidia architecture, skipping experimentation with other existing architectures. Initial training data provided good results on the first track, but the model was unable to complete a lap.

After some learning curve on using the simulator and properly recording, I reset my data and concentrated more on collecting _better_ data for smooth steering and correction. With this new data the car was able to make a complete lap.

#### 2. Final Model Architecture

The final model architecture matches the Nvidia architecture presented in the lesson:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the edges. The following images show a recovery starting from the right side (near the striped barrier) and correcting bcack to the center.

![alt text][image3]

![alt text][image4]

![alt text][image5]

After the collection process, I had 7,570 samples. I augmented this data with a horizontally flipped version, and the left and right camera perspectives for a total of 30,280. 

The Keras model normalizes this data by cropping the image and normalizing pixels to ranges of 0 through 1.

The generator that returns data to the Keras model first shuffles the data before generating permutations for each augmentation. The augmentations are evenly represented in the generated data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The model was able to complete a loop with only 3 epochs, averaging 1297 seconds (21 minutes) per epoch on my 13" MacBook Pro. I did not experiment with an optimizer, starting with and sticking with an Adam optimizer.
