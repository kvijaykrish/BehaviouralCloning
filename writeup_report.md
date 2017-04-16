# **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/AugImage.png "Augumented Image"
[image4]: ./examples/OriginalSampleSetBin.png "OriginalSampleSetBin"
[image5]: ./examples/SampleSetBin.png "SampleSetBin"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (not modified)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* model.html containing the ipython script to create and train the model and the results of training epoch
* video.mp4 - A video recording of my simulator vehicle driving autonomously at least one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and  drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with below architecture

Model Architecture: Lenet Architecture
1. Normalization Layer: 160, 320, 3
2. Cropping Layer     : 65, 320, 3 
3. Convolution2D Layer: 61, 316, 6 (5x5)
4. MaxPooling2D Layer : 30, 158, 6
5. Convolution2D Layer: 26, 154, 6 (5x5)
6. MaxPooling2D Layer : 13, 77, 6
7. Flatten            : 6006
8. Dense              : 120
9. Dense              : 84
10. Dense             : 1

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model was trained using augumentd data set derived from the original images in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road using horizontal translation with apropriate steering angle correction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

##### Architecture 1
My first step was to use a convolution neural network model similar to the LeNet architecture I thought this model might be appropriate because it was very good for image processing classification and regularization

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The objective was to reduce the mean squared error on the training and the validation set

I modified the model so that I introduced a normalization layer and a cropping layer.

##### Alternate Architecture 2
Then I tried out the Nvidia model architecture. However the mean square error on the training set started to increases to huge value and did not converge. 

##### Back to Architecture 1:

Hence I went back to the architecture 1 where the mean square error was reducing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially in the turnings where there where no lane markings (brown muddy sides)  to improve the driving behavior in these cases, I added more training data as described in the section "3. Creation of the Training Set & Training Process"

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Here is a detailed layers of the architecture:
1. | Layer (type)                    | Output Shape         |Param #     |Connected to                |     
2. |lambda_1 (Lambda)                |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]        |     
3. |cropping2d_1 (Cropping2D)        |(None, 65, 320, 3)    |0           |lambda_1[0][0]              |     
4. |convolution2d_1 (Convolution2D)  |(None, 61, 316, 6)    |456         |cropping2d_1[0][0]          |     
5. |maxpooling2d_1 (MaxPooling2D)    |(None, 30, 158, 6)    |0           |convolution2d_1[0][0]       |     
6. |convolution2d_2 (Convolution2D)  |(None, 26, 154, 6)    |906         |maxpooling2d_1[0][0]        |     
7. |maxpooling2d_2 (MaxPooling2D)    |(None, 13, 77, 6)     |0           |convolution2d_2[0][0]       |     
8. |flatten_1 (Flatten)              |(None, 6006)          |0           |maxpooling2d_2[0][0]        |     
9. |dense_1 (Dense)                  |(None, 120)           |720840      |flatten_1[0][0]             |     
10. |dense_2 (Dense)                  |(None, 84)            |10164       |dense_1[0][0]               |     
11. |dense_3 (Dense)                  |(None, 1)             |85          |dense_2[0][0]               |     

#### 3. Creation of the Training Set & Training Process

I used the original data set provided by Udacity. This has 8036 images. Most of the images were straight driving where the steering angle was <0.01. The number of images were counted based on the steering angle and histogram was created. This is shown below:

![alt text][image4]

Since there was more training data with straingt roads and less training data with turining bends, the training was overfitting
After Training the simulation car used to run out on the turnings. Especially the turing with no proper lane markings.

I then created more number of references to the images with abs(steering angle) > 0.15 in driving_log.csv. The histogram now looks much more distributed as shown below

![alt text][image5]


Still the training data set has more images with sterring angle < 0.01. To augment the data set, I included horizontal translation of images and also flipped images and angles. For example, here is an imageset that has then been flipped, translated for 3 images (center, right and left camera):

![alt text][image3]


If the abs(steering angle) of the original center camera image was less than 0.01 then translation was done with a probability of 0.5. This was to reatin original images as well as to simulate car entering the road from out of road conditions.

After the augumentation process, I had 12560 number of data points. In the generator I had used 6 images for each data point.
All  3 camera images (left right , center) times two (flipped and no flipped). So total is 12560 * 2 * 3 which is 75360 images. I then preprocessed this data by brightness correction, normalization, cropping. 

Batch size was 32 * 2* 3 i.e 192
No. of Epochs = 4

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by model.html I used an adam optimizer so that manually training the learning rate wasnt necessary.

The training time on normal cpu, model loss on training and validation set for each epoch is as shown below:

1. Epoch 1/4: 60288/60288 - 6178s - loss: 0.7414 - val_loss: 0.0459
2. Epoch 2/4: 60288/60288 - 6818s - loss: 0.0402 - val_loss: 0.0373
3. Epoch 3/4: 60288/60288 - 6501s - loss: 0.0351 - val_loss: 0.0342
4. Epoch 4/4: 60288/60288 - 6978s - loss: 0.0328 - val_loss: 0.0321

Finally the the model was saved.

With the trained model, the simmulator was able to run more than 1 lap in autonomous mode in track1. The video is available in video.mp4 at 60 fps.

