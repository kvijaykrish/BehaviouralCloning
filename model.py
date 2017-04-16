import csv
import cv2
import numpy as np
import sklearn

samples = []
# Open the driveing log csv file and read the samples
with open('./data/driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

#Use 80% data set for training set and 20% for validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

images = []
measurements = []
print (len(samples))

#############################################
# Augument the images: Horizontal Translation
# Apply random horizontal translation to simulate car at various positions in the road
# For each pixel translation apply corresponding steering angle shift
#############################################
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320,160))
    return image_tr,steer_ang

#############################################
# Preprocessing: Apply random brightness
#############################################
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# Steering Angle correction factor for center, Left and Right Camera
correction = [0.0,0.25,-0.25]

# Generator: to generate data for training rather than storing the training data in memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # For each center, left and right camera images:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    # Read center camera steering angle
                    c_angle = float(batch_sample[3])
                    # Apply steering angle correction for center, left and right camera
                    angle = c_angle+correction[i]

                    ##########################################
                    # Augument data set: Use translated images
                    ##########################################
                    # The data set has lots of images with almost 0 steering angle
                    # Replace the images with almost 0 steering anlge with random translated images
                    # Use probability of 0.5 to retain original image or transalted image
                    if abs(c_angle) < 0.01:
                        skip0angle = np.random.randint(2)
                        if skip0angle==0:
                            image, angle = trans_image(image, angle, 100)
                    image = augment_brightness_camera_images(image)
                    images.append(image)
                    angles.append(angle)
                    
                    ##########################################
                    # Augument data set: Use flipped images
                    ##########################################
                    image = cv2.flip(image,1)
                    image = augment_brightness_camera_images(image)
                    images.append(image)
                    angles.append(angle*-1.0)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

#############################################
# Model Architecture: Lenet Architecture
# 1. Normalization Layer: 160, 320, 3
# 2. Cropping Layer     : 65, 320, 3 
# 3. Convolution2D Layer: 61, 316, 6
# 4. MaxPooling2D Layer : 30, 158, 6
# 5. Convolution2D Layer: 26, 154, 6
# 6. MaxPooling2D Layer : 13, 77, 6
# 7. Flatten            : 6006
# 8. Dense              : 120
# 9. Dense              : 84
# 10. Dense             : 1
#############################################
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row,col,ch)))
# Crop of top 70 (scenary) and bottom 25 (hood) pixels: 
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# Print model architecture summary
model.summary()
# Compile and fit the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch = len(train_samples)*6,
                                     validation_data = validation_generator, 
                                     nb_val_samples =len(validation_samples)*6,
                                     nb_epoch=4, verbose=1)
# Save the trained model with weights
model.save('model.h5')
