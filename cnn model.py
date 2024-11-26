import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
from keras.layers import *
from keras.models import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import (Model)

# 3. Set up data generators for training and testing
train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set directories for train and test datasets
train_dir =  '.\Training'
test_dir = '.\Testing'

# 4. Load data from the directories
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(224, 224),
batch_size=32,
class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(224, 224),
batch_size=32,
class_mode='categorical'
)

# 5. Build the Convolutional Neural Network (CNN) model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layers
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))

# Output layer with softmax for classification
model.add(Dense(4, activation='softmax')) # 4 categories: glioma, meningioma, notumor, pituitary


# 6. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# 6. Set the steps_per_epoch and validation_steps
steps_per_epoch = len(train_generator)
validation_steps = len(test_generator)



# 9. Evaluate the model on the test data
evaluation = model.evaluate(test_generator)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')