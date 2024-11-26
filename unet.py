import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

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

# 5. Define the modified UNet model for classification
def unet_classification_model(input_size=(224, 224, 3), num_classes=4):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Global Average Pooling and Dense Output Layer for Classification
    gpool = GlobalAveragePooling2D()(c4)
    outputs = Dense(num_classes, activation='softmax')(gpool)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
# Instantiate and compile the model
model = unet_classification_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 6. Set the steps_per_epoch and validation_steps
steps_per_epoch = len(train_generator)
validation_steps = len(test_generator)

hist = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping]
)
# 9. Evaluate the model on the test data
evaluation = model.evaluate(test_generator)
print(f'Test Loss: {evaluation[0]}')
print(f'Test Accuracy: {evaluation[1]}')