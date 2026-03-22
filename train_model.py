import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Data augmentation (IMPORTANT)
datagen = ImageDataGenerator(

    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1

)

datagen.fit(x_train)

# Build CNN
model = keras.Sequential([

    layers.Input(shape=(28,28,1)),

    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(10,activation='softmax')

])

# Compile
model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

model.summary()

# Train
history = model.fit(

    datagen.flow(x_train,y_train,batch_size=32),

    epochs=8,

    validation_data=(x_test,y_test)

)

# Evaluate
test_loss,test_accuracy = model.evaluate(x_test,y_test)

print("Final Test Accuracy:",test_accuracy)

# Save model
model.save("digit_model.h5")

# Save training history for dashboard
np.save("train_acc.npy",history.history['accuracy'])

np.save("val_acc.npy",history.history['val_accuracy'])

np.save("train_loss.npy",history.history['loss'])

np.save("val_loss.npy",history.history['val_loss'])

print("Model saved successfully")