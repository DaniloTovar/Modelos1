import argparse
import pickle
import os
from loguru import logger
import numpy as np
import pandas as pd

from keras import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.api.optimizers import RMSprop
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.utils import to_categorical

def fitModel(epochs, batch_size, Xtr, Ytr):
    # Set the random seed
    random_seed = np.random.seed(2)

    # Define model validation data
    X_train, X_val, Y_train, Y_val = train_test_split(Xtr, Ytr, test_size=0.1, random_state=random_seed)

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    # Define the optimizer
    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # With data augmentation to prevent overfitting (accuracy 0.99286)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    # Fit the model
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, validation_data=(X_val, Y_val),
                        verbose=2, steps_per_epoch=((X_train.shape[0] // batch_size)),
                        validation_steps=((X_train.shape[0] // batch_size))
                        , callbacks=[learning_rate_reduction])

    return model

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True, type=str, help='a csv file with train data')
parser.add_argument('--model_file', required=True, type=str, help='where the trained model will be stored')
parser.add_argument('--overwrite_model', default=False, action='store_true', help='if set, overwrites the model file if it exists')

args = parser.parse_args()

model_file = args.model_file
data_file = args.data_file
overwrite = args.overwrite_model

if os.path.isfile(model_file):
    if overwrite:
        logger.info(f"overwriting existing model file {model_file}")
    else:
        logger.info(f"model file {model_file} exists. exitting. use --overwrite_model option")
        exit(-1)

logger.info("loading train data")
z = pd.read_csv(data_file).values
Xtr = z[:,:2]
Xtr = Xtr / 255.0
Xtr = Xtr.values.reshape(-1,28,28,1)
ytr = z[:,-1]
ytr = to_categorical(ytr, num_classes = 10)

logger.info("fitting model")
m = fitModel(5, 86, Xtr, ytr)

logger.info(f"saving model to {model_file}")
with open(model_file, "wb") as f:
    pickle.dump(m, f)