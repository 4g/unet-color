#!/usr/bin/env python

"""U-Net network for training a colorizer(colors black and white images)"""

__author__      = "Apurva Gupta"
__copyright__   = "Ha .. Ha Ha"

from keras.layers import Convolution2D , UpSampling2D, Input, merge, BatchNormalization, Activation, Dropout
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.models import Model
from tqdm import tqdm
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import utils
import cv2
import numpy as np
from keras.models import load_model
from keras.models import save_model
import argparse

class Colorizer:
    def __init__(self):
        self.model = Colorizer.model()
        self.compile()

    def summary(self):
        print self.model.summary()

    def predict(self, X):
        return self.model.predict(X)

    def compile(self):
        self.model.compile(loss='mae', optimizer=Adam(lr=0.0002, beta_1=0.5))

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)

    def train(self, X, y, batch_size=128, epochs=10):
        for i in tqdm(range(epochs), desc='Epochs'):
            num_batches = X.shape[0] / batch_size
            batch_index = 0
            for j in tqdm(range(num_batches-1),desc="Batch", leave=False):
                batch_index += batch_size
                X_batch = X[batch_index:batch_index + batch_size]
                y_batch = y[batch_index:batch_index + batch_size]
                self.train_on_batch(X_batch, y_batch)
                yield i,j

    def train_on_batch(self, X, y):
        return self.model.train_on_batch(X,y)

    @staticmethod
    def model():
        inputs = Input((3, 64, 64))
        conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs)
        conv1 = BatchNormalization(mode=2)(conv1)
        conv1 = (LeakyReLU(0.2))(conv1)

        conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
        conv1 = BatchNormalization(mode=2)(conv1)
        conv1 = (LeakyReLU(0.2))(conv1)

        conv2 = Convolution2D(64, 3, 3, border_mode='same')(conv1)
        conv2 = BatchNormalization(mode=2)(conv2)
        conv2 = (LeakyReLU(0.2))(conv2)

        conv2 = Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same')(conv2)
        conv2 = BatchNormalization(mode=2)(conv2)
        conv2 = (LeakyReLU(0.2))(conv2)

        conv3 = Convolution2D(128, 3, 3, border_mode='same')(conv2)
        conv3 = BatchNormalization(mode=2)(conv3)
        conv3 = (LeakyReLU(0.2))(conv3)

        conv3 = Convolution2D(128, 3, 3, subsample=(2,2),border_mode='same')(conv3)
        conv3 = BatchNormalization(mode=2)(conv3)
        conv3 = (LeakyReLU(0.2))(conv3)

        conv4 = Convolution2D(256, 3, 3, border_mode='same')(conv3)
        conv4 = BatchNormalization(mode=2)(conv4)
        conv4 = (LeakyReLU(0.2))(conv4)

        conv4 = Convolution2D(256, 3, 3, subsample=(2,2), border_mode='same')(conv4)
        conv4 = BatchNormalization(mode=2)(conv4)
        conv4 = (LeakyReLU(0.2))(conv4)

        up7 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(128, 3, 3, border_mode='same')(up7)
        conv7 = BatchNormalization(mode=2)(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Dropout(0.2)(conv7)

        conv7 = Convolution2D(128, 3, 3, border_mode='same')(conv7)
        conv7 = BatchNormalization(mode=2)(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Dropout(0.2)(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(64, 3, 3, border_mode='same')(up8)
        conv8 = BatchNormalization(mode=2)(conv8)
        conv8 = Activation('relu')(conv8)

        conv8 = Convolution2D(64, 3, 3, border_mode='same')(conv8)
        conv8 = BatchNormalization(mode=2)(conv8)
        conv8 = Activation('relu')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(32, 3, 3, border_mode='same')(up9)
        conv9 = BatchNormalization(mode=2)(conv9)
        conv9 = Activation('relu')(conv9)

        conv9 = Convolution2D(32, 3, 3, border_mode='same')(conv9)
        conv9 = BatchNormalization(mode=2)(conv9)
        conv9 = Activation('relu')(conv9)

        conv10 = Convolution2D(3, 1, 1, activation='tanh')(conv9)

        model = Model(input=inputs, output=conv10)

        return model

def main(args):
    colorizer = Colorizer()

    X = utils.data(size=(64,64), count=5000, path=args.data_path)

    np.random.shuffle(X)
    y = np.copy(X)
    X = np.expand_dims(np.mean(X, axis=1), axis=1)
    X = np.repeat(X, repeats=3, axis=1)
    print X.shape

    for i,j in colorizer.train(X, y, epochs=50, batch_size=32):
        sample = X[:100]
        osample = y[:100]
        predicted = colorizer.predict(sample)
        img1 = utils.arrange_images(sample)
        img2 = utils.arrange_images(predicted)
        img3 = utils.arrange_images(osample)
        cv2.imshow('f1',img1)
        cv2.imshow('f2',img2)
        cv2.imshow('f3', img3)
        cv2.waitKey(10)

    colorizer.save(args.model_path)
    colorizer.load(args.mode_path)

    X = utils.data(size=(64,64), count=8000, path=args.data_path)

    X = np.expand_dims(np.mean(X, axis=1), axis=1)
    X = np.repeat(X, repeats=3, axis=1)
    print X.shape

    for i in range(100):
        sample = X[i*100:(i+1)*100]
        predicted = colorizer.predict(sample)
        img1 = utils.arrange_images(sample)
        img2 = utils.arrange_images(predicted)
        cv2.imshow('f1',img1)
        cv2.imshow('f2',img2)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args)
