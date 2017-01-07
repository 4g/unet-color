import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.layers import Convolution2D , Dense, UpSampling2D, InputLayer, Input, Merge, Flatten, BatchNormalization,  merge, MaxPooling2D, Dropout
from keras.models import Sequential
from tqdm import tqdm
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import utils
import cv2
import numpy as np
from keras.models import load_model, Model
from keras.models import save_model
from keras.layers import Activation

class Colorizer:
    def __init__(self):
        self.model = Colorizer._model()
        self.adversary = Colorizer._adversary()
        self.combined()
        self.compile()

    def summary(self):
        print self.model.summary()

    def predict(self, X):
        return self.model.predict(X)

    def compile(self):
        loss = ['mae', 'binary_crossentropy']
        loss_weights = [1E2, 1]

        self.model.compile(loss='mae', optimizer=Adam(lr=.0002, beta_1=0.5))

        self.adversary.trainable = False

        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        self.adversary.trainable = True

        self.adversary.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    def combined(self):
        model = Sequential()
        model.add(self.model)
        self.adversary.trainable = False
        model.add(self.adversary)
        self.gan = model

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)

    def train(self, X, y, batch_size=128, epochs=10):
        # self.model.fit(X, X, batch_size=128,validation_split=0.1)
        for i in range(epochs):
            num_batches = X.shape[0] / batch_size
            batch_index = 0
            for j in range(num_batches):
                batch_index += batch_size
                X_batch = X[batch_index:batch_index + batch_size]
                y_batch = y[batch_index:batch_index + batch_size]
                print i , j, self.train_on_batch(X_batch, y_batch)
                yield i,j

    def train_on_batch(self, X, y):
        # self.model.train_on_batch(X,y)
        batch_size = len(X)
        generated = self.model.predict(X)
        real_fake = np.concatenate((y, generated))
        real_labels = [1] * batch_size
        fake_labels = [0] * batch_size
        self.adversary.trainable = True
        d_loss = self.adversary.train_on_batch(real_fake, real_labels + fake_labels)
        self.adversary.trainable = False
        gan_loss = self.gan.train_on_batch(X, real_labels)
        self.adversary.trainable = True
        return gan_loss, d_loss

    @staticmethod
    def _adversary():
        model = Sequential()
        model.add(Convolution2D(
                            64, 5, 5,
                            border_mode='same',
                            input_shape=(3, 32, 32),subsample=(2,2)))
        model.add(LeakyReLU(0.2))
        model.add(Convolution2D(128, 5, 5,subsample=(2,2)))
        model.add(BatchNormalization(mode=2))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(LeakyReLU())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    @staticmethod
    def _model():
        inputs = Input((3, 32, 32))
        conv1 = Convolution2D(32, 3, 3, border_mode='same')(inputs)
        conv1 = BatchNormalization(mode=2)(conv1)
        conv1 = Activation(LeakyReLU(0.2))(conv1)

        conv1 = Convolution2D(32, 3, 3, border_mode='same')(conv1)
        conv1 = BatchNormalization(mode=2)(conv1)
        conv1 = Activation(LeakyReLU(0.2))(conv1)

        conv2 = Convolution2D(64, 3, 3, border_mode='same')(conv1)
        conv2 = BatchNormalization(mode=2)(conv2)
        conv2 = Activation(LeakyReLU(0.2))(conv2)

        conv2 = Convolution2D(64, 3, 3, subsample=(2,2), border_mode='same')(conv2)
        conv2 = BatchNormalization(mode=2)(conv2)
        conv2 = Activation(LeakyReLU(0.2))(conv2)

        conv3 = Convolution2D(128, 3, 3, border_mode='same')(conv2)
        conv3 = BatchNormalization(mode=2)(conv3)
        conv3 = Activation(LeakyReLU(0.2))(conv3)

        conv3 = Convolution2D(128, 3, 3, subsample=(2,2),border_mode='same')(conv3)
        conv3 = BatchNormalization(mode=2)(conv3)
        conv3 = Activation(LeakyReLU(0.2))(conv3)

        conv4 = Convolution2D(256, 3, 3, border_mode='same')(conv3)
        conv4 = BatchNormalization(mode=2)(conv4)
        conv4 = Activation(LeakyReLU(0.2))(conv4)

        conv4 = Convolution2D(256, 3, 3, subsample=(2,2), border_mode='same')(conv4)
        conv4 = BatchNormalization(mode=2)(conv4)
        conv4 = Activation(LeakyReLU(0.2))(conv4)

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
        print model.summary()
        return model

        # return model

colorizer = Colorizer()
X = utils.cifar100_data(size=(32,32), count=100000)
np.random.shuffle(X)
y = np.copy(X)
X = np.expand_dims(np.mean(X, axis=1), axis=1)
X = np.repeat(X, repeats=3, axis=1)
print X.shape

for i,j in colorizer.train(X, y, epochs=100, batch_size=128):
    sample = X[:100]
    predicted = colorizer.predict(sample)
    # predicted = np.mean((predicted, sample), axis=0)
    img1 = utils.arrange_images(sample)
    img2 = utils.arrange_images(predicted)
    cv2.imshow('f1',img1)
    cv2.imshow('f2',img2)
    cv2.waitKey(10)

# # z = [x for x in colorizer.train(X, y, epochs=20)]
# #
# colorizer.save("colorgan")
# colorizer.load("colorgan")

colorizer.summary()

X = utils.dogs_cats_data(size=(32,32), count=10000)
X = np.expand_dims(np.mean(X, axis=1), axis=1)
X = np.repeat(X, repeats=3, axis=1)
print X.shape

for i in range(100):
    sample = X[i*100:(i+1)*100]
    predicted = colorizer.predict(sample)
    predicted = np.mean((predicted, sample),axis=0)
    img1 = utils.arrange_images(sample)
    img2 = utils.arrange_images(predicted)
    cv2.imshow('f1',img1)
    cv2.imshow('f2',img2)
    cv2.waitKey(0)


