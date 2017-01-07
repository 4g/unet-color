import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
from keras.datasets import cifar100

def load_images(path,size,count):
    images = []
    for fname in tqdm(os.listdir(path)[:count]):
        try:
            fname = os.path.join(path, fname)
            image = cv2.imread(fname)
            image = cv2.resize(image,size, interpolation=cv2.INTER_LINEAR)
            image = rec(image)
            images.append(image)
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            pass

    return (np.asarray(images, dtype=np.float32) - 127.5) / 127.5

def rec(img):
    return np.moveaxis(img, 2, 0)

def flickr_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/flickr_data/Flickr8k_Dataset/Flicker8k_Dataset/",size=size,count=count)

def laptop_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/brimages/",size=size, count=count)

def dogs_cats_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/dogs_cats/train/",size=size, count=count)

def face_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/img_align_celeba/", size=size, count=count)

def xkcd_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/imgs.xkcd.com/comics/", size=size, count=count)

def cifar_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/cifar/test/", size=size, count=count)

def indoor_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/indoor_flattened/", size=size, count=count)

def cifar100_data(size, count=None):
    return (cifar100.load_data()[0][0][:count] - 127.5) / 127.5

def flowers_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/flowers/", size=size, count=count)

def cars_data(size, count=None):
    return load_images(path="/home/apurva/Desktop/deep_learning/data/cars/", size=size, count=count)

def get_mixed_data(size, count=100, sources=None):
    if sources is None:
        sources = [cifar_data, xkcd_data, face_data, dogs_cats_data, flickr_data, laptop_data, indoor_data, flowers_data, cars_data]

    data = None
    for source in sources:
        sample = source(size, count)
        if data is None:
            data = sample
        else:
            data = np.concatenate((data, sample))
    return data

def arrange_images(Y):
    concat_image = None
    Y = (Y + 1)/2
    for yi in np.split(Y, 10):
        image = None
        for y in yi:
            img = cv2.merge((y[0, :, :], y[1, :, :], y[2, :, :]))
            if image is None:
                image = img
            else:
                image = np.concatenate((image, img))
        if concat_image is None:
            concat_image = image
        else:
            concat_image = np.concatenate((concat_image, image), axis=1)
    return concat_image