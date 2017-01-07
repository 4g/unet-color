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

def data(path, size, count=None):
    return load_images(path=path,size=size,count=count)

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