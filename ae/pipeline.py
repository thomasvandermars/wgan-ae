import numpy as np
import cv2
import tensorflow as tf

from skimage.util import random_noise
from keras.utils import Sequence

def add_blur(image, kernel_size = 3):
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return image

class ae_data_generator(Sequence):
    
    def __init__(self, filenames, batch_size, img_dims, blur_kernel = 3, noise_var = 0.001):
        self.filenames = filenames
        self.batch_size = batch_size
        self.img_dims = img_dims
        self.blur_kernel = blur_kernel
        self.noise_var = noise_var
        pass
    
    def __len__(self):
        
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        # extract the batch
        batch = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # lists for input and target images
        X, Y = [], []
        
        # iterate through batch of filenames...
        for fname in batch:
        
            # read in image, convert to RGB and resize
            image = cv2.imread(fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_dims[1], self.img_dims[0])) # note that cv2.resize takes (width, height)

            Y.append(image) # add as target image

            # add contrast, blur en random noise to the image
            #image = tf.image.adjust_contrast(image, 1.5)
            image = add_blur(image, self.blur_kernel)
            image = random_noise(image/255., mode = "localvar", local_vars = np.ones(image.shape)*self.noise_var)

            X.append(image) # add the blurred and noisy image as input...
            pass
        
        return [np.array(X), np.array(Y)/255.]