import os
import time
import math
from datetime import datetime
import numpy as np
from PIL import Image

from keras import Input

def open_image(name):
    try:
        image = Image.open(name)
        img_arr = np.array(image)
        #image.show()
        return img_arr, name
    except IOError: 
        print("Cannot open image!")

def calculate_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
  
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1
    
def build_encoder():
  
    input_layer = Input(shape = (64, 64, 3))
  
    ## 1st Convolutional Block
    enc = Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
  
    ## 2nd Convolutional Block
    enc = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
  
    ## 3rd Convolutional Block
    enc = Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
  
    ## 4th Convolutional Block
    enc = Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
  
    ## Flatten layer
    enc = Flatten()(enc)
  
    ## 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha = 0.2)(enc)
  
    ## 2nd Fully Connected Layer
    enc = Dense(100)(enc)
  
    ## Create a model
    model = Model(inputs = [input_layer], outputs = [enc])
    return model
    
if __name__ == '__main__':
    img = open_image('input_img/1.jpg')
    print("Oke")