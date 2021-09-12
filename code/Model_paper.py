#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

import numpy
import random

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ZeroPadding2D,Input,Add, Permute, Cropping2D, Activation, Maximum,Dropout, Flatten, Dense, Conv2D, MaxPooling2D, MaxPool2D,BatchNormalization, Convolution2D, ReLU, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D


from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import  concatenate


import tensorflow.keras


# In[ ]:


## (1) ResNet

def resnet(input_shape):
    
    input_tensor = input_shape
    
    def conv1_layer(x):    
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        return x   
    
    def conv2_layer(x):         
        x = MaxPooling2D((3, 3), 2)(x)     

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)


                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                shortcut = x        

        return x
    
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(3, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model


# In[ ]:

## (4) ResNet RF-1

def resnet_1(input_shape):
    
    input_tensor = input_shape
    
    def conv1_layer(x):    
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        return x   
    
    def conv2_layer(x):         
        x = MaxPooling2D((3, 3), 2)(x)     

        shortcut = x

        for i in range(3):
            if (i == 0 or i ==1):
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)


                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                shortcut = x        

        return x
    
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(3, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model


## (2) DWS-ResNet


def resnet_dws2(input_shape):
    
    input_tensor = input_shape
    
    def conv1_layer(x):    
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        return x   
    
    def conv2_layer(x):         
        x = MaxPooling2D((3, 3), 2)(x)     

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)


                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
           
                shortcut = x        

        return x
    
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(3, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model


# DWS ResNet RF-1
def resnet_dws3(input_shape):
    
    input_tensor = input_shape
    
    def conv1_layer(x):    
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (3, 3), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        return x   
    
    def conv2_layer(x):         
        x = MaxPooling2D((3, 3), 2)(x)     

        shortcut = x

        for i in range(3):
            if (i == 0 or i ==1):
                x = DepthwiseConv2D( (3, 3), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)


                x = Add()([x, shortcut])
                x = Activation('relu')(x)

                shortcut = x

            else:
                x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                shortcut = x        

        return x
    
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(3, activation='softmax')(x)

    model = Model(input_tensor, output_tensor)

    return model

