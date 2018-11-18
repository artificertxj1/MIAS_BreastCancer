# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:24:36 2018

@author: x_j_t
"""

import os
import sys
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Input
from keras.layers import concatenate, Flatten, Reshape, Lambda, multiply, Dropout
from keras.optimizers import Adam
from keras.models import Model, Sequential
import numpy as np

def vgg_preprocess(arg):
    r=arg[:,:,:,0]-103.939
    g=arg[:,:,:,1]-116.779
    b=arg[:,:,:,2]-123/68
    return tf.stack([r,g,b],axis=3)

def VGG(params):
    img_width=params['img_width']
    img_height=params['img_height']
    img_input=Input(shape=(img_height, img_width,3))
    img=Lambda(vgg_preprocess)(img_input)
    orig_model=VGG16(input_shape=[img_height,img_width,3], include_top=False, weights='imagenet')
    orig_model.trainable=False
    depth=orig_model.get_output_shape_at(0)[-1]
    #get feature maps
    features=orig_model(img)
    bn_features=BatchNormalization()(features)
    ##add attention map 
    attn_layer=Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
    attn_layer=Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn_layer)
    attn_layer=Conv2D(1, kernel_size=(1,1), padding='valid', activation='sigmoid')(attn_layer)
    ##insert the attention maps into all layers
    up_c2_w=np.ones((1,1,1,depth))
    up_c2  = Conv2D(depth, kernel_size = (1,1), padding = 'same', activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable=False
    attn_layer=up_c2(attn_layer)
    mask_features=multiply([attn_layer, bn_features])
    gap_features=GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap=Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
    #gap_dr=Dropout(0.5)(gap)
    dr_steps = Dense(512, activation = 'relu')(gap)
    dr_steps = Dense(256, activation = 'relu')(dr_steps)
    y = Dense(1, activation = 'sigmoid')(dr_steps)
    model=Model(inputs=[img_input], outputs=[y])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model
