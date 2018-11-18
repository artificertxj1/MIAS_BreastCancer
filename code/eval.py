# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:13:01 2018

@author: x_j_t
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from glob import glob
import h5py
import networks
import param
import pandas as pd
import numpy as np
import os

def data_generator(params, row_df):
    img_width=params['img_width']
    img_height=params['img_height']
    x_img=np.zeros((1, img_height, img_width, 3))
    y=np.zeros((1,1))
    scan_img=row_df['scan']
    scan_img=np.repeat(scan_img[:, :, np.newaxis], 3, axis=2)
    y_i=row_df['SEVERITY']
    x_img[0,:,:,:]=scan_img
    y[0]=y_i
    return x_img,y
    
def test(): 
    params=param.get_params()               
    valid_info_path=params['valid_info_path']
    model_path=params['model_path']
    with h5py.File(valid_info_path,'r') as f:
        test_df=pd.DataFrame({k: v.value if len(v.shape)==1 else [sub_v for sub_v in v] for k,v in f.items()})
    for key in test_df.columns:
        if isinstance(test_df[key].values[0], bytes):
            test_df[key]=test_df[key].map(lambda x:x.decode())
    os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True	
    set_session(tf.Session(config=config))
    model=networks.VGG(params)
    model.load_weights(os.path.join(model_path,'45000_VGG_noMask_Pretrain.h5'))
    model_path=params['model_path']
    for index, row in test_df.iterrows():
        x,y=data_generator(params, row)
        pred=model.predict_on_batch(x)
        #if pred==y:
        #    acc+=1
        print(pred,y)

if __name__=='__main__':
    test()