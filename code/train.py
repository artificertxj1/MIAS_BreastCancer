# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:33:37 2018

@author: x_j_t
"""
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from skimage.io import imread
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from glob import glob
import h5py
import DataGenerator
import networks
import param

def write_h5(out_path, out_df):
    with h5py.File(out_path, 'w') as h:
        for k, arr_dict in tqdm(out_df.to_dict().items()): 
            try:
                s_data = np.stack(arr_dict.values(), 0)

                try:
                    h.create_dataset(k, data = s_data, compression = 'gzip')
                except TypeError as e: 
                    try:
                        h.create_dataset(k, data = s_data.astype(np.string_))
                    except TypeError as e2: 
                        print('%s could not be added to hdf5, %s' % (k, repr(e), repr(e2)))

            except ValueError as e:
                print('%s could not be created, %s' % (k, repr(e)))
                all_shape = [np.shape(x) for x in arr_dict.values()]

def print_process(step, train_loss):
    print("At {} step, binary entropy loss is: {}".format(step, train_loss))
    
def train(): 
    params=param.get_params()               
    train_info_path=params['train_info_path']
    valid_info_path=params['valid_info_path']
    #valid_info_path='../DATA/validInfo.h5'
    h5_path='../DATA/mias.h5'
    if not os.path.exists(train_info_path) or not os.path.exists(valid_info_path):
        if not os.path.isfile(h5_path):
            print("The h5 file doesn't exist, exit on error\n")
            sys.exit(1)
        with h5py.File(h5_path,'r') as f:
            data_df=pd.DataFrame({k: v.value if len(v.shape)==1 else [sub_v for sub_v in v] for k,v in f.items()})
        for key in data_df.columns:
            if isinstance(data_df[key].values[0], bytes):
                data_df[key]=data_df[key].map(lambda x:x.decode()) #change the bytes to utf8
                # hot encoding severity to 0 and 1
        abnorm_type={"SEVERITY":{"B":0, "M":1}}
        data_df.replace(abnorm_type,inplace=True)
        # split the data into train and valid data frames
        train_df, valid_df=train_test_split(data_df, test_size=0.1, random_state=2018, stratify=data_df[['SEVERITY']])
        try:
            write_h5(train_info_path, train_df)
            write_h5(valid_info_path, valid_df)
        except AttributeError as e:
            print("failed to save the train and valid dataset, quit on error\n")
            sys.exit(1)
    else:
        with h5py.File(train_info_path,'r') as f:
            train_df=pd.DataFrame({k: v.value if len(v.shape)==1 else [sub_v for sub_v in v] for k,v in f.items()})
        for key in train_df.columns:
            if isinstance(train_df[key].values[0], bytes):
                train_df[key]=train_df[key].map(lambda x:x.decode())
    os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True	
    set_session(tf.Session(config=config))
    train_feed=DataGenerator.create_feed(params, train_df)
    model=networks.VGG(params)
    
    model_path=params['model_path']
    iter_num=params['train_iter']
    for step in range(iter_num):
        x,y=next(train_feed)
        train_loss=model.train_on_batch(x,y)
        print_process(step, train_loss)
        if step>0 and step% 5000==0:
            model.save(os.path.join(model_path,str(step)+'_VGG_noMask_Pretrain.h5'))

if __name__=='__main__':
    train()
    
