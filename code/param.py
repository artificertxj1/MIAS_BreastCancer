# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:53:05 2018

@author: x_j_t
"""

def get_params():
    params={}
    params['data_path']='../DATA/mias.h5'
    params['train_info_path']='../DATA/trainInfo.h5'
    params['valid_info_path']='../DATA/validInfo.h5'
    params['model_path']='../model'
    params['sigma']= 200 
    params['img_width']=1024
    params['img_height']=1024
    params['batch_size']=3
    params['scale_max']=1.15
    params['scale_min']=0.85
    params['max_rotate_degree']=15
    params['max_px_shift']=params['img_width']*0.1
    params['max_sat_factor']=0.15
    params['train_iter']=50000
    return params
