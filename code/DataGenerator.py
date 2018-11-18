# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:51:10 2018

@author: x_j_t
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
import math

def data_generator(params, data_df, do_augment=True):
    img_width=params['img_width']
    img_height=params['img_height']
    batch_size=params['batch_size']
    sigma=params['sigma']
    while True:
        x_img=np.zeros((batch_size, img_height, img_width, 3))
        y=np.zeros((batch_size,1))
        for i in range(batch_size):
            imgInd=np.random.choice(len(data_df),1)[0]
            row_df=data_df.iloc[imgInd]
            scan_img=row_df['scan']
            #scan_img=np.repeat(scan_img[:, :, np.newaxis], 3, axis=2)
            y_i=row_df['SEVERITY']
            ## some x and y are NOTE or NaN not float wth is NOTE?
            try:
                center=[float(row_df['X']), float(row_df['Y'])]
            except ValueError:
                center=[None,None]
            if do_augment:
                rflip, rscale, rshift, rdegree, rsat=rand_augmentations(params)
                scan_img=augment(scan_img, rflip, rscale, rshift, rdegree, rsat, img_height, img_width, center, sigma)
            x_img[i,:,:,:]=scan_img
            y[i]=y_i
        #break
        yield (x_img, y)
    #return x_img,y


def create_feed(params, data_df, do_augment=True):
   feed=data_generator(params, data_df, do_augment)
   return feed

 
## random augmentations
def rand_scale(param):
    rnd=np.random.rand()
    return (param['scale_max']-param['scale_min'])*rnd+param['scale_min']

def rand_rot(param):
    return (np.random.rand()-0.5)*2*param['max_rotate_degree']

def rand_shift(param):
    shift_px=param['max_px_shift']
    x_shift=int(shift_px * (np.random.rand()-0.5))
    y_shift=int(shift_px * (np.random.rand()-0.5))
    return x_shift, y_shift

def rand_sat(param):
    min_sat= 1-param['max_sat_factor']
    max_sat= 1+param['max_sat_factor']
    return np.random.rand()*(max_sat-min_sat)+min_sat


def rand_augmentations(param):
    rflip=np.random.rand()
    rscale=rand_scale(param)
    rshift=rand_shift(param)
    rdegree=rand_rot(param)
    rsat=rand_sat(param)
    return rflip, rscale, rshift, rdegree, rsat

def make_gaussian_map(img_width, img_height, center, var_x, var_y, theta):
    xv,yv=np.meshgrid(np.array(range(img_width)), np.array(range(img_height)), sparse=False, indexing='xy')
    a=np.cos(theta)**2 / (2*var_x) + np.sin(theta)**2 /(2*var_y)
    b=-np.sin(2*theta) / (4*var_x) + np.sin(2*theta) / (4*var_y)
    c=np.sin(theta)**2 / (2*var_x) + np.cos(theta)**2/ (2*var_y)
    return np.exp(-(a*(xv-center[0])*(xv-center[0]) + 2*b*(xv-center[0])*(yv-center[1]) + c*(yv-center[1])*(yv-center[1])))

def augment(I, rflip, rscale, rshift, rdegree, rsat, img_height, img_width, center, sigma):
    # first make a Gaussian map around the center of the abnormal part
    #if center[0]!=None and not math.isnan(center[0]):
    #    gaussMap=make_gaussian_map(img_width, img_height, center, sigma*sigma, sigma*sigma, 0.0)
    #    I= np.multiply(I, gaussMap)
    I=np.repeat(I[:, :, np.newaxis], 3, axis=2)
    I=aug_flip(I, rflip)
    I=aug_scale(I, rscale)
    I=aug_rotate(I,img_width, img_height, rdegree)
    I=aug_saturation(I, rsat)
    return I


def aug_flip(I, rflip):
    if (rflip<0.5):
        return I
    I=np.fliplr(I)
    return I

def aug_scale(I, rscale):
    I=cv2.resize(I, (0,0), fx=rscale, fy=rscale)
    return I

def aug_rotate(I, img_width, img_height, rdegree):
    h=I.shape[0]
    w=I.shape[1]
    center=(int((w-1.0)/2.0), int((h-1.0)/2.0))
    R=cv2.getRotationMatrix2D(center, rdegree,1)
    I=cv2.warpAffine(I, R, (img_width, img_height))
    return I


def aug_saturation(I, rsat):
    I=I*rsat
    I[I>255.0]=255.0
    #I[I<-1.0]=-1.0
    return I#.astype(int)
