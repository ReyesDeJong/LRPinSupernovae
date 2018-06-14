#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing ZTF database to be saved as a samplesx21x21x3 numpy array in a pickle 

@author: asceta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import io
import gzip
import base64
from astropy.io import fits
import matplotlib.pylab as plt
import pickle as pkl
import numpy as np


class ZTF_data_preprocessor(object):
    
    """
    Cosntructor
    """
    def __init__(self, data_path = None):
        self.data_path = data_path
        
    def save_to_pickle(self, save_path, array2save = None, file_name="samples", data_path = None):
        if data_path is None:
            data_path = self.data_path
        
        if array2save is None:
            array2save = self.get_preprocessed_data(data_path)
        with open(save_path+'/'+file_name+'.pkl', 'wb') as f:
            pkl.dump(array2save, f)
        
    def get_preprocessed_data(self, path = None):
        if path is None:
            path = self.data_path
        
        list_samples = self.json2list(path)
        numpy_misshape_clean_samples = self.clean_misshaped(list_samples)
        cropped_at_center_samples = self.crop_at_center(numpy_misshape_clean_samples)
        zero_filled_samples = self.zero_fill_nans(cropped_at_center_samples)
        normalized_samples = self.normalize_01(zero_filled_samples)
        
        return normalized_samples
    
    def set_data_path(self, path):
        self.data_path = path
      
        
    def json2list(self, path):
        #load json
        with open(path, "r") as f:
            dataset = json.load(f)
        
        samples_list = []
        for i in range(len(dataset['query_result'])):
    
            channels = []
            for k, imstr in enumerate(['Template', 'Science', 'Difference']):
                stamp = dataset['query_result'][i]['cutout'+imstr]['stampData']
                stamp = base64.b64decode(stamp["$binary"].encode())
    
                with gzip.open(io.BytesIO(stamp), 'rb') as f:
                    with fits.open(io.BytesIO(f.read())) as hdul:
                        img = hdul[0].data
                        channels.append(img)
            samples_list.append(np.array(channels))
        return samples_list
    
    
    def check_samples_shapes(self, samples):
        miss_shaped_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if sample.shape[0]!=3 or sample.shape[1]!=63 or sample.shape[2]!=63:
                #print("sample %i of shape %s" %(i,str(sample.shape)))
                miss_shaped_sample_idx.append(i)
        return miss_shaped_sample_idx
    
    def clean_misshaped(self, samples):
        miss_shaped_sample_idx = self.check_samples_shapes(samples)
        print('%d misshaped samples removed' %len(miss_shaped_sample_idx))
        for index in sorted(miss_shaped_sample_idx, reverse=True):
            samples.pop(index)
        return np.moveaxis(np.array(samples), 1, -1)
    
    def crop_at_center(self, sample_numpy, cropsize=21):
        center = int((sample_numpy.shape[1]-1)/2)
        crop_side = int((cropsize-1)/2)
        crop_begin = center-crop_side
        crop_end = center+crop_side+1
        #print(center)
        #print(crop_begin, crop_end)
        return sample_numpy[:,crop_begin:crop_end,crop_begin:crop_end,:]
    
    def zero_fill_nans(self, samples_numpy):
        samples_with_nan_idx = []
        for i in range(samples_numpy.shape[0]):
            if(np.isnan(samples_numpy[i,...]).any()):
                samples_with_nan_idx.append(i)
        print('%d samples with NaNs' %len(samples_with_nan_idx))
        return np.nan_to_num(samples_numpy)
    
    def normalize_01(self, samples_numpy):
        for i in range(samples_numpy.shape[0]):
            for j in range(samples_numpy.shape[3]):
                sample = samples_numpy[i,:,:,j]
                normalized_sample = (sample-np.min(sample))/np.max(sample-np.min(sample))
                samples_numpy[i,:,:,j] = normalized_sample
        return samples_numpy
    
    def print_sample(self, img):
        fig = plt.figure()
        for k, imstr in enumerate(['Template', 'Science', 'Difference']):
            ax = fig.add_subplot(1,3,k+1)
            ax.axis('off')
            ax.set_title(imstr)
            ax.matshow(img[...,k])
            
if __name__ == "__main__":
    path_data = '/home/asceta/LRPpaper/datasets/ZTF'
    path_reals = path_data+'/broker_reals.json'
    path_bogus = path_data+'/broker_bogus.json'
    
    data_processor = ZTF_data_preprocessor()
    
    print('Number of reals: %d' %len(data_processor.json2list(path_reals)))
    print('Number of bogus: %d' %len(data_processor.json2list(path_bogus)))
    
    print("\nReals")
    preprocessed_reals = data_processor.get_preprocessed_data(path_reals)
    print("Bogus")
    preprocessed_bogus = data_processor.get_preprocessed_data(path_bogus)
    
    print('\nNumber of reals after preprocessing: %d' % preprocessed_reals.shape[0])
    print('Number of bogus after preprocessing: %d' % preprocessed_bogus.shape[0])
    
    data_processor.save_to_pickle(path_data, array2save = preprocessed_bogus)
    