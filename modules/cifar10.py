from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import matplotlib.pyplot as plt

#from urllib.request import urlretrieve
import tarfile

import pickle
import numpy as np
from sklearn.model_selection import train_test_split

DIR_BINARIES='/home/asceta/Documents/cifar10/cifar-10-batches-py/'

def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f)#, encoding='latin1')
    f.close()
    return dic

def batch_to_bc01(batch):
    ''' Converts CIFAR sample to bc01 tensor'''
    return batch.reshape([-1, 3, 32, 32])

def batch_to_b01c(batch):
    ''' Converts CIFAR sample to b01c tensor'''
    return batch_to_bc01(batch).transpose(0,2,3,1)

def labels_to_one_hot(labels):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, 10], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels

class CIFAR10:
    def __init__(self, batch_size=100, validation_proportion=0.1, augment_data=False):
        data_available = os.path.isfile(DIR_BINARIES+'data_batch_1')
        """
        if not data_available:
            print('Downloading CIFAR 10...')
            tmp_filename = '/tmp/cifar-10-python.tar.gz'
            urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                        tmp_filename)
            tar = tarfile.open(tmp_filename, "r:gz")
            tar.extractall(path=DIR_BINARIES+'../')
            tar.close()
        """
        # Training set
        self.train_data_list = []
        self.train_labels = []
        for bi in range(1,6):
            d = unpickle(DIR_BINARIES+'data_batch_'+str(bi))
            self.train_data_list.append(d['data'])
            self.train_labels += d['labels']
        self.train_labels = np.asarray(self.train_labels)
        self.train_data = np.concatenate(self.train_data_list, axis=0).astype(np.float32)
        
        # Validation set
        assert validation_proportion > 0. and validation_proportion < 1.
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            self.train_data, self.train_labels, test_size=validation_proportion, random_state=1)
                
        # Test set
        d = unpickle(DIR_BINARIES+'test_batch')
        self.test_data = d['data'].astype(np.float32)
        self.test_labels = np.asarray(d['labels'])

        # Normalize data
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data = (self.train_data-self.mean)/self.std
        self.validation_data = (self.validation_data-self.mean)/self.std
        self.test_data = (self.test_data-self.mean)/self.std

        # Converting to b01c and one-hot encoding
        self.train_data = batch_to_b01c(self.train_data)
        self.validation_data = batch_to_b01c(self.validation_data)
        self.test_data = batch_to_b01c(self.test_data)
        self.train_labels = labels_to_one_hot(self.train_labels)
        self.validation_labels = labels_to_one_hot(self.validation_labels)
        self.test_labels = labels_to_one_hot(self.test_labels)

        np.random.seed(seed=1)
        self.augment_data = augment_data
            
        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels)//self.batch_size
        self.current_batch = 0
        self.current_epoch = 0
        
    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch*self.batch_size
        end_idx = start_idx + self.batch_size 
        batch_data = self.train_data[start_idx:end_idx]
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch

        if self.augment_data:
            if np.random.randint(0, 2) == 0:
                batch_data = batch_data[:, :, ::-1, :]
            batch_data += np.random.randn(self.batch_size, 1, 1, 3)*0.05
            
        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch+1)%self.n_batches
        if self.current_batch != batch_idx+1:
            self.current_epoch += 1

            # shuffle training data
            new_order = np.random.permutation(np.arange(len(self.train_labels)))
            self.train_data = self.train_data[new_order]
            self.train_labels = self.train_labels[new_order]
            
        return ((batch_data, batch_labels), batch_idx)
    
    def getEpoch(self):
        return self.current_epoch

    # TODO: refactor getTestSet and getValidationSet to avoid code replication
    def getTestSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.test_labels)//self.batch_size):
                start_idx = i*self.batch_size
                end_idx = start_idx + self.batch_size 
                batch_data = self.test_data[start_idx:end_idx]
                batch_labels = self.test_labels[start_idx:end_idx]
        
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (self.test_data, self.test_labels)

    def getValidationSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.validation_labels)//self.batch_size):
                start_idx = i*self.batch_size
                end_idx = start_idx + self.batch_size 
                batch_data = self.validation_data[start_idx:end_idx]
                batch_labels = self.validation_labels[start_idx:end_idx]

                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (self.validation_data, self.validation_labels)

    def reset(self):
        self.current_batch = 0
        self.current_epoch = 0
        
    def getImageRGB(self, index, img_num=1):
        im = self.test_data[index:index+img_num,:,:,:]
        label = self.test_labels[index:index+img_num]
        std = batch_to_b01c(self.std)
        mean = batch_to_b01c(self.mean)
        image_view = im*std+mean
        
        return im, label, np.uint8(image_view)
        
if __name__=='__main__':
    cifar10 = CIFAR10(batch_size=1000)
    while cifar10.getEpoch()<2:
        batch, batch_idx = cifar10.nextBatch()
        print(batch_idx, cifar10.n_batches, cifar10.getEpoch())
    batches = cifar10.getTestSet(asBatches=True)
    print(len(batches))
    data, labels = cifar10.getValidationSet()
    print(labels.sum(axis=0))
    data, labels = cifar10.getTestSet()
    print(labels.sum(axis=0))
    #RGB
    im = data[0,:,:,:]
    std = batch_to_b01c(cifar10.std)
    mean = batch_to_b01c(cifar10.mean)
    image = im*std+mean
    image = image[0,:,:,:]
    cv2.imshow('image.jpg',cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1048608)
    for i in range(1,10):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    cv2.imwrite('imageRGB.jpg',cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
