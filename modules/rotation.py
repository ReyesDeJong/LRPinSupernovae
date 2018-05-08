"""
Correct rotation as in DeepHits
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules.module import Module

#import pdb
#import activations
import numpy as np
na = np.newaxis

class Rotation(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, rotation_num = 4, batch_size = None, input_dim = None, input_depth=None,
                 name="rotation", param_dir=None, epsilon = 1e-3, decay = 0.5):
        
        self.name = name
        #self.input_tensor = input_tensor
        Module.__init__(self)
        
        #comment (it's done in fwd pass)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_depth = input_depth
        

        
        self.rotation_num = rotation_num
        
              


    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
                if len(inp_shape)!=4:
                    mod_shape = [self.batch_size, self.input_dim, self.input_dim, self.input_depth]
                    self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
                raise ValueError('Expected dimension of input tensor: 4')
                
#join_aux = tf.reshape(conv_layer_5, [4, BATCH_SIZE, -1])
#fc_input = tf.reshape(tf.transpose(join_aux, perm=[1, 0, 2]), [BATCH_SIZE, -1])
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.check_input_shape()
        
        self.batch_size = tf.divide(tf.shape(input_tensor)[0],self.rotation_num)
        #self.input_dim = tf.shape(self.input_tensor)[1]
        #self.input_depth = tf.shape(self.input_tensor)[3]
        #pdb.set_trace()
        
        #input images, featuremap height, feturemap width, num_input_channels
        self.in_N, self.in_h, self.in_w, self.in_depth = self.input_tensor.get_shape().as_list()
        self.in_N = tf.shape(self.input_tensor)[0]

        with tf.name_scope(self.name):
            #rotation
            self.join_aux = tf.reshape(self.input_tensor, [self.rotation_num, -1, self.in_h*self.in_w*self.in_depth])
            self.fc_input = tf.reshape(tf.transpose(self.join_aux, perm=[1, 0, 2]), [-1, self.rotation_num*self.in_h*self.in_w*self.in_depth])
            #self.join_aux = tf.reshape(self.input_tensor, [4, 50, -1])
            #self.fc_input = tf.reshape(tf.transpose(tf.reshape(self.input_tensor, [4, 50, -1]), perm=[1, 0, 2]), [50, -1])
        
                    
            self.activations = self.fc_input
            
        return self.fc_input
    
    #FATAL ERROR
    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''

        #self.check_shape(R)
        #if len(R_shape)!=4:
        #R = R.reshape((-1, self.in_h, self.in_w, self.in_depth))  
        #return R
        # [50, 4*2304]
        self.R = R
        #R_shape = R.get_shape().as_list()
        #[50,4,2304]        
        self.disjoin_aux = tf.reshape(self.R, [-1, self.rotation_num, self.in_h*self.in_w*self.in_depth])  
        #[4,50,2304]
        self.transpose = tf.transpose(self.disjoin_aux, perm=[1, 0, 2])
        #[200,6,6,64]
        return tf.reshape(self.transpose, [-1, self.in_h, self.in_w, self.in_depth])
        
    def _epsilon_lrp(self,R, epsilon=1e-12):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        return self._simple_lrp(R)
    
    
    

    def _ww_lrp(self,R): 
        '''
        LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
        '''
        return self._simple_lrp(R)

    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        return self._simple_lrp(R)

#Cambio en funcion compute_zs, nose si es necesairo dejar el stibilizer false #changed to True
    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        return self._simple_lrp(R)
