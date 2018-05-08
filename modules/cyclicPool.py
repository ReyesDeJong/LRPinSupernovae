from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules.module import Module

#import pdb
#import activations
import numpy as np
na = np.newaxis

class DenseCyclicPool(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, rotation_num = 4, batch_size = None, input_dim = None, input_depth=None,
                 name="cyclic_avgPool"):
        
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
                if len(inp_shape)!=2:
                    self.input_dim =  np.prod(inp_shape[1:])
                    self.input_tensor = tf.reshape(self.input_tensor,[-1, self.input_dim])
        except:
                raise ValueError('Expected dimension of input tensor: 2')
                
#join_aux = tf.reshape(conv_layer_5, [4, BATCH_SIZE, -1])
#fc_input = tf.reshape(tf.transpose(join_aux, perm=[1, 0, 2]), [BATCH_SIZE, -1])
    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        self.check_input_shape()
        #self.input_dim = tf.shape(self.input_tensor)[1]
        #self.input_depth = tf.shape(self.input_tensor)[3]
        #pdb.set_trace()
        
        #input images, featuremap height, feturemap width, num_input_channels
        self.in_N, self.in_feat = self.input_tensor.get_shape().as_list()
        self.in_N = tf.shape(self.input_tensor)[0]

        with tf.name_scope(self.name):
            #rotation
            self.flat_reshape = tf.reshape(self.input_tensor, [self.rotation_num,-1,self.in_feat])
            
            self.pooled_flat = tf.reduce_sum(self.flat_reshape, [0])/self.rotation_num
                    
            self.activations = self.pooled_flat
            
        return self.activations
    

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
        #self.disjoin_aux = tf.reshape(self.input_tensor, [-1, self.rotation_num, self.in_h*self.in_w*self.in_depth])  
        #[4,50,2304]
        #self.transpose = tf.transpose(self.disjoin_aux, perm=[1, 0, 2])
        #[200,6,6,64]
        #return tf.reshape(self.transpose, [-1, self.in_h, self.in_w, self.in_depth])
        return R
        
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
    
    def check_shape(self, R):
        self.R = R
        #R_shape = self.R.get_shape().as_list()
        R_shape = tf.shape(self.R)
        activations_shape = self.activations.get_shape().as_list()
        #if len(R_shape)!=4:
        if R_shape[0]!=2:
            self.R = tf.reshape(self.R, [-1, activations_shape[1]])
