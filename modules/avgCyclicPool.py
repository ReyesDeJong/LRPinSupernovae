'''
@author: Vignesh Srinivasan
@author: Sebastian Lapushkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from module import Module
import numpy as np


from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

#avCyclicPool2 is more stable but may be slower or expensive
#
class CyclicAvgPool(Module):

    def __init__(self, pool_h=4, pool_w=1, pool_stride=None, pad = 'SAME',name='cyclic_avgpool'):
        self.name = name
        Module.__init__(self)
        #rows
        self.pool_h = pool_h
        #cols
        self.pool_w = pool_w
        #kernel [1,4,1,1]
        self.pool_kernel = [1]+[self.pool_h, self.pool_w]+[1]
        #stride [1,4,1,1]
        self.stride_h = self.pool_h
        self.stride_w = self.pool_w
        self.pool_stride = self.pool_kernel

        self.pad = pad
        

    def check_input_shape(self):
        inp_shape = self.input.get_shape().as_list()
        try:
                if len(inp_shape)!=2:
                    self.input_dim =  np.prod(inp_shape[1:])
                    self.input= tf.reshape(self.input,[-1, self.input_dim])
        except:
                raise ValueError('Expected dimension of input tensor: 2')
        
    def forward(self,input_tensor, batch_size=10, img_dim=28):
        self.input = input_tensor
        self.check_input_shape()
        # in_shape = self.input_tensor.get_shape().as_list()
        # self.in_N = in_shape[0]
        # self.in_depth = in_shape[-1]
        self.in_N, self.in_feat= self.input.get_shape().as_list()
        self.in_N = tf.shape(self.input)[0]
        
        with tf.name_scope(self.name):
            #[4,batch,feat]
            self.flat_reshape = tf.reshape(self.input, [4,-1,self.in_feat])
            #[batch,4,feat]
            self.transp = tf.transpose(self.flat_reshape, perm=[1, 0, 2])
            #[batch,4,feat,1]
            self.expanded = tf.expand_dims(self.transp, -1)
            
            self.input_tensor = self.expanded
            self.in_N, self.in_h, self.in_w, self.in_depth = self.input_tensor.get_shape().as_list()
            self.in_N = tf.shape(self.input_tensor)[0]
            
            #[10,1,784,1]
            self.activations = tf.nn.avg_pool(self.expanded, ksize=self.pool_kernel, strides=self.pool_stride, padding=self.pad, name=self.name )
            
            self.out = tf.reshape(self.activations, [-1,self.in_feat])
            #tf.summary.histogram('activations', self.activations)
        return self.out

    def clean(self):
        self.activations = None

    # def lrp(self,R,*args,**kwargs):
    #     return R
    


    
    def check_shapePy(self, R):

        activations_shape = self.activations.get_shape().as_list()
        #if len(R_shape)!=4:
        if len(R.shape)!=4:
            R = R.reshape((-1, activations_shape[1], activations_shape[2], activations_shape[3]))
        
        return R

    
    

    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        self.check_shape(R)
        image_patches = self.extract_patches()
        Z = image_patches
        Zs = self.compute_zs(Z)
        result = self.compute_result(Z,Zs)
        return self.reshape_Routput(self.restitch_image(result))

    def _epsilon_lrp(self,R, epsilon=1e-12):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = image_patches
        Zs = self.compute_zs(Z, epsilon=epsilon)
        result = self.compute_result(Z,Zs)
        return self.reshape_Routput(self.restitch_image(result))
    
    
    def _epsilon_lrpPy(self, sess, feed_dict, R, epsilon=1e-12):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        R=self.check_shapePy(R)
        
        inp_shape = self.input_tensor.get_shape().as_list()
        #bad practice
        name=next(iter(feed_dict))
        N = feed_dict.get(name).shape[0]
        H = inp_shape[1]
        W = inp_shape[2]
        D = inp_shape[3]
        X = sess.run(self.input_tensor, feed_dict=feed_dict)

        hpool   = self.pool_h
        wpool   = self.pool_w
        hstride = self.pool_stride[1]
        wstride = self.pool_stride[2]

        #assume the given pooling and stride parameters are carefully chosen.
        Hout = (H - hpool) / hstride + 1
        Wout = (W - wpool) / wstride + 1

        Rx = np.zeros((N,H,W,D))
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = X[:, i*hstride:i*hstride+hpool , j*wstride:j*wstride+wpool , : ] #input activations.
                Zs = Z.sum(axis=(1,2),keepdims=True)
                Zs += epsilon*((Zs >= 0)*2-1) # add a epsilon stabilizer to cushion an all-zero input

                Rx[:,i*hstride:i*hstride+hpool: , j*wstride:j*wstride+wpool: , : ] += (Z/Zs) * R[:,i:i+1,j:j+1,:]  #distribute relevance propoprtional to input activations per layer

        return Rx
    
    
    def _ww_lrp(self,R): 
        '''
        due to uniform weights used for sum pooling (1), this method defaults to _flat_lrp(R)
        '''
        return self._flat_lrp(R)
    
    def _flat_lrp(self,R):
        '''
        distribute relevance for each output evenly to the output neurons' receptive fields.
        '''
        self.check_shape(R)

        Z = tf.ones([self.in_N, self.Hout,self.Wout, self.pool_h,self.pool_w, self.in_depth])
        Zs = self.compute_zs(Z)
        result = self.compute_result(Z,Zs)
        return self.reshape_Routput(self.restitch_image(result))
    
    #[OJO] CAMBIAZO EN *alpha *beta OJO esto cuando no sea ReLU
    def _alphabeta_lrp(self,R,alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        self.check_shape(R)

        image_patches = self.extract_patches()
        Z = image_patches
        if not alpha == 0:
            Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
            Zsp = self.compute_zs(Zp)
            #Ralpha = alpha * self.compute_result(Zp,Zsp)
            Ralpha = self.compute_result(Zp,Zsp)
        else:
            Ralpha = 0
        if not beta == 0:
            Zn = tf.where(tf.less(Z,0),Z, tf.zeros_like(Z))
            Zsn = self.compute_zs(Zn)
            #Rbeta = beta * self.compute_result(Zn,Zsn)
            Rbeta = self.compute_result(Zn,Zsn)
        else:
            Rbeta = 0

        result = Ralpha + Rbeta
        return self.reshape_Routput(self.restitch_image(result))


    def check_shape(self, R):
        self.R = R
        R_shape = self.R.get_shape().as_list()
        #R_shape = tf.shape(self.R)
        #activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
        #if R_shape[0]!=4:
            self.R = tf.expand_dims(self.R, -1)
            self.R = tf.expand_dims(self.R, 1)
        N,self.Hout,self.Wout,NF = self.R.get_shape().as_list()
        N = tf.shape(self.R)[0]
        
    def reshape_Routput(self, Rout):
        N, h, w, depth = Rout.get_shape().as_list()
        return tf.reshape(Rout, [-1, h*w*depth])

    def extract_patches(self):
        image_patches = tf.extract_image_patches(self.input_tensor, ksizes=[1, self.pool_h,self.pool_w, 1], strides=[1, self.stride_h,self.stride_w, 1], rates=[1, 1, 1, 1], padding=self.pad)
        return tf.reshape(image_patches, [self.in_N, self.Hout,self.Wout, self.pool_h,self.pool_w, self.in_depth])
        
    def compute_z(self, image_patches):
        return tf.multiply(tf.expand_dims(self.weights, 0), tf.expand_dims(image_patches, -1))
        
    #1e-12
    def compute_zs(self, Z, stabilizer=True, epsilon=1e-12):
        #[QUESO]
        Zs = tf.reduce_sum(Z, [3,4], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        if stabilizer==True:
            stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
            Zs += stabilizer
        return Zs

    def compute_result(self, Z, Zs):
        #import pdb; pdb.set_trace()
        result = (Z/Zs) * tf.reshape(self.R, [self.in_N,self.Hout,self.Wout,1,1,self.in_depth])
        return tf.reshape(result, [self.in_N,self.Hout,self.Wout, self.pool_h*self.pool_w*self.in_depth])

    def restitch_image(self, result):
        return self.patches_to_images(result, self.in_N, self.in_h, self.in_w, self.in_depth, self.Hout, self.Wout, self.pool_h, self.pool_w, self.stride_w,self.stride_h )
    #                            self, grad, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c, stride_h, stride_r
    
    # # Old methods
    def _simple__lrp(self,R):
        import time; start_time = time.time()

        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf= self.pool_h, self.pool_w
        hstride, wstride = self.stride_h, self.stride_w
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        #import pdb; pdb.set_trace()
        Z = image_patches
        #Z = tf.where(Z, tf.ones_like(Z, dtype=tf.float32), tf.zeros_like(Z,dtype=tf.float32) )
        #Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   (Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,NF])
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx



    #stride are backwards!!!    
    # '''                    (result, self.in_N, self.in_h, self.in_w, self.in_depth, self.Hout, self.Wout, self.pool_h, self.pool_w, self.stride_w,self.stride_h )
    def patches_to_images(self, grad, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c, stride_h, stride_r ):
        rate_r = 1
        rate_c = 1
        padding = self.pad
        #[10,4,784,1] in 
        #[10,1,784,1] out
        
        ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
        ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

        if padding == 'SAME':
          if rows_out * 2 != rows_in:
            rows_out = int(ceil((rows_in) / stride_r))
            #rows_out = int(ceil((rows_in+1) / stride_r))
            #self.rows_out = int(ceil((rows_in) / stride_r))
            #self.rows_in=rows_in
            #self.stride_r = stride_r
            #cols_out = int(ceil((cols_in+1) / stride_h))
            cols_out = int(ceil((cols_in) / stride_h))
            #self.cols_out = cols_out
          else:    
            rows_out = int(ceil(rows_in / stride_r))
            cols_out = int(ceil(cols_in / stride_h))
          pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
          pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

        elif padding == 'VALID':
          if rows_out * 2 != rows_in:
            rows_out = int(ceil(((rows_in+1) - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil(((cols_in+1) - ksize_c_eff + 1) / stride_h))
          else:    
            rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
          pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
          pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

        pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

        grad_expanded = array_ops.transpose(
            array_ops.reshape(grad, (batch_size, rows_out,
                                     cols_out, ksize_r, ksize_c, channels)),
            (1, 2, 3, 4, 0, 5)
        )
        grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

        row_steps = range(0, rows_out * stride_r, stride_r)
        col_steps = range(0, cols_out * stride_h, stride_h)

        idx = []
        for i in range(rows_out):
            for j in range(cols_out):
                r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
                r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

                idx.extend([(r * (cols_in) + c,
                   i * (cols_out * ksize_r * ksize_c) +
                   j * (ksize_r * ksize_c) +
                   ri * (ksize_c) + ci)
                  for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                  for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                  if 0 <= r and r < rows_in and 0 <= c and c < cols_in
                ])

        sp_shape = (rows_in * cols_in,
              rows_out * cols_out * ksize_r * ksize_c)

        sp_mat = sparse_tensor.SparseTensor(
            array_ops.constant(idx, dtype=ops.dtypes.int64),
            array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
            sp_shape
        )

        jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

        grad_out = array_ops.reshape(
            jac, (rows_in, cols_in, batch_size, channels)
        )
        grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
        
        return grad_out

