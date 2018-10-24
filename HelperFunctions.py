from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

TRANSFORMER_REDUCE = 8

def transformer_sota(x, n_trans, v, i):
    '''Transformer Layer State Of The Art Method'''
    if n_trans == 0:
    	return x

    with tf.variable_scope('transformer_%i'%(i)) as scope:
        """f_k = tf.get_variable('f_k_%i'%(i), [1])
        g_k = tf.get_variable('g_k_%i'%(i), 1)
        h_k = tf.get_variable('h_k_%i'%(i), 1)"""

        _, og_h, og_w, og_f = x.outputs.get_shape()

        n_filt = abs(v/TRANSFORMER_REDUCE)
        fconv = Conv2d(x, n_filt, (1, 1), strides =(1,1),name='f_%i'%(i))
        gconv = Conv2d(x, n_filt, (1, 1), strides =(1,1),name='g_%i'%(i))
        hconv = Conv2d(x, og_f, (1, 1), strides =(1,1),name='h_%i'%(i))

        fflat = ReshapeLayer(fconv, [-1, og_h * og_w, n_filt], name='f_reshape_%i'%(i)).outputs
        gflat = ReshapeLayer(gconv, [-1, og_h * og_w, n_filt], name='g_reshape_%i'%(i)).outputs
        hreshape = ReshapeLayer(hconv, [-1, og_h * og_w, og_f], name='h_reshape_%i'%(i)).outputs

        #pre_map = tf.einsum('abc,ade->abd', fflat, gflat, name='pre_attention_%i'%(i)) #Replace with matmul (will require some transposing probably)

        pre_map = tf.matmul(fflat, tf.transpose(gflat, [0, 2, 1]), name='pre_attention_%i'%(i))
        att_map = tf.nn.softmax(pre_map, axis=1, name='attention_%i'%(i))

        #flat_focus = tf.einsum('abc,ade->adc', hreshape, att_map, name='flat_focus_%i'%(i)) #Replace with matmul (will require some transposing probably)
        reshape_focus = tf.matmul(att_map, hreshape, name='pre_focus%i'%(i))
        trans_focus = reshape_focus
        #trans_focus = tf.transpose(reshape_focus, name='trans_attention_%i'%(i))


        print(trans_focus.get_shape())
        focus = tf.reshape(trans_focus, [-1, og_f, og_h, og_w])
        focus = tf.transpose(focus, [0,2,3,1], name='focus_%i'%(i))
        print(focus.get_shape())
        focus = InputLayer(focus)

    return focus


def transformer_wide(x, n_trans, v, i):
    '''Transformer Layer Our Method'''
    if n_trans == 0:
    	return x

    with tf.variable_scope('transformer_%i'%(i)) as scope:
        """f_k = tf.get_variable('f_k_%i'%(i), [1])
        g_k = tf.get_variable('g_k_%i'%(i), 1)
        h_k = tf.get_variable('h_k_%i'%(i), 1)"""

        _, og_h, og_w, og_f = x.outputs.get_shape()

        n_filt = abs(v/TRANSFORMER_REDUCE)
        fconv = Conv2d(x, n_trans * n_filt, (1, 1), strides =(1,1),name='f_%i'%(i))
        gconv = Conv2d(x, n_trans * n_filt, (1, 1), strides =(1,1),name='g_%i'%(i))
        hconv = Conv2d(x, og_f, (1, 1), strides =(1,1),name='h_%i'%(i))

        freshape = ReshapeLayer(fconv, [-1, og_h * og_w, n_filt, n_trans], name='f_reshape_%i'%(i)).outputs
        greshape = ReshapeLayer(gconv, [-1, og_h * og_w, n_filt, n_trans], name='g_reshape_%i'%(i)).outputs # Also try fully connected for generating the multiple maps?
        hreshape = ReshapeLayer(hconv, [-1, og_h * og_w, og_f, 1], name='h_reshape_%i'%(i)).outputs

        fmult = tf.transpose(freshape, [0,3,1,2], name='fmult_%i'%(i))
        gmult = tf.transpose(greshape, [0,3,1,2], name='gmult_%i'%(i))
        hmult = tf.transpose(hreshape, [0,3,1,2], name='hmult_%i'%(i))

        print(fmult.get_shape())
        print(gmult.get_shape())
        print(hmult.get_shape())

        pre_map = tf.matmul(fmult, tf.transpose(gflat, [0, 1, 3, 2]),name='pre_attention_%i'%(i))
        att_map = tf.nn.softmax(pre_map, axis=2, name='attention_%i'%(i))

        print(pre_map.get_shape())
        print(att_map.get_shape())

        reshape_focus = tf.matmul(att_map, hmult,name='pre_attention_%i'%(i))
        trans_focus = reshape_focus
        #trans_focus = tf.transpose(reshape_focus, [0,2,1], name='trans_attention_%i'%(i))


        print(trans_focus.get_shape())
        focus = tf.reshape(trans_focus, [-1, og_h, og_w, og_f])
        print(focus.get_shape())
        focus = InputLayer(focus)

    return focus