from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorlayer.layers import *

TRANSFORMER_REDUCE = 2

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
        fconv = Conv2d(x, n_filt, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='f_%i'%(i))
        gconv = Conv2d(x, n_filt, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='g_%i'%(i))
        hconv = Conv2d(x, n_filt, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='h_%i'%(i))

        print("convs")
        print(fconv.outputs.get_shape())
        print(gconv.outputs.get_shape())
        print(hconv.outputs.get_shape())


        freshape = ReshapeLayer(fconv, [-1, og_h * og_w, n_filt], name='f_reshape_%i'%(i)).outputs
        greshape = ReshapeLayer(gconv, [-1, og_h * og_w, n_filt], name='g_reshape_%i'%(i)).outputs
        hreshape = ReshapeLayer(hconv, [-1, og_h * og_w, n_filt], name='h_reshape_%i'%(i)).outputs

        print("reshapes")
        print(freshape.get_shape())
        print(greshape.get_shape())
        print(hreshape.get_shape())

        pre_map = tf.matmul(freshape, tf.transpose(greshape, [0, 2, 1]), name='pre_attention_%i'%(i))
        with tf.variable_scope('softmax_%i'%(i)) as scope:
            att_map = tf.nn.softmax(pre_map, axis=1)

        print("maps")
        print(pre_map.get_shape())
        print(att_map.get_shape())

        unshaped_focus = tf.matmul(att_map, hreshape, name='pre_focus%i'%(i))
        focus = tf.reshape(unshaped_focus, [-1, og_h, og_w, n_filt])
        
        focus = Conv2d(InputLayer(focus), og_f, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='focus_%i'%(i)).outputs
        
        #My conv2d
        #focus = tf.nn.conv2d(focus, filter=tf.get_variable("focuser", [1, 1, n_filt, og_f]), strides=[1,1,1,1], padding="SAME", name ='focus%i'%(i))
        #focus = tf.add(focus, tf.get_variable("focus_bias", focus.get_shape()[1:]), name = "focus_bias_add")

        out = InputLayer(focus + x.outputs, name = 'resnet_%i'%(i))

        print("focuses")
        print(unshaped_focus.get_shape())
        print(focus.get_shape())



        

    return out


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
        hconv = Conv2d(x, n_trans * n_filt, (1, 1), strides =(1,1),name='h_%i'%(i))

        freshape = ReshapeLayer(fconv, [-1, og_h * og_w, n_filt, n_trans], name='f_reshape_%i'%(i)).outputs
        greshape = ReshapeLayer(gconv, [-1, og_h * og_w, n_filt, n_trans], name='g_reshape_%i'%(i)).outputs # Also try fully connected for generating the multiple maps?
        hreshape = ReshapeLayer(hconv, [-1, og_h * og_w, n_filt, n_trans], name='h_reshape_%i'%(i)).outputs

        print("reshapes")
        print(freshape.get_shape())
        print(greshape.get_shape())
        print(hreshape.get_shape())

        fmult = tf.transpose(freshape, [0,3,1,2], name='fmult_%i'%(i))
        gmult = tf.transpose(greshape, [0,3,1,2], name='gmult_%i'%(i))
        hmult = tf.transpose(hreshape, [0,3,1,2], name='hmult_%i'%(i))

        print("intermediate vectors")
        print(fmult.get_shape())
        print(gmult.get_shape())
        print(hmult.get_shape())

        pre_map = tf.matmul(fmult, tf.transpose(gmult, [0, 1, 3, 2]),name='pre_attention_%i'%(i))
        att_map = tf.nn.softmax(pre_map, axis=2, name='attention_%i'%(i))

        print("maps")
        print(pre_map.get_shape())
        print(att_map.get_shape())

        unshaped_focus = tf.matmul(att_map, hmult, name='pre_focus%i'%(i))
        trans_focus = tf.transpose(unshaped_focus, [0, 2, 1, 3])
        shaped_focus = tf.reshape(trans_focus, [-1, og_h, og_w, n_trans*n_filt])
        

        focus = Conv2d(InputLayer(shaped_focus), og_f, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='focus_%i'%(i)).outputs

        #focus = tf.nn.conv2d(shaped_focus, filter=tf.get_variable("focuser", [1, 1, n_filt*n_trans, og_f]), strides=[1,1,1,1], padding="SAME", name ='focus%i'%(i))
        #focus = tf.add(focus, tf.get_variable("focus_bias", focus.get_shape()[1:]), name = "focus_bias_add")
        
        out = InputLayer(focus + x.outputs, name = 'resnet_%i'%(i))

        print("focuses")
        print(unshaped_focus.get_shape())
        print(trans_focus.get_shape())
        print(shaped_focus.get_shape())
        print(focus.get_shape())

    return out


def transformer_parallel(x, n_trans, v, i):
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
        hconv = Conv2d(x, n_trans * n_filt, (1, 1), strides =(1,1),name='h_%i'%(i))

        freshape = ReshapeLayer(fconv, [-1, og_h * og_w, n_filt, n_trans], name='f_reshape_%i'%(i)).outputs
        greshape = ReshapeLayer(gconv, [-1, og_h * og_w, n_filt, n_trans], name='g_reshape_%i'%(i)).outputs # Also try fully connected for generating the multiple maps?
        hreshape = ReshapeLayer(hconv, [-1, og_h * og_w, n_filt, n_trans], name='h_reshape_%i'%(i)).outputs

        print("reshapes")
        print(freshape.get_shape())
        print(greshape.get_shape())
        print(hreshape.get_shape())

        fmult = tf.transpose(freshape, [0,3,1,2], name='fmult_%i'%(i))
        gmult = tf.transpose(greshape, [0,3,1,2], name='gmult_%i'%(i))
        hmult = tf.transpose(hreshape, [0,3,1,2], name='hmult_%i'%(i))

        print("intermediate vectors")
        print(fmult.get_shape())
        print(gmult.get_shape())
        print(hmult.get_shape())

        fmult_pieces = tf.unstack(fmult, num=n_trans, axis=1, name = 'funstack_%i'%(i))
        gmult_pieces = tf.unstack(gmult, num=n_trans, axis=1, name = 'gunstack_%i'%(i))
        hmult_pieces = tf.unstack(hmult, num=n_trans, axis=1, name = 'hunstack_%i'%(i))
        focuses = []
        for j in range(n_trans):
            pre_map = tf.matmul(fmult_pieces[j], tf.transpose(gmult_pieces[j], [0, 2, 1]), name='pre_attention_%i_%i'%(i,j))
            att_map = tf.nn.softmax(pre_map, axis=1, name='attention_%i_%i'%(i,j))
            focuses.append(tf.matmul(att_map, hmult_pieces[j], name='pre_focus_%i_%i'%(i,j)))

        print("maps")
        print(pre_map.get_shape())
        print(att_map.get_shape())

        unshaped_focus = tf.stack(focuses, axis=1, name = 'map_stack_%i'%(i)) 
        trans_focus = tf.transpose(unshaped_focus, [0, 2, 1, 3])
        shaped_focus = tf.reshape(trans_focus, [-1, og_h, og_w, n_trans*n_filt])
        

        focus = Conv2d(InputLayer(shaped_focus), og_f, (1, 1), act=tf.nn.leaky_relu, strides =(1,1),name='focus_%i'%(i)).outputs

        #focus = tf.nn.conv2d(shaped_focus, filter=tf.get_variable("focuser", [1, 1, n_filt*n_trans, og_f]), strides=[1,1,1,1], padding="SAME", name ='focus%i'%(i))
        #focus = tf.add(focus, tf.get_variable("focus_bias", focus.get_shape()[1:]), name = "focus_bias_add")
        
        out = InputLayer(focus + x.outputs, name = 'resnet_%i'%(i))

        print("focuses")
        print(unshaped_focus.get_shape())
        print(trans_focus.get_shape())
        print(shaped_focus.get_shape())
        print(focus.get_shape())

    return out