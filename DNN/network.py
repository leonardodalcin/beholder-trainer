# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 12:25:10 2018

@author: Administrator
"""

import tensorflow as tf

n_classes = 2
# NN Arch parameter 
num_of_res = 6 

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def bias_init(shape, bias_value=0.01):
    return tf.Variable(tf.constant(bias_value, shape=shape))

def conv2d_custom(input, filter_size, num_of_channels, num_of_filters, activation=tf.nn.relu, dropout=None,
                  padding='SAME', max_pool=True, strides=(1, 1)):  
    weights = weights_init([filter_size, filter_size, num_of_channels, num_of_filters])
    bias = bias_init([num_of_filters])
    
    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding) + bias
    
    if activation != None:
        layer = activation(layer)
    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
    return layer

def flatten(layer):
    shape = layer.get_shape()
    num_elements_ = shape[1:4].num_elements()
    flattened_layer = tf.reshape(layer, [-1, num_elements_])
    return flattened_layer, num_elements_

def dense_custom(input, input_size, output_size, activation=tf.nn.relu, dropout=None):

    weights = weights_init([input_size, output_size])
    bias = bias_init([output_size])
    
    layer = tf.matmul(input, weights) + bias
    if activation != None:
        layer = activation(layer)
    
    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)
        
    return layer

def residual_unit(layer):
    step0 = conv2d_custom(layer, 3, 24, 24, activation=None, max_pool=False)
    step1 = tf.layers.batch_normalization(step0)
    step2 = tf.nn.relu(step1)
    step3 = conv2d_custom(step2, 3, 24, 24, activation=None, max_pool=False) 
    step4 = tf.layers.batch_normalization(step3)
    step5 = layer + step4
    return tf.nn.relu(step5)

def resnetMold(X, keep_prob):
    X = (X - 127.5) / 255
    
    step3x3 = conv2d_custom(X, 3, 1, 3, max_pool = False)
    step5x5 = conv2d_custom(X, 5, 1, 3, max_pool=False)
    concat_l = tf.concat([step3x3, step5x5],3)

    step5x5_2 = conv2d_custom(concat_l, 5, 6, 12, max_pool=False)
    step7x7_2 = conv2d_custom(concat_l, 7, 6, 12, max_pool=False)
    prev1 = tf.concat([step5x5_2, step7x7_2],3)   

    for i in range(num_of_res): 
        prev1 = residual_unit(prev1)

    #after all resunits we have last conv layer, than flattening and output layer
    last_conv = conv2d_custom(prev1, 3, 24, n_classes, activation=None, max_pool=False)
    flat, features = flatten(last_conv)
    print("Features:", features)
    dense = dense_custom(flat, features, 1000, dropout=keep_prob)
    dense = dense_custom(dense, 1000, 100, dropout=keep_prob)
    output = dense_custom(dense, 100, n_classes, dropout=keep_prob)

    return output




