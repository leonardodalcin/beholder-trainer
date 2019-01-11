import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import glob2
from sklearn.model_selection import train_test_split
import os
from DNN.network import *
from random import *

INPUT_WIDTH = 96
INPUT_HEIGHT = 96
BATCH_SIZE = 32
rate = 0.0001

import logging as log
# filename='app.log', filemode='w',
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S')

class Predictor(object):

    def __init__(self, input_folder, output_file, pretrained_model):

        args = {
            "input_folder": input_folder,
            "output_file": output_file,
            "pretrained_model": pretrained_model
        }

        test_fn = glob2.glob(args.input_folder + "//*.pgm")

        if len(test_fn) == 0:
            print("No input files found... The folder should contain all the"
                  "PGM files in the parent directory.")
            exit()

        X_test = []
        X_test.extend(test_fn)

        tf.reset_default_graph()

        f = open(args.output_file, "w+")

        # load resnet Classifier
        graph_resClass = tf.Graph()
        sess_resClass = tf.Session(graph=graph_resClass)
        with graph_resClass.as_default():
            # Input/ prediction / cost function / optimizer
            keep_prob = tf.placeholder_with_default(1.0, shape=())

            X = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH, 1], name="X")
            y = tf.placeholder(tf.int32, shape=[None], name="y")

            with tf.name_scope('Model'):
                logits = resnetMold(X, keep_prob)

            one_hot_y = tf.one_hot(y, n_classes)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)
            with tf.name_scope('Loss'):
                loss_op = tf.reduce_mean(cross_entropy)
            with tf.name_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=rate)
            training_op = optimizer.minimize(loss_op)

            accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(one_hot_y, 1),
                                                        predictions=tf.argmax(logits, 1))
            confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.argmax(one_hot_y, 1),
                                                                   predictions=tf.argmax(logits, 1))

            prediction = tf.argmax(logits, 1, name='arg_output', output_type=tf.int32)
            softmax_out = tf.nn.softmax(logits, name='softmax_output')

            tot_loss_ph = tf.placeholder(tf.float32, name='tot_loss')

            tf.summary.scalar("loss", loss_op)
            tf.summary.scalar("accurary", accuracy_op)
            tf.summary.scalar("tot_loss", tot_loss_ph)
            init = tf.global_variables_initializer()
            sess_resClass.run(init)

            saver_resnet = tf.train.Saver()
            if args.pretrained_model != "":
                saver_resnet.restore(sess_resClass, args.pretrained_model)
            else:
                raise ValueError("model required for Resnet Classifier")
            for file in X_test:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
                outp = sess_resClass.run(softmax_out, feed_dict={X: img.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 1),
                                                                 keep_prob: 1.0})
                f.write(file + "[--]" + str(outp[0]) + "\n")

            f.close()
