# -*- coding: UTF-8 -*-

from datetime import datetime
import psutil
import operator
import os
from functools import reduce
from math import sqrt
import tensorflow as tf

# Data Storage Constants
DataUnits = {'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024 ** 3}


# Print the current time along with a string
def msgtime(prefix=''):
    print(prefix + str(datetime.now()))


# Returns the current memory usage by the python process
def str_memusage(datatype):
    process = psutil.Process(os.getpid())
    assert datatype in DataUnits, 'Datatype must be in [B,M,G]'
    return str(process.memory_info().rss / DataUnits[datatype]) + datatype


# Progress Bar Printing from
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def print_prog_bar(iteration, total, prefix='', suffix='',
                   decimals=2, length=90, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    strdec = str(decimals)
    percent = ("{0:." + strdec + "f}").format(100 *
                                              (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()


# Fully Connected Network Information Printer
# Prints the stats given the network definition
# Net def is a list of numbers
def fcn_stats(net_def):
    num_l = len(net_def)
    num_hl = num_l - 2
    zip_weights = zip(net_def[0:-1], net_def[1:])
    layer_weights = list(map(lambda zi: zi[0] * zi[1], zip_weights))
    num_weights = reduce(operator.add, layer_weights)
    num_biases = reduce(operator.add, net_def[1:])
    tot_dim = num_weights + num_biases
    print('#*******NET STATS*******#')
    print('Layers\t\t\t:', num_l)
    print('Hidden\t\t\t:', num_hl)
    print('Weight Dims\t\t:', num_weights)
    print('Bias Dims\t\t:', num_biases)
    print('Total Dims\t\t:', tot_dim)


# Calculate
def chical(c1, c2):
    psi = c1 + c2
    chi = abs(2.0 / (2.0 - psi - sqrt(psi * psi - 4.0 * psi)))
    return chi

# Activation Function
def activate(input_layer, act='relu', name='activation'):
    if act is None:
        return input_layer
    if act == 'relu':
        return tf.nn.relu(input_layer, name)
    if act == 'sqr':
        return tf.square(input_layer, name)
    if act == 'sqr_sigmoid':
        return tf.nn.sigmoid(tf.square(input_layer, name))
    if act == 'sigmoid':
        return tf.nn.sigmoid(input_layer, name)


# Fully connected custom layer for PSO
# Supported activation function types : None,relu,sqr,sqr_sigmoid,sigmoid
def fc(input_tensor, n_output_units, scope,
       activation_fn='relu', uniform=False):
    shape = [input_tensor.get_shape().as_list()[-1], n_output_units]
    # Use the Scope specified
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Init Weights
        if uniform:
            weights = tf.Variable(tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-10,
                maxval=10),
                name='weights')
        else:
            weights = tf.Variable(tf.truncated_normal(
                shape=shape,
                mean=0.0,
                stddev=0.1,
                dtype=tf.float32),
                name='weights')
        # Init Biases
        biases = tf.Variable(
            tf.zeros(shape=[n_output_units]), name='biases', dtype=tf.float32)
        # Particle Best
        pbest_w = tf.get_variable(
            shape=shape,
            name='pbest_w',
            initializer=tf.random_uniform_initializer)
        pbest_b = tf.get_variable(
            shape=[n_output_units],
            name='pbest_b',
            initializer=tf.random_uniform_initializer)

        # Velocities
        vel_weights = tf.Variable(tf.random_uniform(
            shape=shape,
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_weights')
        vel_biases = tf.Variable(tf.random_uniform(
            shape=[n_output_units],
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_biases')

        # Perform actual feedforward
        act = tf.matmul(input_tensor, weights) + biases
        pso_tupple = [weights, biases,
                      pbest_w, pbest_b,
                      vel_weights, vel_biases]
        # Activate And Return
        return activate(act, activation_fn), pso_tupple


# Magnitude Clipper
# Magmax can be either a Tensor or a Float
def maxclip(tensor, magmax):
    # assertion commented out to allow usage of both Tensor & Integer
    # assert magmax > 0, "magmax argument in maxclip must be positive"
    return tf.minimum(tf.maximum(tensor, -magmax), magmax)
