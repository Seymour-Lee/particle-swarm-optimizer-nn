# -*- coding: UTF-8 -*-

# NOTE: Local Best Version Under Development to be integrated with clinn.py
import itertools
from functools import reduce
import operator
import time
import random
import math
import argparse
from tensorflow.contrib.pso.python.utils import msgtime, str_memusage, print_prog_bar, fcn_stats, chical, maxclip, fc
from tensorflow.contrib.pso.python.pso import ParticleSwarmOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import xlwt

book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('pso', cell_overwrite_ok=True)
sheet.write(0, 0, 'step')
sheet.write(0, 1, 'loss')
sheet.write(0, 2, 'accu')

def compute_accuracy(v_x, v_y):
    global prediction
    #input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x:v_x})
    #find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    #calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get input content
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
    return result

def add_layer(inputs, in_size, out_size, activation_function=None,):
    #init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    #calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
    
def train_mnist():
    #load mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #define placeholder for input
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    #add layer
    prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax)
    #calculate the loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), reduction_indices=[1]))
    #use Gradientdescentoptimizer
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    #init session
    sess = tf.Session()
    #init all variables
    sess.run(tf.global_variables_initializer())
    #start training
    for i in range(4500):
        #get batch to learn easily
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))

# train_mnist()

# Suppress Unecessary Warnings
tf.logging.set_verbosity(tf.logging.ERROR)


# Function to Build the Parser for CLI
def build_parser():
    parser = argparse.ArgumentParser(description='CLI Utility for NNPSO')

    # Dataset Generation Parameters
    parser.add_argument('--bs', type=int, default=32,
                        help='batchsize', metavar='N_BATCHSIZE')
    parser.add_argument('--xorn', type=int, default=784,
                        help='Number of XOR Inputs', metavar='N_IN')

    # PSO Parameters
    parser.add_argument('--pno', type=int, default=32,
                        help='number of particles', metavar='N_PARTICLES')
    parser.add_argument('--gbest', type=float, default=0.8,
                        help='global best for PSO', metavar='G_BEST_FACTOR')
    parser.add_argument('--lbest', type=float, default=0.7,
                        help='local best for PSO', metavar='L_BEST_FACTOR')
    parser.add_argument('--pbest', type=float, default=0.6,
                        help='local best for PSO', metavar='P_BEST_FACTOR')
    parser.add_argument('--veldec', type=float, default=1,
                        help='Decay in velocity after each position update',
                        metavar='VELOCITY_DECAY')
    parser.add_argument('--vr', action='store_true',
                        help='Restrict the Particle Velocity')
    parser.add_argument('--mv', type=float, default=0.005,
                        help='Maximum velocity for a particle if restricted',
                        metavar='MAX_VEL')
    parser.add_argument('--mvdec', type=float, default=1,
                        help='Multiplier for Max Velocity with each update',
                        metavar='MAX_VEL_DECAY')
    # Hyrid Parmeters
    parser.add_argument('--hybrid', action='store_true',
                        help='Use Adam along with PSO')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning Rate if Hybrid Approach',
                        metavar='LEARNING_RATE')
    parser.add_argument('--lbpso', action='store_true',
                        help='Using Local Best Variant of PSO')

    # Other Parameters
    parser.add_argument('--iter', type=int, default=int(5000),
                        help='number of iterations', metavar='N_INTERATIONS')
    parser.add_argument('--hl', nargs='+', type=int,
                        help='hiddenlayers for the network', default=[10])# default=[3, 2])

    parser.add_argument('--pi', type=int, default=100,
                        help='Nos iteration for result printing',
                        metavar='N_BATCHSIZE')

    return parser


# Xorgenerator Function
def xor_next_batch(batch_size, n_input):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        x = []
        y = []
        ans = 0
        for j in range(n_input):
            x.append(random.randint(0, 1))
            ans ^= x[j]
        y.append(ans)
        batch_y.append(y)
        batch_x.append(x)
    return batch_x, batch_y


# TODO : Add Printing Control

msgtime('Script Launched\t\t:')
msgtime('Building Parser\t\t:')
parser = build_parser()
msgtime('Parser Built\t\t:')
msgtime('Parsing Arguments\t:')
args = parser.parse_args()
msgtime('Arguments Parsed\t:')
print('Arguments Obtained\t:', vars(args))

# XOR Dataset Params
N_IN = args.xorn
N_BATCHSIZE = args.bs


# PSO params
N_PARTICLES = args.pno
P_BEST_FACTOR = args.pbest
G_BEST_FACTOR = args.gbest
L_BEST_FACTOR = args.lbest
# Velocity Decay specifies the multiplier for the velocity update
VELOCITY_DECAY = args.veldec
# Velocity Restrict is computationally slightly more expensive
VELOCITY_RESTRICT = args.vr
MAX_VEL = args.mv
# Allows to decay the maximum velocity with each update
# Useful if the network needs very fine tuning towards the end
MAX_VEL_DECAY = args.mvdec

# Hybrid Parameters
HYBRID = args.hybrid
LEARNING_RATE = args.lr
LBPSO = args.lbpso


# Other Params
N_ITERATIONS = args.iter
HIDDEN_LAYERS = args.hl
PRINT_ITER = args.pi

# Chi cannot be used for low value of pbest & lbest factors
# CHI = chical(P_BEST_FACTOR, L_BEST_FACTOR)
CHI = 1  # Temporary Fix


# Basic Neural Network Definition
# Simple feedforward Network
LAYERS = [N_IN] + HIDDEN_LAYERS # + [1]
print('Network Structure\t:', LAYERS)


t_VELOCITY_DECAY = tf.constant(value=VELOCITY_DECAY,
                               dtype=tf.float32,
                               name='vel_decay')
t_MVEL = tf.Variable(MAX_VEL,
                     dtype=tf.float32,
                     name='vel_restrict',
                     trainable=False)


# A list of lists having N_IN elements all either 0 or 1
# xor_in = [list(i) for i in itertools.product([0, 1], repeat=N_IN)]
# print(xor_in)
# print(len(xor_in))
# # A list having 2^N lists each having xor of each input list in the list
# # of lists
# xor_out = list(map(lambda x: [(reduce(operator.xor, x))], xor_in))
# print(xor_out)
# print(len(xor_out))

net_in = tf.placeholder(dtype=tf.float32,
                        shape=[None, N_IN],
                        # shape=[N_BATCHSIZE, N_IN],
                        name='net_in')

label = tf.placeholder(dtype=tf.float32,
                       shape=[None, 10],
                       # shape=[N_BATCHSIZE, 1],
                       name='net_label')

print('Mem Usage\t\t:', str_memusage(datatype='M'))
msgtime('Building Network\t:')

optimizer = ParticleSwarmOptimizer(xorn=N_IN,
                                    bs=N_BATCHSIZE,
                                    pno=N_PARTICLES,
                                    pbest=P_BEST_FACTOR,
                                    gbest=G_BEST_FACTOR,
                                    lbest=L_BEST_FACTOR,
                                    veldec=VELOCITY_DECAY,
                                    vr=VELOCITY_RESTRICT,
                                    mv=MAX_VEL,
                                    mvdec=MAX_VEL_DECAY,
                                    hybrid=HYBRID,
                                    lr=LEARNING_RATE,
                                    lbpso=LBPSO,
                                    iters=N_ITERATIONS,
                                    hl=HIDDEN_LAYERS,
                                    pi=PRINT_ITER)

fcn_stats(LAYERS)
print('after fcn_stats')

train_step, test_step = optimizer.minimize(net_in = net_in,
                              label = label)

init = tf.global_variables_initializer()

mnist_pso = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist_pso.test.labels)
with tf.Session() as sess:
    sess.run(init)

    # Write The graph summary
    summary_writer = tf.summary.FileWriter('/tmp/tf/logs', sess.graph_def)
    start_time = time.time()
    for i in range(N_ITERATIONS):
        # Reinitialize the Random Values at each iteration

        # xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
        batch_x, batch_y = mnist_pso.train.next_batch(100)
        _tuple = sess.run(train_step, feed_dict={
            net_in: batch_x, label: batch_y})
        _losses = None
        if not LBPSO:
            _losses, _, gfit, gbiases, vweights, vbiases, gweights = _tuple
        else:
            _losses, _, vweights, vbiases = _tuple
        if ((i + 1) % PRINT_ITER == 0) or (i == 0):
            print('Losses:', _losses, 'Iteration:', i+1)
            if not LBPSO:
                print('Gfit:', gfit)
            else:
                print('Best Particle', min(_losses))
            v_x = mnist_pso.test.images[:300]
            v_y = mnist_pso.test.labels[:300]
            results = []
            for net in test_step:
                y_pre = sess.run(net, feed_dict={net_in: v_x})
                correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
                #calculate average
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #get input content
                result = sess.run(accuracy,feed_dict={net_in: v_x, label: v_y})
                results.append(result)
            print(results)
            print('Best Accuracy: ', max(results))
            sheet.write((i + 1) / 100 + 1, 0, str(i + 1))
            sheet.write((i + 1) / 100 + 1, 1, str(gfit))
            sheet.write((i + 1) / 100 + 1, 2, str(max(results)))
    end_time = time.time()
    # Close the writer
    summary_writer.close()

    print('Total Time:', end_time - start_time)

book.save('pso_' + str(LEARNING_RATE) + '_' + str(MAX_VEL_DECAY) + '.xls')