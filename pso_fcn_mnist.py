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

# MULTI-PARTICLE NEURAL NETS

losses = []
nets = []
pweights = []
pbiases = []
vweights = []
vbiases = []

random_values = []

# Positional Updates
bias_updates = []
weight_updates = []

# Velocity Updates
vweight_updates = []
vbias_updates = []

# Fitness Updates
fit_updates = []


# Control Updates - Controling PSO inside tf.Graph
control_updates = []

# Hybrid Updates - Using of PSO + Traditional Approaches
hybrid_updates = []

gweights = None
gbiases = None
gfit = None

if not LBPSO:
    gweights = []
    gbiases = []
    gfit = tf.Variable(float("inf"), name='gbestfit', trainable=False)

# TODO:Parellized the following loop
# TODO:See if the Conditional Function Lambdas can be optimized

print('before fcn_stats')
fcn_stats(LAYERS)
print('after fcn_stats')
for pno in range(N_PARTICLES):
    weights = []
    biases = []
    pweights = []
    pbiases = []
    lweights = None
    lbiases = None
    if LBPSO:
        # Initialize the list
        lweights = []
        lbiases = []
    pbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=P_BEST_FACTOR),
        name='pno' + str(pno + 1) + 'pbestrand',
        trainable=False)
    gbestrand = None
    lbestrand = None
    if not LBPSO:
        gbestrand = tf.Variable(tf.random_uniform(
            shape=[], maxval=G_BEST_FACTOR),
            name='pno' + str(pno + 1) + 'gbestrand',
            trainable=False)
    else:
        lbestrand = tf.Variable(tf.random_uniform(
            shape=[], maxval=L_BEST_FACTOR),
            name='pno' + str(pno + 1) + 'lbestrand',
            trainable=False)

    # Append the random values so that the initializer can be called again
    random_values.append(pbestrand)
    if not LBPSO:
        random_values.append(gbestrand)
    else:
        random_values.append(lbestrand)
    pfit = None
    with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
        init = tf.constant(float("inf"))
        pfit = tf.get_variable(name=str(pno + 1),
                               initializer=init)

    pfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'fit')

    localfit = None
    if LBPSO:
        localfit = tf.Variable(float("inf"), name='pno' + str(pno + 1) + 'lfit')
    net = net_in
    # Define the parameters

    for idx, num_neuron in enumerate(LAYERS[1:]):
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        net, pso_tupple = fc(input_tensor=net,
                             n_output_units=num_neuron,
                             activation_fn='softmax', # 'sigmoid',
                             scope=layer_scope,
                             uniform=True)
        w, b, pw, pb, vw, vb = pso_tupple
        vweights.append(vw)
        vbiases.append(vb)
        weights.append(w)
        biases.append(b)
        pweights.append(pw)
        pbiases.append(pb)
        lw = None
        lb = None
        if LBPSO:
            lw = tf.Variable(pw.initialized_value(), name='lbest_w')
            lb = tf.Variable(pb.initialized_value(), name='lbest_b')
            lbiases.append(lb)
            lweights.append(lw)

        # Multiply by the Velocity Decay
        nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
        nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

        # Differences between Particle Best & Current
        pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
        pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)

        # Differences between the Local Best & Current
        ldiffw = None
        ldiffb = None
        if LBPSO:
            ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
            ldiffb = tf.multiply(tf.subtract(lb, w), lbestrand)

        # Define & Reuse the GBest
        gw = None
        gb = None
        if not LBPSO:
            with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
                gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                     shape=[LAYERS[idx], LAYERS[idx + 1]],
                                     initializer=tf.zeros_initializer)

                gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                     shape=[LAYERS[idx + 1]],
                                     initializer=tf.zeros_initializer)

        # If first Particle add to Global Else it is already present
        if pno == 0 and not LBPSO:
            gweights.append(gw)
            gbiases.append(gb)
        gdiffw = None
        gdiffb = None
        # Differences between Global Best & Current
        if not LBPSO:
            gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
            gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)
        else:
            ldiffw = tf.multiply(tf.subtract(lw, w), lbestrand)
            ldiffb = tf.multiply(tf.subtract(lb, b), lbestrand)

        vweightdiffsum = None
        vbiasdiffsum = None
        if LBPSO:
            vweightdiffsum = tf.multiply(
                tf.add_n([nextvw, pdiffw, ldiffw]),
                CHI)
            vbiasdiffsum = tf.multiply(tf.add_n([nextvb, pdiffb, ldiffb]), CHI)
        else:
            vweightdiffsum = tf.add_n([nextvw, pdiffw, gdiffw])
            vbiasdiffsum = tf.add_n([nextvb, pdiffb, gdiffb])

        vweight_update = None
        if VELOCITY_RESTRICT is False:
            vweight_update = tf.assign(vw, vweightdiffsum, validate_shape=True)
        else:
            vweight_update = tf.assign(vw, maxclip(vweightdiffsum, t_MVEL),
                                       validate_shape=True)

        vweight_updates.append(vweight_update)
        vbias_update = None
        if VELOCITY_RESTRICT is False:
            vbias_update = tf.assign(vb, vbiasdiffsum, validate_shape=True)
        else:
            vbias_update = tf.assign(vb, maxclip(vbiasdiffsum, t_MVEL),
                                     validate_shape=True)

        vbias_updates.append(vbias_update)
        weight_update = tf.assign(w, w + vw, validate_shape=True)
        weight_updates.append(weight_update)
        bias_update = tf.assign(b, b + vb, validate_shape=True)
        bias_updates.append(bias_update)

    # Define loss for each of the particle nets
    loss = tf.nn.l2_loss(net - label)
    # loss = tf.reduce_mean(-tf.reduce_sum(label*tf.log(net), reduction_indices=[1]))
    particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
    fit_update = tf.assign(pfit, particlebest, validate_shape=True)
    fit_updates.append(fit_update)
    if not LBPSO:
        globalbest = tf.cond(loss < gfit, lambda: loss, lambda: gfit)
        fit_update = tf.assign(gfit, globalbest, validate_shape=True)
        fit_updates.append(fit_update)
    control_update = tf.assign(t_MVEL, tf.multiply(t_MVEL, MAX_VEL_DECAY),
                               validate_shape=True)
    control_updates.append(control_update)
    if HYBRID:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        hybrid_update = optimizer.minimize(loss)
        hybrid_updates.append(hybrid_update)

    # Multiple Length Checks
    assert len(weights) == len(biases)
    assert len(pweights) == len(pbiases)
    assert len(pweights) == len(weights)

    for i in range(len(weights)):
        # Particle Best
        pweight = tf.cond(loss <= pfit, lambda: weights[
                          i], lambda: pweights[i])
        fit_update = tf.assign(pweights[i], pweight, validate_shape=True)
        fit_updates.append(fit_update)
        pbias = tf.cond(loss <= pfit, lambda: biases[i], lambda: pbiases[i])
        fit_update = tf.assign(pbiases[i], pbias, validate_shape=True)
        fit_updates.append(fit_update)

        if LBPSO:
            lneigh = (pno - 1) % N_PARTICLES
            rneigh = (pno + 1) % N_PARTICLES
            lneighscope = 'pno' + str(lneigh + 1) + 'fc' + str(i + 1)
            rneighscope = 'pno' + str(rneigh + 1) + 'fc' + str(i + 1)
            lneigh_weight = None
            lneigh_bias = None
            rneigh_weight = None
            rneigh_bias = None
            lfit = None
            rfit = None

            with tf.variable_scope(lneighscope, reuse=tf.AUTO_REUSE):
                lneigh_weight = tf.get_variable(
                    shape=[LAYERS[i], LAYERS[i + 1]],
                    name='pbest_w',
                    initializer=tf.random_uniform_initializer)
                # [LAYERS[idx + 1]]
                lneigh_bias = tf.get_variable(
                    shape=[LAYERS[i + 1]],
                    name='pbest_b',
                    initializer=tf.random_uniform_initializer)
            with tf.variable_scope(rneighscope, reuse=tf.AUTO_REUSE):
                rneigh_weight = tf.get_variable(
                    shape=[LAYERS[i], LAYERS[i + 1]],
                    name='pbest_w',
                    initializer=tf.random_uniform_initializer)
                # [LAYERS[idx + 1]]
                rneigh_bias = tf.get_variable(
                    shape=[LAYERS[i + 1]],
                    name='pbest_b',
                    initializer=tf.random_uniform_initializer)

            with tf.variable_scope("fitnessvals", reuse=tf.AUTO_REUSE):
                init = tf.constant(float("inf"))
                lfit = tf.get_variable(name=str(lneigh + 1), initializer=init)
                rfit = tf.get_variable(name=str(rneigh + 1), initializer=init)

            new_local_weight = None
            new_local_bias = None
            new_local_fit = None

            # Deal with Local Fitness
            neighbor_best_fit = tf.cond(lfit <= rfit,
                                        lambda: lfit, lambda: rfit)
            particle_best_fit = tf.cond(pfit <= localfit,
                                        lambda: pfit, lambda: localfit)
            best_fit = tf.cond(neighbor_best_fit <= particle_best_fit,
                               lambda: neighbor_best_fit,
                               lambda: particle_best_fit)
            fit_update = tf.assign(localfit, best_fit, validate_shape=True)
            fit_updates.append(fit_update)

            # Deal with Local Best Weights
            neighbor_best_weight = tf.cond(lfit <= rfit,
                                           lambda: lneigh_weight,
                                           lambda: rneigh_weight)
            particle_best_weight = tf.cond(pfit <= localfit,
                                           lambda: pweights[i],
                                           lambda: lweights[i])
            best_weight = tf.cond(neighbor_best_fit <= particle_best_fit,
                                  lambda: neighbor_best_weight,
                                  lambda: particle_best_weight)
            fit_update = tf.assign(
                lweights[i], best_weight, validate_shape=True)
            fit_updates.append(fit_update)

            # Deal with Local Best Biases
            neighbor_best_bias = tf.cond(lfit <= rfit,
                                         lambda: lneigh_bias,
                                         lambda: rneigh_bias)
            particle_best_bias = tf.cond(pfit <= localfit,
                                         lambda: pbiases[i],
                                         lambda: lbiases[i])
            best_bias = tf.cond(neighbor_best_fit <= particle_best_fit,
                                lambda: neighbor_best_bias,
                                lambda: particle_best_bias)
            fit_update = tf.assign(lbiases[i], best_bias, validate_shape=True)
            fit_updates.append(fit_update)

        if not LBPSO:
            # Global Best
            gweight = tf.cond(loss <= gfit,
                              lambda: weights[i],
                              lambda: gweights[i])
            fit_update = tf.assign(gweights[i], gweight, validate_shape=True)
            fit_updates.append(fit_update)
            gbias = tf.cond(loss <= gfit,
                            lambda: biases[i],
                            lambda: gbiases[i])
            fit_update = tf.assign(gbiases[i], gbias, validate_shape=True)
            fit_updates.append(fit_update)

    # Update the lists
    nets.append(net)
    losses.append(loss)
    print_prog_bar(iteration=pno + 1,
                   total=N_PARTICLES,
                   suffix=str_memusage('M'))
print(nets)

msgtime('Completed\t\t:')

_var_list = (
    tf.trainable_variables() +
    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
# pylint: disable=protected-access
_var_list += tf.get_collection(tf.GraphKeys._STREAMING_MODEL_PORTS)
print(len(_var_list))
for var in _var_list:
    print(var)
# print(_var_list)

# Initialize the entire graph
init = tf.global_variables_initializer()
msgtime('Graph Init Successful\t:')


'''
List of all the variables
for var in tf.global_variables():
    print(var)
'''

# Define the updates which are to be done before each iterations
random_updates = [r.initializer for r in random_values]
updates = weight_updates + bias_updates + \
    random_updates + vbias_updates + vweight_updates + \
    fit_updates + control_updates + hybrid_updates
req_list = None
if not LBPSO:
    req_list = losses, updates, gfit, gbiases, vweights, vbiases, gweights
else:
    req_list = losses, updates, vweights, vbiases


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
        _tuple = sess.run(req_list, feed_dict={
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
            for net in nets:
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