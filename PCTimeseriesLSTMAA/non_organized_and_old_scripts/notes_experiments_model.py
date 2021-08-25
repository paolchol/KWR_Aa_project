# -*- coding: utf-8 -*-
"""
Notes and experiments on model building

    - Basic RNNs in Tensor Flow:
        From Géron, 2017. Old version of TensorFlow
    - 

@author: colompa
"""

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
#Disable the default activate eager execution
tf_v1.disable_eager_execution()

#%% Basic RNNs in TensorFlow
#From Geron, 2017

#%% "Raw" model
# Model definition
#This model will run over two time steps: X0 and X1, and will take input vectors
#of size 3 at each time step
n_inputs = 3
n_neurons = 5

X0 = tf_v1.placeholder(dtype = tf_v1.float32, shape = [None, n_inputs])
X1 = tf_v1.placeholder(tf_v1.float32, [None, n_inputs])

Wx = tf.Variable(tf_v1.random_normal(shape = [n_inputs, n_neurons]), dtype = tf_v1.float32)
Wy = tf.Variable(tf_v1.random_normal(shape = [n_neurons, n_neurons]), dtype = tf_v1.float32)
b = tf.Variable(tf.zeros([1, n_neurons]), dtype = tf_v1.float32)

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf_v1.global_variables_initializer()

# Model running
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf_v1.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
#Y0_val and Y1_val are the output at timesteps t0 and t1

#%% Model with Static Unrolling through time
n_steps = 2
n_inputs = 3
X = tf_v1.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf_v1.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf_v1.nn.static_rnn(basic_cell, X_seqs, dtype = tf_v1.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([ # t = 0 t = 1
[[0, 1, 2], [9, 8, 7]], # instance 0
[[3, 4, 5], [0, 0, 0]], # instance 1
[[6, 7, 8], [6, 5, 4]], # instance 2
[[9, 0, 1], [3, 2, 1]], # instance 3
])

with tf_v1.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict = {X: X_batch})

#This approach still builds a graph containing one cell per time step. This
#can lead to out-of-memory errors during backpropagation

#%% Dynamic Unrolling through time
n_steps = 2
n_inputs = 3
n_neurons = 5
X = tf_v1.placeholder(tf_v1.float32, [None, n_steps, n_inputs])
basic_cell = tf_v1.nn.rnn_cell.BasicRNNCell(num_units = n_neurons)
outputs, states = tf_v1.nn.dynamic_rnn(basic_cell, X, dtype = tf_v1.float32)

#WARNING:
   # `tf.nn.rnn_cell.BasicRNNCell` is deprecated and will be removed in a future version.
   # This class is equivalent as `tf.keras.layers.SimpleRNNCell`, and will be replaced by that in Tensorflow 2.0.

#%% Training to predict time series

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

# WRAPPED CELL
#Having 100 neurons, we will have an output vector of size 100 at each time step
#What we want is a single output value at each time step. So, an OutputProjectionWrapper can be used.
#It will apply a Fully Connected layer on top of each output, not affecting the cell state
#The cell will be then defined as below

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu),
    output_size = n_outputs)


#Although using an OutputProjectionWrapper is the simplest solution to reduce
#the dimensionality of the RNN’s output sequences down to just one value per
#time step (per instance),
#Other solution: (figure 14-10)

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

#So:
    ## Structure definition
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1
X = tf_v1.placeholder(tf_v1.float32, [None, n_steps, n_inputs])
y = tf_v1.placeholder(tf_v1.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    ## Training definition
learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))   #cost function, Mean Square Error (MSE)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #Adam optimizer
training_op = optimizer.minimize(loss)  #training op
init = tf.global_variables_initializer()    #variable optimization op
    ## Execution
n_iterations = 10000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = [...] # fetch the next training batch
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    ## Prediction (once the model is trained)
X_new = [...] # New sequences
y_pred = sess.run(outputs, feed_dict={X: X_new})

#The placeholder X acts as the input layer. During the execution phase, it will
#be replaced with one training batch at a time. All the instances in the training batch
# will be processed simultaneosly by the neural network

#%% Deep RNN
#A deep RNN is a RNN with multiple layers of cells

n_neurons = 100
n_layers = 3
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

#%% LSTM cell

#Use this instead of the BasicRNNCell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

#Add Peephole Connection
#This provides the previous long-term state to the controllers of the forget and input gates
#as an input, and the current long-term state is added as input to the controller of the output gate
lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)

#GRU cell
gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)



