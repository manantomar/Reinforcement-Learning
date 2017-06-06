
"""
    This is an implementation of the Actor Mimic Network described in the
    Actor Critic paper. The network is designed in order to distill expert
    policies which are trained on model based prediction. Therefore, to be
    consistent with the paper, the AMN comprises of the same architecture
    as the individual policies.

    Expert code flow :

    experiment.run --> agent.step --> agent._do_training --> network.train --> which basically calls a minibatch update
    |                                           |  --->  combine all these in one file which describes the environment, samples (probably another file)
    --------------------------------------------         and calls training updates, referencing to a train function in AMN class
"""

import tensorflow as tf
import numpy as np
import random

class AMN:

    # initiate tensorboard summaries
    def __init__(self, ):

        self.Q_val = self.build_net(num_actions)

        # load weights for expert 1, 2, 3...
        # one hot encoded vector
        teacher = tf.placeholder("float", [BATCH, num_actions])

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(teacher, self.Q_val))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
        # sample an action from

    def train_step(true_policy, sampled_state, sampled_action):

        train_step.run(feed_dict = {
                teacher : true_policy,
                state : sampled_state,
                action : sampled_action})

        loss = cost.eval(feed_dict = {
                teacher : true_policy,
                state : sampled_state,
                action : sampled_action})

    def weight_variable(name, shape):
        initial = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape, initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def build_net(self, num_actions):

        state = tf.placeholder("float", [BATCH, 8, 84, 84])
        action = tf.placeholder("float", [BATCH, 4])

        s_in = tf.reshape(state, [-1, 4, 84, 84])
        a_in = tf.reshape(action, [-1])
        # add action embeddings

        #w1_embed = weight_variable("w1_embed", [-1, ])

        w1_conv = weight_variable("w1_conv", [4, 4, 4, 64])
        b1_conv = bias_variable([64])

        # 22 x 22
        w2_conv = weight_variable("w2_conv", [4, 4, 64, 64])
        b2_conv = bias_variable([64])

        # 10 x 10
        w3_conv = weight_variable("w3_conv", [3, 3, 64, 64])
        b3_conv = bias_variable([64])

        # 8 x 8
        w_fc1 = weight_variable("w_fc1", [8, 512])
        b_fc1 = bias_variable([512])

        conv1 = tf.nn.relu(conv2d(s_in, w1_conv, 4) + b1_conv)
        conv2 = tf.nn.relu(conv2d(conv1, w2_conv, 2) + b2_conv)
        conv3 = tf.nn.relu(conv2d(conv2, w3_conv, 4) + b3_conv)

        conv3_reshaped = tf.reshape(conv3, [-1, _])

        fc1 = tf.nn.relu(tf.matmul(conv3_reshaped, w_fc1) + b_fc1)
        fc1_reshaped = tf.reshape(fc1, [-1, 512*2])

        latent_curr_true = fc1_reshaped[:,0:512]
        latent_next_true = fc1_reshaped[:,512:1024]

        w_fc2 = weight_variable("w_fc2", [512, num_actions])
        b_fc2 = bias_variable([num_actions])

        l_out = tf.nn.relu(tf.matmul(latent_curr_true, w_fc2) + b_fc2)

        return l_out

    def Qval_to_action(self, Qval):

        #self.Qvalue = Qval
        num = tf.exp(Qval / T)
        policy = num  / tf.reduce_sum(num)

        return policy
