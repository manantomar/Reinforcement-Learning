
"""
    The network is designed in order to distill expert
    policies which are trained on model based prediction. Therefore, to be
    consistent with the paper, the AMN comprises of the same architecture
    as the individual policies.
"""

import tensorflow as tf
import numpy as np
import random


class net_v4:

    # initiate tensorboard summaries
    def __init__(self, discount, clip_delta, batch, lambda_reg):

        self.discount = discount
        self.clip_delta = clip_delta
        self.batch = batch
        self.lambda_reg = lambda_reg

        #num_actions = tf.placeholder("uint8", ())
        self.Q_val, self.f_pred, self.f_true = self.build_net(6)

        self.next_Q_val = tf.placeholder("float", [self.batch, 6])
        #self.action = tf.placeholder("float", [self.batch, 4])
        self.reward = tf.placeholder("float", [self.batch, ])
        self.done = tf.placeholder("float", [self.batch, ])

        target = self.reward + self.discount * \
        tf.to_float((np.ones_like(self.done) - self.done)) * tf.reduce_max(self.next_Q_val, axis=1, keep_dims=True)

        # not yet clear what it does actually
        action_mask = np.equal(tf.reshape(np.arange(16), [1,-1]), tf.reshape(self.action[:,0], [-1,1]))
        out = tf.reshape(tf.reduce_sum((self.Q_val*action_mask), 1), [-1,1])

        self.diff = target - out
        self.diff_reg = self.f_true - self.f_pred

        if self.clip_delta > 0:
            quadratic_part = tf.minimum(abs(self.diff), self.clip_delta)
            linear_part = abs(self.diff) - quadratic_part
            self.loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            self.loss = 0.5 * self.diff ** 2

        self.loss += tf.reduce_sum(0.5 * self.lambda_reg * (self.diff_reg ** 2), 1)
        self.loss = tf.reduce_sum(self.loss)
        tf.summary.scalar("loss", self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate = 0.00025)
        self.train_step = optimizer.minimize(self.loss)
        # sample an action from

    def q_val(self, state):

        return self.Q_val.eval(feed_dict = {self.state : state})

    def train(self, state, action, reward, done, merged_summary_op):

        state_padded = np.zeros((state.shape[0], state.shape[1]+1, state.shape[2], state.shape[3]))
        state_padded[:,:-1] = state

        next_Q_val = self.Q_val.eval(feed_dict = {self.state : state_padded[:, 1:]})

        self.train_step.run(feed_dict = {
                self.state : state_padded[:, :-1],
                self.action : action,
                self.next_Q_val : next_Q_val,
                self.reward : reward,
                self.done : done})

        cost = self.loss.eval(feed_dict = {
                self.state : state,
                self.action : action,
                self.next_Q_val : next_Q_val,
                self.reward : reward,
                self.done : done})
        diff_reg = self.diff_reg.eval(feed_dict = {
                self.state : state,
                self.action : action,
                self.next_Q_val : next_Q_val,
                self.reward : reward,
                self.done : done})

        diff = self.diff.eval(feed_dict = {
                self.state : state,
                self.action : action,
                self.next_Q_val : next_Q_val,
                self.reward : reward,
                self.done : done})
        #print("loss", np.sum(0.5 * self.lambda_reg * (diff_reg ** 2)))
        #print("loss diff", np.sum(diff))
        print("loss total", cost)

        summary = merged_summary_op.eval(feed_dict = {
                self.state : state,
                self.action : action,
                self.next_Q_val : next_Q_val,
                self.reward : reward,
                self.done : done})

        return summary

    def weight_variable(self, name, shape):
        initial = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name = name, shape = shape, initializer = initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def build_net(self, num_actions):

        self.state = tf.placeholder("float", [None, 84, 84, 8])
        self.action = tf.placeholder("int32", [None, 4])

        s_in = tf.reshape(self.state, [-1, 84, 84, 4])
        a_in = tf.reshape(self.action, [-1, ])
        # add action embeddings
        embeddings = tf.Variable(tf.random_uniform([num_actions, 256], -1.0, 1.0))

        a_embed = tf.nn.embedding_lookup(embeddings, a_in)
        a_embed_reshaped = tf.reshape(a_embed, [-1, 4*256])

        w1_conv = self.weight_variable("w1_conv", [8, 8, 4, 64])
        b1_conv = self.bias_variable([64])

        # 22 x 22
        w2_conv = self.weight_variable("w2_conv", [4, 4, 64, 64])
        b2_conv = self.bias_variable([64])

        # 10 x 10
        w3_conv = self.weight_variable("w3_conv", [3, 3, 64, 64])
        b3_conv = self.bias_variable([64])

        # 8 x 8
        w_fc1 = self.weight_variable("w_fc1", [3136, 512])
        b_fc1 = self.bias_variable([512])

        conv1 = tf.nn.relu(self.conv2d(s_in, w1_conv, 4) + b1_conv)
        conv2 = tf.nn.relu(self.conv2d(conv1, w2_conv, 2) + b2_conv)
        conv3 = tf.nn.relu(self.conv2d(conv2, w3_conv, 1) + b3_conv)

        print("conv1", conv1.shape)
        print("conv2", conv2.shape)
        print("conv3", conv3.shape)

        conv3_reshaped = tf.reshape(conv3, [-1, 7*7*64])

        fc1 = tf.nn.relu(tf.matmul(conv3_reshaped, w_fc1) + b_fc1)
        fc1_reshaped = tf.reshape(fc1, [-1, 512*2])

        l_curr_true = fc1_reshaped[:,0:512]
        l_next_true = fc1_reshaped[:,512:1024]

        w_fc_act = self.weight_variable("w_fc_act", [256*4, 512])
        b_fc_act = self.bias_variable([512])

        w_fc_curr = self.weight_variable("w_fc_curr", [512, 512])
        b_fc_curr = self.bias_variable([512])

        fc_act = tf.nn.relu(tf.matmul(a_embed_reshaped, w_fc_act) + b_fc_act)
        fc_curr = tf.nn.relu(tf.matmul(l_curr_true, w_fc_curr) + b_fc_curr)

        l_concat = tf.concat([fc_act, fc_curr], 1)

        w_fc_pred = self.weight_variable("w_fc_pred", [1024,512])
        b_fc_pred = self.bias_variable([512])

        fc_next_pred = tf.nn.relu(tf.matmul(l_concat, w_fc_pred) + b_fc_pred)

        w_fc2 = self.weight_variable("w_fc2", [512, num_actions])
        b_fc2 = self.bias_variable([num_actions])

        l_out = tf.nn.relu(tf.matmul(l_curr_true, w_fc2) + b_fc2)

        return l_out, fc_next_pred, l_next_true

    def Qval_to_action(self, Qval):

        #self.Qvalue = Qval
        num = tf.exp(Qval / T)
        policy = num  / tf.reduce_sum(num)

        return policy
