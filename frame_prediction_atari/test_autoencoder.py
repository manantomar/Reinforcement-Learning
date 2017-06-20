from __future__ import print_function

''' We use downsampled gray scale images - 84 X 84,
    consider only every 4th frame as input, applying
    the same action for the intermediate frames.
    Minibatch size is taken to be 32. Each input
    consists of a fixed memory of T = 4 to unroll
    each trajectory and pass in as an input. K, which
    is the prediction step parameter, taken to be 1'''

''' latest model is stored at /Downloads/models3/ '''

import sys
import gym
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import random
from collections import deque
import cv2

flags = tf.app.flags
flags.DEFINE_boolean('train', True, 'Whether to do training or testing')
flags.DEFINE_string('env_name', 'MsPacman-v0', 'The name of gym environment to use')

env = gym.make(flags.env_name)

epsilon = 0.35
MAX_EPISODES = 10000
BATCH = 32 # change to 1 while predicting
max_iter = 10000
ACTIONS = env.action_space.n
FACTORS = 2048
REPLAY_MEMORY = 1000000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def conv2d_nopad(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def deconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = "SAME")

def deconv2d_nopad(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = "VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class autoencoder():

    def __init__(self, ):

        self.pred_frame = self.build_encoder()
        self.y = tf.placeholder("float", [BATCH, 84, 84])
        self.loss = tf.square(tf.norm(self.y - self.pred_frame))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        self.summaries = tf.summary.merge_all(tf.summary.scalar("loss", self.loss))

    def build_encoder(self, ):

        # input - Batch X 84 X 84 X 4
        self.state = tf.placeholder("float", [None, 84, 84, 4])
        self.action = tf.placeholder("float", [None, ACTIONS])

        # 6 X 6 X 4 x 64 - stride 2
        W_conv1 = weight_variable([6, 6, 4, 64])
        wconv = tf.get_variable("wconv", shape=[6, 6, 4, 64], initializer=tf.contrib.layers.xavier_initializer())
        b_conv1 = bias_variable([64])

        # 6 X 6 X 64 x 64 - stride 2
        W_conv2 = weight_variable([6, 6, 64, 64])
        b_conv2 = bias_variable([64])

        # 6 X 6 X 64 x 64 - stride 2
        W_conv3 = weight_variable([6, 6, 64, 64])
        b_conv3 = bias_variable([64])

        # _*16 ie. flattened output from conv3
        W_fc1 = weight_variable([10*10*64, 1024])
        b_fc1 = bias_variable([1024])

        #second fully connected layer - 2048 units
        W_fc2 = weight_variable([1024, 2048])
        b_fc2 = bias_variable([2048])

        #W_fc2 = weight_variable([256, ACTIONS])
        #b_fc2 = bias_variable([ACTIONS])

        conv1 = tf.nn.relu(conv2d_nopad(self.state, wconv, 2) + b_conv1)
        #padded_conv1 = tf.pad(conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        #print("padded shape", padded_conv1.shape)

        conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)
        #padded_conv2 = tf.pad(conv2, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

        conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 2) + b_conv3)

        conv3_flat = tf.reshape(conv3, [-1, 10*10*64])
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)
        fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)

        # 6 X 6 X 4 x 64 - stride 2
        W_enc = weight_variable([FACTORS, 2048])
        W_dec = weight_variable([2048, FACTORS])
        W_action = weight_variable([FACTORS, ACTIONS])
        b_interactions = bias_variable([2048])

        #W_henc = tf.matmul(W_enc, fc2)
        #W_a = tf.matmul(W_action, action)
        #fc_interactions = tf.matmul(W_dec, tf.multiply(W_henc, W_a)) + b_interactions

        W_henc = tf.matmul(fc2, tf.transpose(W_enc))
        W_a = tf.matmul(self.action, tf.transpose(W_action))
        fc_interactions = tf.matmul(tf.multiply(W_henc, W_a), tf.transpose(W_dec)) + b_interactions

        # first fully connected layer after multiplicative interaction- 2048
        W_fc3 = weight_variable([2048, 1024])
        b_fc3 = bias_variable([1024])

        # second fully connected layer after multiplicative interaction- 1024 units
        W_fc4 = weight_variable([1024, 10*10*64])
        b_fc4 = bias_variable([10*10*64])

        #fc3 = tf.nn.relu(tf.matmul(fc_interactions, W_fc3) + b_fc3)
        # TRYING OUT AN ALL CONV. NET
        fc3 = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)
        fc4 = tf.nn.relu(tf.matmul(fc3, W_fc4) + b_fc4)

        # reshaping into a 4-D matrix
        fc4_matrix = tf.reshape(fc4, [-1, 10, 10, 64])

        # deconv variables
        W_deconv1 = weight_variable([6, 6, 64, 64])
        b_deconv1 = bias_variable([64])

        W_deconv2 = weight_variable([6, 6, 64, 64])
        b_deconv2 = bias_variable([64])

        W_deconv3 = weight_variable([6, 6, 1, 64])
        b_deconv3 = bias_variable([1])

        # output - 1 x 84 84
        deconv1 = tf.nn.relu(deconv2d(fc4_matrix, W_deconv1, (BATCH, 20, 20, 64), 2) + b_deconv1)
        deconv2 = tf.nn.relu(deconv2d(deconv1, W_deconv2, (BATCH, 40, 40, 64), 2) + b_deconv2)
        deconv3 = deconv2d_nopad(deconv2, W_deconv3, (BATCH, 84, 84, 1), 2) + b_deconv3


        #encode = tf.reshape(tf.image.resize_images(deconv3, [84, 84]), [-1, 84, 84])
        encode = tf.reshape(deconv3, [-1, 84, 84])

        return encode

    def predict(self, sess, s, a):
        """
        Predicts the next state based on the current action.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a : Action input of shape [batch_size, ACTIONS]
        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.pred_frame, { self.state: state, self.action : action })

    def update(self, sess, s, a, y, p):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 84, 84, 4]
          a: Chosen actions of shape [batch_size, ACTIONS]
          y: Targets of shape [batch_size, 84, 84]
          p : Predicted next observation frame of shape [batch_size, 84, 84]
        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { y : target_batch, self.pred_frame : pred_batch,
                    self.state : state_batch, self.action : action_batch }
        summaries, _, loss = sess.run(
            [self.summaries, self.train_step, self.loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def rgb2gray(frame):

    r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def preprocess(frame):

    gray_image = rgb2gray(frame)
    reshaped_image = cv2.resize(gray_image.astype(np.float32), (84, 84))
    x = np.reshape(reshaped_image, [84,84,1])
    x *= 1 / 255.0

    return x

def rollout(sess, prediction_net):


    #tf.summary.scalar("Qval", encode)
    merged_summary_op = tf.summary.merge_all()

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print("Loaded model checkpoint {}...".format(checkpoint))

    #summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    D = deque()
    num_episodes = 0
    k = 0

    while num_episodes < MAX_EPISODES:
        ob = env.reset()

        obf = preprocess(ob)
        s_t = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))
        observations, actions = [], []

        i = 0

        for t in range(10000):
            env.render() #optional

            if i == 0:
                #action_id = env.action_space.sample()
                #action_id = random.randint(0,5)
                action_id = 0
                action_vector = np.zeros(ACTIONS)
                action_vector[action_id] = 1
                #actions.append(action_vector)

            ob, reward, done, info = env.step(action_id)

            obf = preprocess(ob)

            s_t1 = np.append(obf, s_t[:,:,0:3], axis = 2)

            # if training, collect data and apply learning updates
            if flags.train:
                # storing current state and the next frame
                D.append((s_t, action_vector, obf))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()


                if num_episodes > 32:
                    minibatch = random.sample(D, BATCH)
                    action_batch = [d[1] for d in minibatch]
                    state_batch = [d[0] for d in minibatch]
                    target_batch = [d[2] for d in minibatch]
                    target_batch = np.reshape(target_batch, (BATCH, 84, 84))

                    pred_batch = prediction_net.predict(np.reshape(state_batch, (BATCH, 84, 84, 4)), np.reshape(action_batch, (BATCH, 6)))

                    loss = prediction_net.predict(state_batch, action_batch, target_batch, pred_batch)

                    #summary_writer.add_summary(summary, num_episodes)

                    print("\riteration {} @ Episode {}/{}, loss {}".format(k, num_episodes, MAX_EPISODES, loss), end="")
                    sys.stdout.flush()

                    if k % 1000 == 0:
                        print("\nsaving model now")
                        saver.save(sess, save_path, global_step = t)

                    # display the first frame of the minibatch
                    cv2.imshow("prediction", pred_batch[0])
                    cv2.imshow("target", target_batch[0])
                    cv2.imshow("input", state_batch[0][:,:,0])
                    cv2.waitKey(5)

            else:
                #render video frames while testing
                prediction = prediction_net.predict(sess, np.reshape(s_t, (1, 84, 84, 4)), np.reshape(action_vector, (1, 6)))
                #print("prediction shape", prediction[0])
                cv2.imshow("prediction", prediction[0])
                cv2.waitKey(1)

            k += 1

            if i == 3: #maybe change to 4
                i = 0
            else:
                i +=1

            s_t = s_t1

            if done:
                num_episodes += 1
                break


sess = tf.InteractiveSession()

checkpoint_dir = '/home/manan/Downloads/models3/'
save_path = '/home/manan/Downloads/models/video_prediction.ckpt'
#save_path = '/home/manan/Downloads/models3/video_prediction.ckpt'
#load_path='/home/manan/Downloads/models3/video_prediction.ckpt-302'
logs_path = '/tmp/tensorboard_example'

prediction_net = autoencoder()
rollout(sess, prediction_net)
'''Pong : Actions 2,4 : up
                  3,5 : down
                  0,1 : no movement'''

# basic code for simulating random policy
'''for i_episode in range(2):
    observation = env.reset()
    ob = preprocess(observation)
    print(ob.shape)
    for t in range(10000)
        env.render()
        print(observation)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        #print(action)
        if done == True:
            print("Episode finished")
            break'''
