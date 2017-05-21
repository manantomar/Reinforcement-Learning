
''' We use downsampled gray scale images - 84 X 84,
    consider only every 4th frame as input, applying
    the same action for the intermediate frames.
    Minibatch size is taken to be 32. Each input
    consists of a fixed memory of T = 4 to unroll
    each trajectory and pass in as an input. K, which
    is the prediction step parameter, taken to be 1'''

import gym
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import random
from collections import deque

epsilon = 0.35
MAX_EPISODES = 5
BATCH = 32
max_iter = 5 #10000
ACTIONS = 6
FACTORS = 2048

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def deconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def autoencoder():

    # input - Batch X 84 X 84 X 4
    state = tf.placeholder("float", [None, 84, 84, 4])
    action = tf.placeholder("float", [None, ACTIONS])

    # 6 X 6 X 4 x 64 - stride 2
    W_conv1 = weight_variable([6, 6, 4, 64])
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

    conv1 = tf.nn.relu(conv2d(state, W_conv1, 2) + b_conv1)
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)
    conv3 = tf.nn.relu(conv2d(conv1, W_conv3, 2) + b_conv3)

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
    W_a = tf.matmul(action, tf.transpose(W_action))
    fc_interactions = tf.matmul(tf.multiply(W_henc, W_a), tf.transpose(W_dec)) + b_interactions

    # first fully connected layer after multiplicative interaction- 2048
    W_fc3 = weight_variable([2048, 1024])
    b_fc3 = bias_variable([1024])

    # second fully connected layer after multiplicative interaction- 1024 units
    W_fc4 = weight_variable([1024, 10*10*64])
    b_fc4 = bias_variable([10*10*64])

    fc3 = tf.nn.relu(tf.matmul(fc_interactions, W_fc3) + b_fc3)
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
    deconv1 = tf.nn.relu(deconv2d(fc4_matrix, W_deconv1, (-1, 20, 20, 64), 2) + b_deconv1)
    deconv2 = tf.nn.relu(deconv2d(deconv1, W_deconv2, (-1, 40, 40, 64), 2) + b_deconv2)
    deconv3 = deconv2d(deconv2, W_deconv3, (-1, 80, 80, 1), 2) + b_deconv3 #check back - 84 size


    encode = tf.reshape(tf.image.resize_images(deconv3, [84, 84]), [-1, 84, 84])

    return state, action, encode

def preprocess(frame):
    gray_image = frame.mean(2)
    reshaped_image = imresize(gray_image, (84,84))
    x = np.reshape(reshaped_image, [84,84,1])

    # divide by 255
    ''' clipping code here '''

    return x

def rollout(state, action, encode):

    # reshape the predicted frame
    '''reshape code here'''

    y = tf.placeholder("float", [None, 84, 84])
    pred_frame = encode
    cost = tf.square(tf.norm(y - pred_frame))
    train_step = tf.train.RMSPropOptimizer(1e-6).minimize(cost)

    print("working")
    sess.run(tf.initialize_all_variables())

    D = deque()
    num_episodes = 0

    while num_episodes < MAX_EPISODES:
        ob = env.reset()

        obf = preprocess(ob)
        s_t = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))
        observations, actions = [], []

        i = 1

        for t in range(10000):
            env.render() #optional

            if i == 1:
                action = env.action_space.sample()
                actions.append(action)

            ob, reward, done, info = env.step(action)
            #i += 1

            obf = preprocess(ob)
            s_t1 = np.append(obf, s_t[:,:,0:3], axis = 2)
            observations.append(s_t1)

            if i == 3: #maybe change to 4
                i = 1
            else:
                i +=1

            s_t = s_t1

            if done:
                num_episodes += 1
                break
        D.append((observations, actions))
        print("observations length", D[0][0][0].shape)

    #print("deque length", len(D[0][0]))
    while k < max_iter:
        minibatch = random.sample(D, BATCH)
        a = np.zeros((BATCH, ACTIONS))
        s = np.zeros((BATCH, 84, 84, 4))

        pred_batch = []
        target_batch = []

        for i, m in enumerate(minibatch):
            for t in range(3):
                pred = encode.eval(feed_dict = {action : m[1][t], state : m[0][t] })
                pred_batch.append(pred)
                target_batch.append(m[1][t])

                train_step.run(feed_dict = {
                        y : target_batch,
                        pred_frame : pred_batch})


env = gym.make('Pong-v0')
sess = tf.InteractiveSession()
state, action, encode = autoencoder()
rollout(state, action, encode)
'''Pong : Actions 2,4 : up
                  3,5 : down
                  0,1 : no movement'''

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
