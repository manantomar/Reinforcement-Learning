import gym
from gym import wrappers
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imresize
from collections import deque
import sys
import random
import cv2

INITIAL_EPSILON = 1.
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 1000000
max_episodes = 100000
BATCH = 32
GAMMA = 0.99


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def dqn():

    # input - Batch X 84 X 84 X 4
    #s = tf.placeholder("float", [None, 84, 84, 4])
    s = tf.placeholder("float", [None, 42, 42, 4])
    #print(x.shape)

    # 5 X 5 X 4 x 32 - stride 4
    W_conv1 = weight_variable([5, 5, 4, 32])
    b_conv1 = bias_variable([32])


    # 4 X 4 X 32 x 16 - stride 2
    W_conv2 = weight_variable([4, 4, 32, 16])
    b_conv2 = bias_variable([16])

    # _*16 ie. flattened output from conv2
    W_fc1 = weight_variable([11*11*16, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)
    #h_pool1 = max_pool_2x2(h_conv1)

    #conv1 = tf.layers.conv2d(s, 32, [5, 5], padding="same", activation=tf.nn.relu)
    #print(h_conv1.shape)
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #conv2 = tf.layers.conv2d(pool1, 64, [5, 5], padding="same", activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #print("passes pool2", pool2.shape)
    conv2_flat = tf.reshape(conv2, [-1, 11*11*16])
    fc1 = tf.nn.relu(tf.matmul(conv2_flat, W_fc1) + b_fc1)

    # readout layer
    action_logits = tf.matmul(fc1, W_fc2) + b_fc2

    #dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    #print("passes dense", dense.shape)
    #action_logits = tf.layers.dense(inputs=dense, units=ACTIONS)
    #print("passes logits", action_logits.shape)
    #conv1 = slim.conv2d(x, 10, [5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(conv1, 10, [5,5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
    #net = slim.flatten(conv2)
    #action_logits = slim.fully_connected(net, ACTIONS, activation_fn=None)

    return s, action_logits

def preprocess(frame):
    gray_image = frame.mean(2)
    #reshaped_image = imresize(gray_image, (84,84))
    #import cv2
    #cv2.resize(image, (shape1, shape2))
    reshaped_image = imresize(gray_image, (80,80))
    reshaped_image = imresize(reshaped_image, (42,42))
    # divide by 255 - optional
    ''' clipping code here '''
    x = np.reshape(reshaped_image, [42,42,1])
    return x

def rollout(s, action_logits, sess, max_iter=5000):

    observations, actions, rewards = [], [], []
    ob = env.reset()
    ep_steps = 0
    num_episodes = 0
    epsilon = INITIAL_EPSILON

    obf = preprocess(ob)
    #s_t = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))
    s_t = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (42, 42, 4))

    D = deque()
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    readout_action = tf.reduce_sum(tf.multiply(action_logits, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, load_path)
    print("variables restored and loaded...")

    t = 0
    Q_update_index = 0
    while True and num_episodes < max_episodes:

        #ob_before = ob_now if ob_prev is None else ob_prev
        #obf_prev = np.concatenate((ob_before, ob_now), 2)
        #ob_prev = ob_now

        #readout_t = action_logits.eval(feed_dict = {s : np.reshape(s_t, (1, 84, 84, 4))})
        readout_t = action_logits.eval(feed_dict = {s : np.reshape(s_t, (1, 42, 42, 4))})

        action_index = 0
        action = np.zeros(ACTIONS, np.int32)

        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
            action[action_index] = 1
            #print("random action ", action)
        else:
            action_index = np.argmax(readout_t)
            action[action_index] = 1
            #print("greedy action", action)
        #print("action is ", action)

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 1000

        ob, reward, done, info = env.step(np.argmax(action))
        ep_steps += 1

        obf = preprocess(ob)
        s_t1 = np.append(obf, s_t[:,:,0:3], axis = 2)

        #ob_before = ob_now if ob_prev is None else ob_prev
        #obf_now = np.concatenate((ob_before, ob_now), 2)

        #observations.append(ob)
        #actions.append(action)
        #rewards.append(reward)

        D.append((s_t, action, reward, s_t1, done))
        if len(D) > REPLAY_MEMORY:
                D.popleft()

        if t > 50:
            #training starts
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_t_batch = [d[0] for d in minibatch]
            action_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            s_t1_batch = [d[3] for d in minibatch]

            if Q_update_index % 10000 == 0:
                print("updating target network")
                target_batch = []
                readout_j1_batch = action_logits.eval(feed_dict = {s : s_t1_batch})
                #obf_batch = np.concatenate(ob_before, ob_now), 2)
                for i in range(0, len(minibatch)):

                    if minibatch[i][4]:
                        target_batch.append(reward_batch[i])

                    else:
                        #print("obf_prev_batch shape ", len(obf_prev_batch))
                        target_batch.append(reward_batch[i] + GAMMA*np.max(readout_j1_batch[i]))

            Q_update_index += 1
            #print("obf_prev_batch", obf_prev_batch)
            train_step.run(feed_dict = {
                y : target_batch,
                a : action_batch,
                s : s_t_batch})

        s_t = s_t1
        t += 1

        if t % 10000 == 0:
            print("saving model now")
            saver.save(sess, save_path, global_step = t)

        if done or ep_steps >= max_iter:
            num_episodes += 1
            print("number of episodes", num_episodes)
            ep_steps = 0
            #ob_prev = None
            ob = env.reset()

env = gym.make('Pong-v0')
env = wrappers.Monitor(env, '/tmp/Pong-experiment-2', force=True)
load_path='/home/manan/Downloads/models/pong.ckpt-2920000-2940000'
save_path = '/home/manan/Downloads/models2/pong.ckpt'

#x = tf.placeholder(tf.float32, name )
#sampled_actions = tf.placeholder(tf.int32)
#discounted_reward = tf.placeholder(tf.float32)

#action_logits = dqn(x)
ACTIONS = env.action_space.n
sess = tf.InteractiveSession()
s, action_logits = dqn()
rollout(s, action_logits, sess)
#print(env.action_space.n)
#for i_episode in range(1):
#    observation = env.reset()
#    ob = preprocess(observation)
#    print(ob.shape)
#    for t in range(10000):
#        env.render()
#        #print(observation)
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        #print(action)
#        if done == True:
#            print("Episode finished")
#            break
