from __future__ import print_function
import gym
from gym import wrappers
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imresize
from collections import deque
import sys
import os
import random
import cv2

flags = tf.app.flags
flags.DEFINE_boolean('train', True, 'Whether to do training or testing')
flags.DEFINE_string('env_name', 'Pong', 'The name of gym environment to use')

env = gym.make(flags.FLAGS.env_name + 'NoFrameskip-v0')

ACTIONS = env.action_space.n
INITIAL_EPSILON = 1.
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 1000000
max_episodes = 100000
BATCH = 32
GAMMA = 0.99
max_iter = 5000

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

def weight_variable(name, shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name = name, shape = shape, initializer = initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class dqn():

    def __init__(self, clip_delta, scope, discount):

        self.clip_delta = clip_delta
        self.scope = scope
        self.discount = discount

        with tf.variable_scope(self.scope):

            self.net = self.build_net()

            self.y = tf.placeholder("float", [None])
            self.diff = self.y - tf.reduce_max(self.net, axis = 1)

            if self.clip_delta > 0:
                quadratic_part = tf.minimum(abs(self.diff), self.clip_delta)
                linear_part = abs(self.diff) - quadratic_part
                self.loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
            else:
                self.loss = 0.5 * self.diff ** 2

            self.loss = tf.reduce_mean(self.loss)
            self.train_step = tf.train.AdamOptimizer(0.00025).minimize(self.loss)
            self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            self.summaries = tf.summary.merge_all(tf.summary.scalar("loss", self.loss))

    def build_net(self, ):

        # input - Batch X 84 X 84 X 4
        self.s = tf.placeholder("float", [None, 84, 84, 4])

        # 8 X 8 X 4 x 32 - stride 4
        W_conv1 = weight_variable("w1", [8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        # 4 X 4 X 32 x 64 - stride 2
        W_conv2 = weight_variable("w2", [4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        # 3 X 3 X 64 x 64 - stride 1
        W_conv3 = weight_variable("w3", [3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        # 3*3*64 ie. flattened output from conv3
        W_fc1 = weight_variable("w4",[3136, 512])
        b_fc1 = bias_variable([512])

        W_fc2 = weight_variable("w5",[512, ACTIONS])
        b_fc2 = bias_variable([ACTIONS])

        conv1 = tf.nn.relu(conv2d(self.s, W_conv1, 4) + b_conv1)
        conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)
        conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 1) + b_conv3)

        # flatten the output from conv3 layer
        conv3_flat = tf.reshape(conv3, [-1, 3136])

        # add two fully connected layers
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)
        out_fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)

        return out_fc2

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

def rollout(sess, q_network, target_network):

    merged_summary_op = tf.summary.merge_all()
    q_summary = tf.Summary()

    num_episodes = 0
    epsilon = INITIAL_EPSILON

    replay_memory = deque()

    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables())

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if checkpoint:
        saver.restore(sess, checkpoint)
        print("Loaded model checkpoint {}...".format(checkpoint))

    print("collecting initial rollouts...")
    i = 0
    global_step = 0

    while num_episodes < max_episodes:

        ob = env.reset()
        steps_per_episode = 0
        reward_per_episode = 0

        ob_flkr = preprocess(ob)
        obf_flkr = np.reshape(np.stack((ob_flkr, ob_flkr), axis=2), (84, 84, 2))
        obf = np.amax((obf_flkr[:,:,0], obf_flkr[:,:,1]), (0))
        state = np.reshape(np.stack((obf, obf, obf, obf), axis=2), (84, 84, 4))

        action_index = np.argmax(sess.run(q_network.net, feed_dict = {q_network.s : state.reshape((1, 84, 84, 4)) }))
        loss_per_episode = 0
        reward_per_episode = 0

        for t in range(10000):

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 100000

            ob, reward, done, info = env.step(action_index)

            REWARD = reward
            if reward > 1:
                REWARD = 1
            if reward < -1:
                REWARD = -1

            reward_per_episode += reward
            ob_flkr = preprocess(ob)
            obf_flkr = np.append(ob_flkr, obf_flkr[:,:,0:1], axis = 2)
            obf = np.amax((obf_flkr[:,:,0], obf_flkr[:,:,1]), (0)).reshape((84,84,1))


            if i == 3:
                action_index = 0
                action = np.zeros(ACTIONS, np.int32)

                if random.random() <= epsilon:
                    action_index = random.randrange(ACTIONS)
                    action[action_index] = 1
                else:
                    action_index = np.argmax(sess.run(q_network.net, feed_dict = {q_network.s : state.reshape((1, 84, 84, 4)) }))
                    action[action_index] = 1

                next_state = np.append(obf, state[:,:,0:3], axis = 2)

                replay_memory.append((state, action, REWARD, next_state, done))

                if len(replay_memory) > REPLAY_MEMORY:
                    replay_memory.popleft()

                if global_step > 50:

                    #training starts
                    minibatch = random.sample(replay_memory, BATCH)

                    # get the batch variables
                    state_batch = [d[0] for d in minibatch]
                    action_batch = [d[1] for d in minibatch]
                    reward_batch = [d[2] for d in minibatch]
                    next_state_batch = [d[3] for d in minibatch]
                    done_batch = [d[4] for d in minibatch]

                    next_q_value = sess.run(target_network.net, feed_dict = {target_network.s : next_state_batch})
                    q_value = sess.run(q_network.net, feed_dict = {q_network.s : state_batch})

                    target_batch = np.asarray(reward_batch) + q_network.discount * \
                    (np.ones_like(done_batch) - done_batch) * \
                    np.max(next_q_value, axis=1)

                    _, loss = sess.run([q_network.train_step, q_network.loss], feed_dict = { \
                        q_network.s : state_batch, \
                        q_network.y : target_batch})

                    print("\riteration {} @ episode {}/{}".format(global_step, num_episodes, max_episodes), end="")
                    sys.stdout.flush()

                    if global_step % 10000 == 0:
                        print("\nsaving model now")
                        saver.save(sess, checkpoint_path)
                        print("\nupdating target network...")
                        copy_model_parameters(sess, q_network, target_network)

                    steps_per_episode += 1
                    loss_per_episode += loss

                global_step += 1
                state = next_state

            i += 1

            if i == 4:
                i = 0

            if done or steps_per_episode >= max_iter:
                num_episodes += 1
                if global_step > 50:
                    print("\nloss per episode {}".format(loss_per_episode / steps_per_episode))
                    print("\nreward per episode {}".format(reward_per_episode))
                    Q = np.amax(sess.run(q_network.net, feed_dict = {q_network.s : state.reshape((1, 84, 84, 4)) }))

                    q_summary.value.add(simple_value=steps_per_episode, node_name="episode_lengths", tag="episode_lengths")
                    q_summary.value.add(simple_value=Q, node_name="q_value", tag="q_value")
                    q_summary.value.add(simple_value=reward_per_episode, node_name="episode_reward", tag="episode_reward")
                    q_network.summary_writer.add_summary(q_summary, global_step)
                    q_network.summary_writer.flush()

                ob = env.reset()
                break

monitor_dir = os.path.abspath("./{}-experiment/".format(flags.FLAGS.env_name))
checkpoint_dir = os.path.abspath("./dqn/")
logs_path = os.path.abspath("./tensorboard_example/")

env = wrappers.Monitor(env, monitor_dir, force=True)

#load_path='/home/manan/Downloads/models/pong.ckpt-2920000-2940000'
#save_path = '/home/manan/Downloads/models2/pong.ckpt'

sess = tf.InteractiveSession()

q_network = dqn(1.0, scope="q_net", discount=0.99)
target_network = dqn(1.0, scope="target_network", discount=0.99)

rollout(sess, q_network, target_network)
