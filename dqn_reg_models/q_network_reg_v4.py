"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule, lambda_reg,
                 batch_accumulator, pretrained_net, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.lambda_reg = lambda_reg

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_in, self.l_act_in, self.l_out, self.pred_z, self.true_z = \
                                        self.build_network(network_type, \
                                        input_width, input_height, num_actions,\
                                        num_frames, batch_size)

        if self.freeze_interval > 0:
            self.next_l_in, self.next_l_act_in, self.next_l_out, _d, _d = \
                                self.build_network(network_type, input_width, \
                                input_height, num_actions, num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.imatrix('actions')
        terminals = T.icol('terminals')

        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) images, along with the chosen action and resulting
        # reward and terminal status.
        self.imgs_shared = theano.shared(
            np.zeros((batch_size, num_frames*2+1, input_height, input_width),
                     dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, num_frames), dtype='int32')
            )
        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # Shared variable for a single state, to calculate q_vals.
        self.state_shared = theano.shared(
            np.zeros((num_frames*2, input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals, z_pred, z_true = lasagne.layers.get_output(
                                    [self.l_out, self.pred_z, self.true_z],
                                    inputs = {self.l_in: states / input_scale,
                                        self.l_act_in: actions}
                                )
        
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(
                                    self.next_l_out,
                                    {self.next_l_in: next_states / input_scale, 
                                     self.next_l_act_in: actions}
                                    )
        else:
            next_q_vals = lasagne.layers.get_output(
                                    self.l_out,
                                    {self.l_in: next_states / input_scale, 
                                     self.l_act_in: actions}
                                    )
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        terminalsX = terminals.astype(theano.config.floatX)
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                actions[:, 0].reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards +
                  (T.ones_like(terminalsX) - terminalsX) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output
        diff_reg = z_true - z_pred

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = loss + 0.5 * self.lambda_reg * (diff_reg ** 2).sum(axis=1)

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params([self.l_out, self.pred_z, self.true_z])  
        train_givens = {
            states: self.imgs_shared[:, :-1],
            next_states: self.imgs_shared[:, 1:],
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }

        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)
        q_givens = {
            states: self.state_shared.reshape((1,
                                               self.num_frames*2,
                                               self.input_height,
                                               self.input_width))
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)

    def build_network(self, network_type, input_width, input_height,
                      output_dim, num_frames, batch_size):
        if network_type == "latent_dnn_v4":
            return self.build_latent_network_dnn_v4(input_width, input_height,
                                               output_dim, num_frames,
                                               batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (2f) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 4 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        imgs_padded = np.zeros((imgs.shape[0], imgs.shape[1]+1,
                                imgs.shape[2], imgs.shape[3]), dtype=np.float32)
        imgs_padded[:,:-1] = imgs 

        self.imgs_shared.set_value(imgs_padded)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_latent_network_dnn_v4(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import dnn
        
        """
        States input
        """
        l_in = lasagne.layers.InputLayer(
            shape=(None, num_frames*2, input_width, input_height)
        )
        
        """
        Integer encoding input for actions
        """
        l_act_in = lasagne.layers.InputLayer(
            shape=(None, 4)
        )

        l_act_in_reshaped = lasagne.layers.ReshapeLayer(
            l_act_in,
            shape=(-1, )
        )
        
        """
        Action embedding
        """
        l_act_embed = lasagne.layers.EmbeddingLayer(
            l_act_in_reshaped,
            input_size=output_dim,
            output_size=256,
            W=lasagne.init.HeUniform()
        )
        
        l_act_embed_reshaped = lasagne.layers.ReshapeLayer(
            l_act_embed,
            shape=(-1, num_frames*256)
        )
        
        """
        State embedding
        """
        l_reshaped_in = lasagne.layers.ReshapeLayer(
            l_in, 
            shape=(-1, num_frames, input_width, input_height)
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            l_reshaped_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden_reshaped = lasagne.layers.ReshapeLayer(
            l_hidden1,
            shape=(-1, 512*2)
        )

        """ 
        "True" latent embeddings for current state and future state
        """
        l_latent_1 = lasagne.layers.SliceLayer(
            l_hidden_reshaped,
            indices=slice(0, 512),
            axis=1
        )

        l_out_3 = lasagne.layers.SliceLayer(
            l_hidden_reshaped,
            indices=slice(512, 1024),
            axis=1
        )

        """
        Future state latent embedding prediction using current
        state and future action embeddings
        """
        l_act_project = lasagne.layers.DenseLayer(
            l_act_embed_reshaped,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_state_project = lasagne.layers.DenseLayer(
            l_latent_1,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_project_concat = lasagne.layers.ConcatLayer(
            [l_act_project, l_state_project],
            axis=1
        )

        l_out_2 = lasagne.layers.DenseLayer(
            l_project_concat,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        """
        Action prediction based on current state
        """
        l_out_1 = lasagne.layers.DenseLayer(
            l_latent_1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_in, l_act_in, l_out_1, l_out_2, l_out_3

def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')


if __name__ == '__main__':
    main()
