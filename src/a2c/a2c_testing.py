from abc import abstractclassmethod
from trfl import sequence_advantage_actor_critic_loss
import keras.backend as K
from functools import reduce

if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import gym
import environment.qmaze

import gym
import time
from common.vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np

import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util


from baselines.a2c.utils import Scheduler, find_trainable_variables

from tensorflow import losses
from keras.layers import Input

from baselines.common.models import get_network_builder
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.mpi_running_mean_std import RunningMeanStd

from keras.layers import Dense, TimeDistributed, Input
from keras.models import Sequential
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    return Sequential(layers = [
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
    ])
    
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn

from baselines.a2c.utils import fc
from keras.initializers import Orthogonal
from keras.models import Model as MD

def get_model(n_envs, observation_space, action_space):
    input_placeholder = Input(batch_shape=(n_envs, None) + observation_space.shape)
    policy_latent = TimeDistributed(mlp())(input_placeholder)
    value_latent =TimeDistributed(mlp())(input_placeholder)
    
    policy_probs = TimeDistributed(Dense(action_space.n, bias_initializer = 'zeros', activation='softmax', kernel_initializer = Orthogonal(gain=0.01)))(policy_latent)
    value = TimeDistributed(Dense(1, bias_initializer = 'zeros', kernel_initializer = Orthogonal(gain = 1.0)))(value_latent)
    return MD(inputs = [input_placeholder], outputs = [policy_probs, value])

class ActorCriticModelWrapper:
    def __init__(self, sess, n_envs, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.alpha = 0.99
        self.epsilon = 1e-5

        self._build_graph(sess)
        self.initial_state = None

        self.train_model =  self

    def create_model(self):
        with tf.variable_scope('a2c_model'):
            model = get_model(self.n_envs, self.observation_space, self.action_space)
        
        return model

    @property
    def learning_rate(self):
        return 7e-4

    def _build_graph(self, sess):
        # Keras will use this session
        K.set_session(sess)

        # Create model and outputs
        self.model = self.create_model()
        policy, values = self.model.outputs
        policy_distribution = tf.distributions.Categorical(probs = policy)

        # Create loss placeholders
        actions = tf.placeholder(tf.int32, [self.n_envs, None], name = 'actions')
        adventages = tf.placeholder(tf.float32, [self.n_envs, None], name = 'adventages')
        returns = tf.placeholder(tf.float32, [self.n_envs, None], name = 'returns')
        learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

        # Policy gradient loss
        selected_negative_log_prob = -policy_distribution.log_prob(actions)
        policy_gradient_loss = tf.reduce_mean(adventages * selected_negative_log_prob)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy_distribution.entropy())

        # Value loss
        value_loss = losses.mean_squared_error(tf.squeeze(values, axis = 2), returns)

        # Total loss
        loss = policy_gradient_loss \
            + value_loss * self.value_coefficient \
            - entropy * self.entropy_coefficient

        # Optimize step
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=self.alpha, epsilon=self.epsilon)
        params = self.model.trainable_weights
        grads = tf.gradients(loss, params)
        if self.max_gradient_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_gradient_norm)
        grads = list(zip(grads, params))
        optimize_op = optimizer.apply_gradients(grads)

        # Initialize variables
        tf.global_variables_initializer().run(session = sess)

        # Create train fn
        def train(b_obs, b_states, b_rewards, b_masks, b_actions, b_values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            b_obs = b_obs.reshape([self.n_envs, -1] + list(b_obs.shape[1:]))
            #actions = actions.reshape([nenvs, -1] + list(actions.shape[1:]))
            b_returns = b_rewards.reshape([self.n_envs, -1] + list(b_rewards.shape[1:]))
            b_masks = b_masks.reshape([self.n_envs, -1] + list(b_masks.shape[1:]))
            b_values = b_values.reshape([self.n_envs, -1] + list(b_values.shape[1:]))

            advs = b_returns - b_values

            td_map = {self.model.inputs[0]:b_obs, actions:b_actions, adventages:advs, returns:b_returns, learning_rate:self.learning_rate}
            policy_loss_v, value_loss_v, policy_entropy_v, _ = sess.run(
                [policy_gradient_loss, value_loss, entropy, optimize_op],
                td_map
            )
            return policy_loss_v, value_loss_v, policy_entropy_v


        action = policy_distribution.sample()
        # step_fn = K.function(self.model.inputs, [action])

        self.action = action #TODO:remove

        def step(observation, S = None, M = None):
            observation = observation.reshape([self.n_envs, -1] + list(observation.shape[1:]))
            return sess.run([action, values], feed_dict={
                self.model.inputs[0]: observation
            }) + [None, None]

        def value(observation, S = None, M = None):
            observation = observation.reshape([self.n_envs, -1] + list(observation.shape[1:]))
            return sess.run(values, feed_dict={
                self.model.inputs[0]: observation
            }).reshape([-1])

        self.step = step
        self.train = train
        self.value = value


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

class Runner:
    def __init__(self, env, model, nsteps = 5, gamma = .99):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]

    def run(self):
        batch = []

        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            

            # Take actions in env and look the results
            obs, rewards, dones, _ = self.env.step(actions)

            batch.append((np.copy(self.obs), actions, values, rewards, dones))

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)

        last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()

        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]


        last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()


        # use custom batch code
        terminals = self.dones
        # compare
        
        observations, actions, values, rewards, terminals = list(map(lambda *x: np.stack(x, axis = 1), *batch))
        last_returns = (1.0 - terminals[:, -1]) * last_values
        returns = np.concatenate([np.zeros_like(rewards), np.expand_dims(last_returns, 1)], axis = 1)
        for n in reversed(range(len(batch))):
            returns[:, n] = rewards[:, n] + \
                self.gamma * (1.0 - terminals[:, n]) * returns[:, n + 1]

        if not np.allclose(returns[:, :-1].flatten(), mb_rewards):
            print(returns[:, :-1])
            print(mb_rewards.reshape([16, -1]))

            return 'error'
        else:
            print('ok')

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, model, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment
        observations    tensorflow placeholder in which the observations will be fed
        latent          latent state from which policy distribution parameters should be inferred
        vf_latent       latent state from which value function should be inferred (if None, then latent is used)
        sess            tensorflow session to run calculations in (if None, default session is used)
        **tensors       tensorflow tensors for additional attributes such as state or mask
        """

        self.X = model.inputs[0]
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        #vf_latent = vf_latent if vf_latent is not None else latent

        #vf_latent = tf.layers.flatten(vf_latent)
        #latent = tf.layers.flatten(latent)
        #print(latent)
        #print(latent.shape)

        #vf_latent = tf.reshape(vf_latent, [-1] + list(vf_latent.shape[2:]))
        #latent = tf.reshape(latent, [-1] + list(latent.shape[2:]))

        # Based on the action space, will select what probability distribution type
        #policy_pi = fc(latent, 'pi', env.action_space.n, init_scale=0.01, init_bias=0.0)

        policy_probs, value = model.outputs
        self.pd = tf.distributions.Categorical(probs = policy_probs)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = -self.pd.log_prob(self.action)
        self.sess = sess or tf.get_default_session()

        self.vf = value

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)
        Parameters:
        ----------
        observation     observation data (either single or a batch)
        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)
        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None

        v = v.reshape([-1])
        a = a.reshape([-1])
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        v = self._evaluate(self.vf, ob, *args, **kwargs)
        v = v.reshape([-1])
        return v

    #def save(self, save_path):
    #    tf_util.save_state(save_path, sess=self.sess)

    #def load(self, load_path):
    #    tf_util.load_state(load_path, sess=self.sess)

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    network_type = 'mlp'
    policy_network = mlp(**policy_kwargs)

    def policy_fn(nenvs=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = Input(batch_shape=(nenvs, None) + ob_space.shape)

        extra_tensors = {}

        policy_latent = TimeDistributed(mlp())(X)
        vf_latent =TimeDistributed(mlp())(X)

        from keras.initializers import Orthogonal
        from keras.models import Model as MD
        policy_probs = TimeDistributed(Dense(env.action_space.n, bias_initializer = 'zeros', activation='softmax', kernel_initializer = Orthogonal(gain=0.01)))(policy_latent)
        value = TimeDistributed(Dense(1, bias_initializer = 'zeros', kernel_initializer = Orthogonal(gain = 1.0)))(vf_latent)

        #with tf.variable_scope('pi', reuse=False):
        #    policy_latent = policy_network(tf.reshape(encoded_x, [-1] + list(encoded_x.shape[2:])))


        #with tf.variable_scope('vf', reuse=False):
        #    vf_latent = policy_network(tf.reshape(encoded_x, [-1] + list(encoded_x.shape[2:])))

        policy = PolicyWithValue(
            env=env,
            observations=X,
            model = MD(inputs = [X], outputs = [policy_probs, value]),
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

def create_model(inputs, sess, nenvs):
    from keras.models import Model
    from math import sqrt


    transformed_inputs = Input(batch_shape=(nenvs, None, 4,))
    layer_initializer = initializers.Orthogonal(gain=sqrt(2))
    actor = TimeDistributed(Dense(
        units = 64, 
        activation = 'tanh',
        bias_initializer = 'zeros',
        kernel_initializer = layer_initializer))(transformed_inputs)

    actor = TimeDistributed(Dense(
        units = 64, 
        activation = 'tanh',
        bias_initializer = 'zeros',
        kernel_initializer = layer_initializer))(actor)
    actor = TimeDistributed(Dense(
        units = 2, 
        activation= 'softmax',
        bias_initializer='zeros',
        kernel_initializer = initializers.Orthogonal(gain=0.01)))(actor)


    critic = TimeDistributed(Dense(
        units = 64, 
        activation = 'tanh',
        bias_initializer = 'zeros',
        kernel_initializer = layer_initializer))(transformed_inputs)

    critic = TimeDistributed(Dense(
        units = 64, 
        activation = 'tanh',
        bias_initializer = 'zeros',
        kernel_initializer = layer_initializer))(critic)

    critic = TimeDistributed(Dense(
        units = 1,
        activation = None,
        bias_initializer='zeros',
        kernel_initializer=layer_initializer
    ))(critic)

    model = Model(inputs = [transformed_inputs], outputs = [actor, critic])

    outputs = model(inputs)

    model.pd = tf.distributions.Categorical(probs = outputs[0])
    model.vf = tf.reshape(outputs[1], [-1])
    model.action = tf.reshape(model.pd.sample(), [-1])
    model.neglogp = -model.pd.log_prob(model.action)

    def _evaluate(variables, observation, **extra_feed):
        feed_dict = {inputs: observation}
        return sess.run(variables, feed_dict)

    def step(observation, **extra_feed):
        observation  = observation.reshape([nenvs, -1] + list(observation.shape[1:]))
        a, v, neglogp = _evaluate([model.action, model.vf, model.neglogp], observation, **extra_feed)
        #a = a.reshape([-1])
        #v = v.reshape([-1])
        return (a, v, None, neglogp)
    
    def value(ob, *args, **kwargs):
        return _evaluate(model.vf, ob, *args, **kwargs)

    model.value = value
    model.step = step
    return model

class RolloutStorage(object):
    def __init__(self):
        self._terminates = None
        self._states = None
        self.batch = []

    @property
    def terminates(self):
        return self._terminates

    @property
    def states(self):
        return self._states

    def insert(self, obs, rewards, actions, values, terminates, states = None):
        self.batch.append([obs, rewards, actions, values, terminates])
        self._terminates = terminates
        self._states = states

    def clear(self):
        self.batch.clear()

    def sample(self, gamma, last_values):
        obs, rewards, actions, values, terminates = tuple(map(lambda *x: np.stack(x, axis = 1), *self.batch))

        batch_size = obs.shape[:2]
        returns = np.zeros(shape = (batch_size[0], batch_size[1] + 1), dtype = np.float32)
        returns[:,-1] = last_values
        for step in reversed(range(rewards.shape[1])):
            returns[:, step] = returns[:, step + 1] * \
                gamma * (1.0 - terminates)[:, step] + rewards[:, step]

        return obs, actions, returns[:,:-1], rewards
        

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps, sess = None,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        K.set_session(sess)
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=False):
            I = tf.placeholder(tf.float32, (None, 4), name= 'inputs')
            # model = create_model(I, sess, nenvs)
            
            
            #step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nenvs, nsteps, sess)

        A = tf.placeholder(tf.int32, [nenvs, None])
        ADV = tf.placeholder(tf.float32, [nenvs, None])
        R = tf.placeholder(tf.float32, [nenvs, None])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        #neglogpac = tf.reshape(-model.pd.log_prob(tf.reshape(A, [nenvs, -1])), [-1])
        neglogpac = -train_model.pd.log_prob(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vfval = tf.reshape(train_model.vf, [nenvs, -1])
        vf_loss = losses.mean_squared_error(tf.squeeze(vfval), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        #params = model.trainable_weights
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            obs = obs.reshape([nenvs, -1] + list(obs.shape[1:]))
            #actions = actions.reshape([nenvs, -1] + list(actions.shape[1:]))
            rewards = rewards.reshape([nenvs, -1] + list(rewards.shape[1:]))
            masks = masks.reshape([nenvs, -1] + list(masks.shape[1:]))
            values = values.reshape([nenvs, -1] + list(values.shape[1:]))

            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = train_model
        self.step = train_model.step
        self.value = train_model.value
        self.initial_state = None
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''



    set_global_seeds(seed)
    sess = tf_util.get_session()

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    #model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
    #    max_grad_norm=max_grad_norm, sess = sess, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    model = ActorCriticModelWrapper(sess, nenvs, env.observation_space, env.action_space)
    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("rewards", sum(rewards))
            logger.dump_tabular()
    return model



from common.train import SingleTrainer
class Trainer(SingleTrainer):
    def __init__(self, name, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self.name = name
        self.n_steps = 5
        self.n_env = 16
        self.total_timesteps = 1000000
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.learning_rate = 7e-4
        self.max_grad_norm = 0.5

        self.rollouts = RolloutStorage()

    @abstractclassmethod
    def create_model(self, inputs):
        pass

    @abstractclassmethod
    def create_inputs(self, name = 'main'):
        pass

    def wrap_env(self, env):
        return DummyVecEnv([lambda: gym.make(**self._env_kwargs) for _ in range(self.n_env)])

    def _initialize(self, **model_kwargs):
        model = self._build_graph()

        #self.obs = np.zeros((self.n_env,) + self.env.observation_space.shape, dtype=self.env.observation_space.dtype.name)
        #self.obs[:] = self.env.reset()
        self.obs = None

        self.batch_ob_shape = (self.n_env*self.n_steps,) + self.env.observation_space.shape
        self._experience = []
        self._reports = [(0, 0.0) for _ in range(self.n_env)]
        self._cum_reports = [(0, [], []) for _ in range(self.n_env)]

        self._global_t = 0
        self._lastlog = 0
        self._tstart = time.time()
        self._n_updates = 0
        return model

    def _update_report(self, rewards, terminals):
        self._reports = [(x + (1 - a), y + b) for (x, y), a, b in zip(self._reports, terminals, rewards)]
        for i, terminal in enumerate(terminals):
            if terminal:
                (ep, leng, rew) = self._cum_reports[i]
                leng.append(self._reports[i][0])
                rew.append(self._reports[i][1])
                self._cum_reports[i] = (ep + 1, leng, rew)
                self._reports[i] = (0, 0.0)

    def _collect_report(self):
        output = list(map(lambda *x: reduce(lambda a,b:a+b, x), *self._cum_reports))
        self._cum_reports = [(0, [], []) for _ in range(self.n_env)]
        return output

    def _build_loss(self, policy_probs, values, returns, actions, action_space_size):
        policy_logits = tf.log(tf.clip_by_value(policy_probs, K.epsilon(), 1.0))
        entropy = -tf.reduce_mean(tf.reduce_sum(policy_probs * policy_logits, reduction_indices = -1))
    
        selected_logits = tf.reduce_sum(tf.one_hot(actions, action_space_size) * policy_logits, axis = -1)

        adventages = returns - values
        value_loss = 0.5 * tf.reduce_mean(tf.square(adventages))
        policy_loss = -tf.reduce_mean(tf.stop_gradient(adventages) * selected_logits)

        loss = self.entropy_coef * entropy + value_loss + policy_loss
        return loss, policy_loss, value_loss, entropy 

    def _build_graph(self):
        action_space_size = self.env.action_space.n
        sess = K.get_session()

        inputs = self.create_inputs('main')
        model = self.create_model(inputs)
        policy_probs, baseline_values = model.outputs

        with tf.name_scope('training'):
            returns = tf.placeholder(dtype = tf.float32, shape = (self.n_env, None), name = 'returns')
            actions = tf.placeholder(dtype = tf.int32, shape = (self.n_env, None), name = 'actions')
            learning_rate = tf.placeholder(dtype = tf.float32, name = 'learning_rate')

            baseline_values_shaped = tf.reshape(baseline_values, [self.n_env, -1])
            loss, policy_loss, value_loss, entropy = self._build_loss(policy_probs, baseline_values_shaped, returns, actions, action_space_size)

            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, decay = 0.99, epsilon = 1e-5)

            params = model.trainable_weights
            grads = tf.gradients(loss, params)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads = list(zip(grads, params))

            # Create optimize op
            optimize_op = optimizer.apply_gradients(grads)


        sess.run(tf.global_variables_initializer())

        def train(b_inputs, b_returns, b_actions):
            return sess.run([loss, entropy, policy_loss, value_loss, optimize_op], feed_dict = {
                learning_rate: self.learning_rate,
                inputs: b_inputs,
                returns: b_returns,
                actions: b_actions
            })[:-1]

        def predict_single(s_inputs):
            res = sess.run([policy_probs, baseline_values], feed_dict = {
                inputs: np.expand_dims(s_inputs, 1)
            })

            return (res[0][:, 0], res[1][:, 0, 0])
        
        self._train = train
        self._predict_single = predict_single
        return model

    def act(self, state):
        pvalues, baseline_values = self._predict_single(state)

        s = pvalues.cumsum(axis=1)
        r = np.random.rand(pvalues.shape[0], 1)
        actions = (s < r).sum(axis=1)
        return actions, baseline_values

    def _sample_experience_batch(self):
        if self.obs is None:
            self.obs = self.env.reset()

        for n in range(self.n_steps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values = self.act(self.obs)
            obs, rewards, dones, _ = self.env.step(actions)

            self.rollouts.insert(obs, rewards, actions, values, dones, None)
            self._update_report(rewards, dones)
            self.obs = obs
        
        _, bootstrap_values = self.act(self.obs)
        bootstrap_values = (1.0 - dones) * bootstrap_values # Remove value of finished episode

        # Convert data to acceptable format
        batched = self.rollouts.sample(self.gamma, bootstrap_values)
        self.rollouts.clear()
        return batched, self._collect_report()

    def _get_end_stats(self):
        return None

    def process(self):
        (obs, actions, returns, rewards), ep_stats = self._sample_experience_batch()
        time_moved = obs.shape[1]

        loss, policy_entropy, policy_loss, value_loss = self._train(obs, returns, actions)
        #policy_loss, value_loss, policy_entropy = self.model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-self._tstart
        self._global_t += time_moved * self.n_env
        self._n_updates += 1

        # Calculate the fps (frame per second)
        fps = int(self._global_t/nseconds)
        
        if self._lastlog > 100:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            #ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", self._n_updates)
            logger.record_tabular("total_timesteps", self._global_t)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("rewards", sum(sum(rewards)))
            #logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            self._lastlog = 0
        self._lastlog += 1
        return (time_moved * self.n_env, ep_stats, dict())

from keras.layers import Dense, Input, TimeDistributed
from keras import initializers
from common import register_trainer, make_trainer


@register_trainer('test-a2c', episode_log_interval = 10000, save = False)
class SomeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'CartPole-v0'), model_kwargs = dict(), **kwargs)

    def create_inputs(self, name):
        return Input(batch_shape=(self.n_env, None) + self.env.observation_space.shape)

    def create_model(self, inputs):
        from keras.models import Model
        from math import sqrt

        layer_initializer = initializers.Orthogonal(gain=sqrt(2))
        actor = TimeDistributed(Dense(
            units = 64, 
            activation = 'tanh',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(inputs)

        actor = TimeDistributed(Dense(
            units = 64, 
            activation = 'tanh',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(actor)
        actor = TimeDistributed(Dense(
            units = self.env.action_space.n, 
            activation= 'softmax',
            bias_initializer='zeros',
            kernel_initializer = initializers.Orthogonal(gain=0.01)))(actor)


        critic = TimeDistributed(Dense(
            units = 64, 
            activation = 'tanh',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(inputs)

        critic = TimeDistributed(Dense(
            units = 64, 
            activation = 'tanh',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(critic)

        critic = TimeDistributed(Dense(
            units = 1,
            activation = None,
            bias_initializer='zeros',
            kernel_initializer=layer_initializer
        ))(critic)

        return Model(inputs = [inputs], outputs = [actor, critic])


if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()




