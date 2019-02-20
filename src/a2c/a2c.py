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
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses

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

        return obs, actions, returns[:,:-1]
        

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
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
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
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
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

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    if load_path is not None:
        model.load(load_path)

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
        self.n_steps = 20
        self.n_env = 5
        self.total_timesteps = 1000000
        self.gamma = 0.95
        self.entropy_coef = 0.01
        self.learning_rate = 0.001
        self.max_grad_norm = 40.0

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
        (obs, actions, returns), ep_stats = self._sample_experience_batch()
        time_moved = obs.shape[1]

        loss, policy_entropy, policy_loss, value_loss = self._train(obs, returns, actions)
        #policy_loss, value_loss, policy_entropy = self.model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-self._tstart
        self._global_t += time_moved * self.n_env
        self._n_updates += 1

        # Calculate the fps (frame per second)
        fps = int(self._global_t/nseconds)
        
        '''if self._lastlog > 100:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            #ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", self._n_updates)
            logger.record_tabular("total_timesteps", self._global_t)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("rewards", sum(sum(batch[2])))
            #logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            self._lastlog = 0
        self._lastlog += 1
        '''
        return (time_moved * self.n_env, ep_stats, dict())

from keras.layers import Dense, Input, TimeDistributed
from keras import initializers
from common import register_trainer, make_trainer


@register_trainer('test-a2c', episode_log_interval = 100, save = False)
class SomeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'QMaze-v0'), model_kwargs = dict(), **kwargs)

    def create_inputs(self, name):
        return Input(batch_shape=(self.n_env, None, 49))

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
            units = 4, 
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




