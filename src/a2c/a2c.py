from abc import abstractclassmethod
from trfl import sequence_advantage_actor_critic_loss
import keras.backend as K

import gym
import time
from common.vec_env import SubprocVecEnv
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




class Trainer:
    def __init__(self):
        self.n_steps = 5
        self.n_env = 3
        self.env_kwargs = dict(id = 'QMaze-v0')
        self.total_timesteps = 1000000
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.max_grad_norm = 40.0

    @abstractclassmethod
    def create_model(self, inputs):
        pass

    @abstractclassmethod
    def create_inputs(self, name = 'main'):
        pass

    def _initialize(self):
        self.env = SubprocVecEnv([lambda: gym.make(**self.env_kwargs) for _ in range(self.n_env)])
        self._build_graph()

        #self.obs = np.zeros((self.n_env,) + self.env.observation_space.shape, dtype=self.env.observation_space.dtype.name)
        #self.obs[:] = self.env.reset()
        self.obs = None

        self.batch_ob_shape = (self.n_env*self.n_steps,) + self.env.observation_space.shape
        self._experience = []

    def _build_graph(self):
        sess = K.get_session()

        inputs = self.create_inputs('main')
        model = self.create_model(inputs)

        with tf.name_scope('training'):
            actions = tf.placeholder(dtype = tf.int32, shape = (None, self.n_env), name = 'actions')
            rewards = tf.placeholder(dtype = tf.float32, shape = (None, self.n_env), name = 'rewards')
            terminals = tf.placeholder(dtype = tf.bool, shape = (self.n_env), name = 'terminals')
            bootstrap_value = tf.placeholder(dtype = tf.float32, shape = (self.n_env), name = 'bootstrap_value')
            learning_rate = tf.placeholder(dtype = tf.float32, name = 'learning_rate')

            last_pcontinues = 1.0 - tf.to_float(terminals) * self.gamma
            pcontinues = tf.ones_like(rewards[:-1]) * self.gamma
            pcontinues = tf.concat([pcontinues, tf.reshape(last_pcontinues, [1, -1])], axis = 0)

            policy_logits, baseline_values = model.outputs
            policy_logits = tf.transpose(policy_logits, [1, 0, 2])
            baseline_values = tf.reshape(tf.transpose(baseline_values, [1, 0, 2]), [self.n_env, -1])
            
            losses, extra = sequence_advantage_actor_critic_loss(policy_logits, baseline_values, actions, rewards, pcontinues, bootstrap_value)
            loss = tf.reduce_mean(losses)

            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)

            params = model.trainable_weights
            grads = tf.gradients(loss, params)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads = list(zip(grads, params))

            # Create optimize op
            optimize_op = optimizer.apply_gradients(grads)


        sess.run(tf.global_variables_initializer())

        def train(b_inputs, b_bactions, b_rewards, b_terminals, b_bootstrap_value):
            return sess.run([loss, optimize_op], feed_dict = {
                learning_rate: self.learning_rate,
                inputs: b_inputs,
                actions:b_bactions,
                rewards: b_rewards,
                terminals: b_terminals,
                bootstrap_value: b_bootstrap_value
            })[0]

        def predict_single(s_inputs):
            res = sess.run([policy_logits, baseline_values], feed_dict = {
                inputs: np.expand_dims(s_inputs, 1)
            })
            
            return (res[0][:, 0], res[1][:, 0])
        
        self._train = train
        self._predict_single = predict_single

    def act(self, state):
        pvalues, baseline_values = self._predict_single(state)

        s = pvalues.cumsum(axis=1)
        r = np.random.rand(pvalues.shape[0])
        actions = (s < r).sum(axis=1)
        return actions, baseline_values

    def _sample_experience_batch(self):
        if self.obs is None:
            self.obs = self.env.reset()

        batch = []
        for n in range(self.n_steps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values = self.act(self.obs)
            obs, rewards, dones, _ = self.env.step(actions)
            batch.append([np.copy(self.obs), actions, rewards])
            self.obs = obs
            
            if np.any(dones):
                # Any environment ended
                # We need to end this cycle
                # And return uncomplete minibatch
                break
        
        _, bootstrap_values = self.act(self.obs)
        bootstrap_values = (1.0 - dones) * bootstrap_values # Remove value of finished episode

        # Convert data to acceptable format
        batched = tuple(map(lambda *x: np.stack(x, axis = 1), *batch))
        batched = batched + (dones, bootstrap_values,)
        return batched

    def _get_end_stats(self):
        return None

    def run(self):
        nbatch = self.n_env * self.n_steps
        for update in range(1, self.total_timesteps//nbatch+1):
            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values = self._sample_experience_batch()

            policy_loss, value_loss, policy_entropy = self.model.train(obs, states, rewards, masks, actions, values)
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
                logger.dump_tabular()

        return (nbatch, self._get_end_stats(), dict())

from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model

class SomeTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def create_inputs(self, name):
        return Input(batch_shape=(self.n_env, None, 49))

    def create_model(self, inputs):
        model = TimeDistributed(Dense(256, activation = 'relu'))(inputs)
        policy_logits = TimeDistributed(Dense(4, activation= 'softmax'))(model)
        baseline_values = TimeDistributed(Dense(1, activation = None))(model)
        return Model(inputs = [inputs], outputs = [policy_logits, baseline_values])




