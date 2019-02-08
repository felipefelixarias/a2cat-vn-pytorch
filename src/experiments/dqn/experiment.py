import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir) 

import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


import gym
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

from environment.environment import Environment
from experiments.dqn.model import Model
from experiments.dqn.options import get_options
from experiments.dqn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


flags = get_options("training")

def wrap_environment(env, action_space_size, use_goal_input):
    class EnvironmentWrap:
        def __init__(self, env, action_space_size, use_goal_input):
            self.env = env
            self.action_space_size = action_space_size
            self.use_goal_input = use_goal_input

        def reset(self):
            self.env.reset()
            return self._transform_observation(self.env.last_state)

        @property
        def observation_space(self):
            space = gym.spaces.Box(0.0, 1.0, shape=(84, 84, 6 if self.use_goal_input else 3))
            space.dtype = np.float32
            return space

        @property
        def action_space(self):
            return gym.spaces.Discrete(self.action_space_size)

        def step(self, action):
            new_obs, rew, done, _ = self.env.process(action)


            return (self._transform_observation(new_obs), rew, done, None)

        def stop(self):
            self.env.stop()

        def _transform_observation(self, state):
            if self.use_goal_input:
                return np.concatenate((state['image'], state['goal']), 2)
            else:
                return state['image']

    return EnvironmentWrap(env, action_space_size, use_goal_input)


def create_model(use_goal):
    def model_fn(inpt, num_actions, scope, reuse = False):
        if use_goal:
            (base_input, goal_input) = tf.split(inpt, 2, 3)
            kwargs = dict(base_input = base_input, goal_input = goal_input)
        else:
            kwargs = dict(base_input = inpt)
        m = Model(num_actions, 
            Environment.get_objective_size(flags.env_type, flags.env_name), 
            use_goal,
            None,
            reuse = reuse,
            name = scope, 
            **kwargs)
        return m.base_pi
    return model_fn

if __name__ == '__main__':
    with U.make_session(num_cpu=8):
        # Create the environment
        use_goal = Environment.can_use_goal(flags.env_type, flags.env_name)
        env_original = Environment.create_environment(flags.env_type, flags.env_name)
        env = wrap_environment(env_original, 
            Environment.get_action_size(flags.env_type, flags.env_name),
            use_goal)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph= lambda name: ObservationInput(env.observation_space, name=name),
            q_func=create_model(use_goal),
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                # TODO: show result
                # Store the solution
                pass
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()