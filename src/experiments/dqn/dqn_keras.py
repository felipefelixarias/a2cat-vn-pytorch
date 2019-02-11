import random
import numpy as np
import tensorflow as tf
import functools
import os, sys

import keras.backend as K

if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0,parentdir) 

from model.model_keras import DeepQModel
from environment.environment import Environment
from train.experience import ExperienceFrame, ExperienceReplay, PrioritizedExperienceReplay

from util.metrics import MetricWriter


def update_target_graph(main_graph, target_graph, tau):
    updated_weights = (np.array(main_graph.get_weights()) * tau) + \
        (np.array(target_graph.get_weights()) * (1 - tau))
    target_graph.set_weights(updated_weights)

class DoubleQLearning:
    def __init__(self, 
                model_fn, 
                env, 
                tau, 
                end_epsilon,
                episode_length, 
                pre_train_steps,
                checkpoint_dir,
                log_dir,
                update_frequency,
                exploration_fraction,
                gamma,
                total_timesteps,
                replay_size,
                goal,
                prioritized_replay,
                prioritized_replay_eps,
                device,
                batch_size,
                **kwargs):
        self._env = env
        self._checkpoint_dir = checkpoint_dir
        self._episode_length = episode_length
        self._pre_train_steps = pre_train_steps
        self._gamma = gamma
        self._prioritized_replay = prioritized_replay
        self._prioritized_replay_eps = prioritized_replay_eps
        self._update_frequency = update_frequency
        self._train_frequency = 1
        self._total_timesteps = total_timesteps
        self._batch_size = batch_size
        self._end_epsilon = end_epsilon
        self._exploration_fraction = exploration_fraction
        self._device = device
        self._goal = goal
        self._replay_size = replay_size
        self._tau = tau
        self._beta_start = 0.4
        self._log_dir = log_dir
        self._main_weights_file = self._checkpoint_dir + "/main_weights.h5" # File to save our main weights to
        self._target_weights_file = self._checkpoint_dir + "/target_weights.h5" # File to save our target weights to

    def _process_state(self, state, action = -1, reward = 0):
        return {'image': state['image'], 
            'goal': state['goal'], 
            'action_reward': ExperienceFrame.concat_action_and_reward(action, self._env.action_space.n, reward, state)}

    def _train_on_experience(self, main_qn, target_qn, experience_replay, t):
        # Train batch is [[state,action,reward,next_state,done],...]
        beta = self._beta_start + (1.0 - self._beta_start) * (t / self._total_timesteps)
        train_batch = experience_replay.sample(self._batch_size, beta = beta)

        # Separate the batch into its components
        train_state, train_action, train_reward, \
            train_next_state, train_done, _, batch_idxes = train_batch
            
        # Convert the action array into an array of ints so they can be used for indexing
        train_action = train_action.astype(np.int)

        # Our predictions (actions to take) from the main Q network
        target_q = target_qn.model.predict([
            train_next_state['image'],
            train_next_state['goal'], 
            train_next_state['action_reward']])
        
        # The Q values from our target network from the next state
        target_q_next_state = main_qn.model.predict([
            train_next_state['image'],
            train_next_state['goal'], 
            train_next_state['action_reward']])

        train_next_state_action = np.argmax(target_q_next_state,axis=1)
        train_next_state_action = train_next_state_action.astype(np.int)
        
        # Tells us whether game over or not
        # We will multiply our rewards by this value
        # to ensure we don't train on the last move
        train_gameover = train_done == 0

        # Q value of the next state based on action
        train_next_state_values = target_q_next_state[range(self._batch_size), train_next_state_action]

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (self._gamma * train_next_state_values * train_gameover)
        target_q[range(self._batch_size), train_action] = actual_reward
        
        # Train the main model
        train_return = main_qn.model.train_on_batch([train_state['image'],
            train_state['goal'],
            train_state['action_reward']], target_q)

        loss = train_return[0]
        q_output = train_return[1]

        def compute_priorities():
            td_errors = q_output[range(self._batch_size), train_action] - actual_reward
            return np.abs(td_errors) + self._prioritized_replay_eps

        experience_replay.update_priorities(batch_idxes, priorities = compute_priorities)
        return loss

    def run(self):
        # Reset everything
        K.clear_session()

        # Setup our Q-networks
        main_qn = DeepQModel(self._env.action_space.n, device = self._device)
        target_qn = DeepQModel(self._env.action_space.n, device = self._device)

        # Make the networks equal
        update_target_graph(main_qn.model, target_qn.model, 1)

        # Setup our experience replay
        experience_replay = PrioritizedExperienceReplay(self._replay_size) if self._prioritized_replay else ExperienceReplay(self._replay_size)

        # We'll begin by acting complete randomly. As we gain experience and improve,
        # we will begin reducing the probability of acting randomly, and instead
        # take the actions that our Q network suggests
        prob_random = 1.0
        prob_random_drop = (1.0 - self._end_epsilon) / (self._total_timesteps * self._exploration_fraction)

        num_steps = [] # Tracks number of steps per episode
        rewards = [] # Tracks rewards per episode

        print_every = 100 # How often to print status
        save_every = 10000 # How often to save

        # Setup path for saving
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        metric_writer = MetricWriter(logdir = self._log_dir)

        if os.path.exists(self._main_weights_file):
            print("Loading main weights")
            main_qn.model.load_weights(self._main_weights_file)
        if os.path.exists(self._target_weights_file):
            print("Loading target weights")
            target_qn.model.load_weights(self._target_weights_file)

        # Create an experience replay for the current episode
        episode_buffer = []
        cur_step = 0 # Running sum of number of steps taken in episode
        cur_episode = 0
        sum_loss = 0
        loss_updates = 0
        done = False
        sum_rewards = 0 # Running sum of rewards in episode
        state = self._process_state(self._env.reset())

        # Log stats
        mean_loss = float('nan')
        mean_reward = float('nan')
        mean_episode_length = float('nan')

        for global_t in range(self._total_timesteps):
            if done:
                # At the end of the episode
                # We copy the episode experience to buffer
                experience_replay.extend(episode_buffer)
                episode_buffer.clear()

                # We collect the statistics for the episode
                if global_t > 0:
                    cur_episode += 1
                    rewards.append(sum_rewards)
                    num_steps.append(cur_step)
                
                cur_step = 0 # Running sum of number of steps taken in episode
                sum_rewards = 0 # Running sum of rewards in episode
                
                # Reset the episode
                state = self._process_state(self._env.reset())                
            
            # Exploration phase
            cur_step += 1
            if np.random.rand() < prob_random or \
                global_t < self._pre_train_steps:
                # Act randomly based on prob_random or if we
                # have not accumulated enough pre_train episodes
                action = np.random.randint(self._env.action_space.n)
            else:
                # Decide what action to take from the Q network
                action = np.argmax(main_qn.model.predict([[state['image']], [state['goal']], [state['action_reward']]]))

            # Take the action and retrieve the next state, reward and done
            next_state, reward, done, _ = self._env.step(action)
            next_state = self._process_state(next_state, action, reward)

            episode_buffer.append([state,action,reward,next_state,done])
            sum_rewards += reward
            state = next_state
            # End of exploration phase

            if global_t > self._pre_train_steps:
                # Training the network

                if prob_random > self._end_epsilon:
                    # Drop the probability of a random action
                    prob_random -= prob_random_drop

                if global_t % self._train_frequency == 0:
                    loss = self._train_on_experience(main_qn, target_qn, experience_replay, global_t)
                    sum_loss += loss
                    loss_updates += 1

                if global_t > self._pre_train_steps and global_t % self._update_frequency == 0:         
                    # Update the target model with values from the main model
                    update_target_graph(main_qn.model, target_qn.model, self._tau)

                if global_t % save_every == 0:
                    # Save the model
                    main_qn.model.save_weights(self._main_weights_file)
                    target_qn.model.save_weights(self._target_weights_file)
            
            if global_t % print_every == 0 and global_t != 0:
                # Print progress
                mean_loss = sum_loss / loss_updates if loss_updates > 0 else mean_loss
                mean_reward = np.mean(rewards) if len(rewards) > 0 else mean_reward
                mean_episode_length = np.mean(num_steps) if len(num_steps) > 0 else mean_episode_length

                rewards.clear()
                num_steps.clear()
                

                metrics_row = metric_writer \
                    .record(global_t) \
                    .scalar('epsilon', prob_random) \
                    .scalar('reward', mean_reward) \
                    .scalar('episode_length', mean_episode_length)
                    
                if loss_updates > 0:
                    metrics_row = metrics_row.scalar('loss', mean_loss)

                sum_loss = 0
                loss_updates = 0
                metrics_row.flush()


                print("Num step: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                    global_t, mean_reward, prob_random, mean_loss))

def get_options():
    tf.app.flags.DEFINE_integer('batch_size', 32, 'How many experiences to use for each training step.')
    tf.app.flags.DEFINE_integer('update_frequency', 500, 'How often to perform a training step.')
    tf.app.flags.DEFINE_float('gamma', .99, 'Discount factor on the target Q-values')
        
    tf.app.flags.DEFINE_float('end_epsilon', 0.02, 'Final chance of random action')
    tf.app.flags.DEFINE_float('exploration_fraction', 0.1, 'Percentage of training spend on exploration')
    tf.app.flags.DEFINE_integer('pre_train_steps', 1000, 'How many steps of random actions before training begins.')
    tf.app.flags.DEFINE_integer('total_timesteps', 100000, 'Total number of training steps.')
    tf.app.flags.DEFINE_integer('episode_length', 50, 'The max allowed length of our episode.')
    tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoints", "checkpoint directory")
    tf.app.flags.DEFINE_integer("replay_size", 50000, "Replay buffer size")
    tf.app.flags.DEFINE_boolean("prioritized_replay", True, "Use prioritized replay")
    tf.app.flags.DEFINE_float("prioritized_replay_eps", 10e-6, "Use prioritized replay epsilon")
    tf.app.flags.DEFINE_string("log_dir", "./logs", "log file directory")
    tf.app.flags.DEFINE_float('tau', 0.001, 'Rate to update target network toward primary network')
    tf.app.flags.DEFINE_float('goal', None, 'Target reward (-1) if none')
    return tf.app.flags.FLAGS

class Application:
    def __init__(self, flags):
        self._flags = flags
        pass

    def run(self):
        
        device = "/gpu:0"
        env = Environment.create_environment('maze', 'gr')

        model_fn = lambda name, device: DeepQModel(
                env.get_action_size(),
                0,
                thread_index = name, # -1 for global
                use_lstm = False,
                use_pixel_change = False,
                use_value_replay = False,
                use_reward_prediction = False,
                use_deepq_network = True,
                use_goal_input = env.can_use_goal(),              
                pixel_change_lambda = .05,
                entropy_beta = .001,
                device = device,
            )

        learning = DoubleQLearning(model_fn, env.get_env(), device = device, **self._flags.flag_values_dict())
        learning.run()

if __name__ == '__main__':
    flags = get_options()
    tf.app.run(lambda _: Application(flags).run())