# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import time

from common.env_wrappers import UnrealObservationWrapper
from environment.environment import Environment
from unreal.model import UnrealModel
from collections import namedtuple

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

ExperienceFrame = namedtuple('ExperienceFrame', ['state', 'done', 'new_pixel_change', 'new_action_reward'])

class Trainer(object):
    def __init__(self,
            thread_index,
            global_network,
            initial_learning_rate,
            learning_rate_input,
            local_t_max,
            grad_applier,
            env_kwargs,
            gamma,
            gamma_pc,
            experience_history_size,
            max_global_time_step,
            device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.env_kwargs = env_kwargs
        self._set_properties(**global_network.settings)
        self.local_t_max = local_t_max
       
        self.gamma = gamma
        self.gamma_pc = gamma_pc
        self.experience_history_size = experience_history_size
        self.max_global_time_step = max_global_time_step
        
        self.local_network = UnrealModel(device = device, **global_network.settings)
        self.local_network.prepare_loss()

        self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                                                                             global_network.get_vars(),
                                                                                                             self.local_network.get_vars())
        
        self.sync = self.local_network.sync_from(global_network)
        self.experience = Experience(self.experience_history_size)
        self.local_t = 0
        self.initial_learning_rate = initial_learning_rate
        self.episode_reward = 0
        # For log output
        self.prev_local_t = 0

    def _set_properties(self, use_pixel_change, use_value_replay, use_reward_prediction, action_space_size, **kwargs):
        self.use_pixel_change = use_pixel_change
        self.use_value_replay = use_value_replay
        self.use_reward_prediction = use_reward_prediction
        self.action_space_size = action_space_size

    def prepare(self):
        self.environment = UnrealObservationWrapper(gym.make(**self.env_kwargs))
        self._last_state = self.environment.reset()

    def stop(self):
        self.environment.close()
        
    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    
    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    
    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

        
    def set_start_time(self, start_time):
        self.start_time = start_time


    def _fill_experience(self, sess):
        """
        Fill experience buffer until buffer is full.
        """
        
        
        policy, _ = self.local_network.run_base_policy_and_value(sess, self._last_state)
        action = self.choose_action(policy)
        
        new_state, _, done, _ = self.environment.step(action)
        
        frame = ExperienceFrame(self._last_state, done, new_state['pixel_change'], new_state['last_action_reward'])
        self.experience.add_frame(frame)
        self._last_state = new_state
        
        if done:
            self._last_state = self.environment.reset()
        if self.experience.is_full():
            self._last_state = self.environment.reset()
            print("Replay buffer filled")


    def _print_log(self, global_t):
        if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,    elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
        

    def _process_base(self, sess, global_t, summary_writer, summary_op, score_input):
        # [Base A3C]
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        start_lstm_state = self.local_network.base_lstm_state_out

        # t_max times loop
        for _ in range(self.local_t_max):
            # Prepare last action reward
            policy, value = self.local_network.run_base_policy_and_value(sess, self._last_state)
            
            
            action = self.choose_action(policy)

            states.append(self._last_state)
            actions.append(action)
            values.append(value)

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(policy))
                print(" V={}".format(value))

            last_state = self._last_state

            # Process game
            new_state, reward, done, _ = self.environment.step(action)
            frame = ExperienceFrame(last_state, done, new_state['pixel_change'], new_state['last_action_reward'])
            self._last_state = new_state

            # Store to experience
            self.experience.add_frame(frame)

            self.episode_reward += reward
            self.local_t += 1
            
            rewards.append(reward)
            
            if done:
                terminal_end = True
                print("score={}".format(self.episode_reward))

                self._record_score(sess, summary_writer, summary_op, score_input,
                                                     self.episode_reward, global_t)
                    
                self.episode_reward = 0
                self.environment.reset()
                self.local_network.reset_state()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_base_value(sess, self._last_state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_adv = []
        batch_R = []

        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + self.gamma * R
            adv = R - Vi
            a = np.zeros([self.action_space_size])
            a[ai] = 1.0

            batch_si.append(si)
            batch_a.append(a)
            batch_adv.append(adv)
            batch_R.append(R)

        batch_si.reverse()
        batch_a.reverse()
        batch_adv.reverse()
        batch_R.reverse()
        
        return batch_si, batch_a, batch_adv, batch_R, start_lstm_state

    
    def _process_pc(self, sess):
        # [pixel change]
        # Sample 20+1 frame (+1 for last next state)
        pc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        # Reverse sequence to calculate from the last
        pc_experience_frames.reverse()

        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []
        
        pc_R = np.zeros([20,20], dtype=np.float32)
        if not pc_experience_frames[1].done:
            pc_R = self.local_network.run_pc_q_max(sess, pc_experience_frames[0].state)


        for frame in pc_experience_frames[1:]:
            pc_R = frame.pixel_change + self.gamma_pc * pc_R
            a = np.zeros([self.action_space_size])
            a[frame.action] = 1.0
            
            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)

        batch_pc_si.reverse()
        batch_pc_a.reverse()
        batch_pc_R.reverse()
        
        return batch_pc_si, batch_pc_a, batch_pc_R

    
    def _process_vr(self, sess):
        # [Value replay]
        # Sample 20+1 frame (+1 for last next state)
        vr_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
        # Reverse sequence to calculate from the last
        vr_experience_frames.reverse()

        batch_vr_si = []
        batch_vr_R = []

        vr_R = 0.0
        if not vr_experience_frames[1].done:
            vr_R = self.local_network.run_vr_value(sess, vr_experience_frames[0].state)
        
        # t_max times loop
        for frame in vr_experience_frames[1:]:
            vr_R = frame.reward + self.gamma * vr_R
            batch_vr_si.append(frame.state)
            batch_vr_R.append(vr_R)

        batch_vr_si.reverse()
        batch_vr_R.reverse()

        return batch_vr_si, batch_vr_R

    
    def _process_rp(self):
        # [Reward prediction]
        rp_experience_frames = self.experience.sample_rp_sequence()
        # 4 frames

        batch_rp_si = []
        batch_rp_c = []
        
        for i in range(3):
            batch_rp_si.append(rp_experience_frames[i].state)

        # one hot vector for target reward
        r = rp_experience_frames[3].reward
        rp_c = [0.0, 0.0, 0.0]
        if r == 0:
            rp_c[0] = 1.0 # zero
        elif r > 0:
            rp_c[1] = 1.0 # positive
        else:
            rp_c[2] = 1.0 # negative
        batch_rp_c.append(rp_c)
        return batch_rp_si, batch_rp_c
    
    
    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        # Fill experience replay buffer
        if not self.experience.is_full():
            self._fill_experience(sess)
            return 0

        start_local_t = self.local_t

        cur_learning_rate = self._anneal_learning_rate(global_t)

        # Copy weights from shared to local
        sess.run( self.sync )

        # [Base]
        batch_si, batch_a, batch_adv, batch_R, start_lstm_state = \
                    self._process_base(sess,
                                                         global_t,
                                                         summary_writer,
                                                         summary_op,
                                                         score_input)
        feed_dict = {
            self.local_network.base_input: [x['observation'] for x in batch_si],
            self.local_network.base_last_action_reward_input: [x['last_action_reward'] for x in batch_si],
            self.local_network.base_a: batch_a,
            self.local_network.base_adv: batch_adv,
            self.local_network.base_r: batch_R,
            self.local_network.base_initial_lstm_state: start_lstm_state,
            # [common]
            self.learning_rate_input: cur_learning_rate
        }

        # TODO: use goal here

        # [Pixel change]
        if self.use_pixel_change:
            batch_pc_si, batch_pc_a, batch_pc_R = self._process_pc(sess)

            pc_feed_dict = {
                self.local_network.pc_input: [x['observation'] for x in batch_pc_si],
                self.local_network.pc_last_action_reward_input: [x['last_action_reward'] for x in batch_pc_si],
                self.local_network.pc_a: batch_pc_a,
                self.local_network.pc_r: batch_pc_R
            }
            feed_dict.update(pc_feed_dict)

        # [Value replay]
        if self.use_value_replay:
            batch_vr_si, batch_vr_R = self._process_vr(sess)
            
            vr_feed_dict = {
                self.local_network.vr_input: [x['observation'] for x in batch_vr_si],
                self.local_network.vr_last_action_reward_input : [x['last_action_reward'] for x in batch_vr_si],
                self.local_network.vr_r: batch_vr_R
            }
            feed_dict.update(vr_feed_dict)

        # [Reward prediction]
        if self.use_reward_prediction:
            batch_rp_si, batch_rp_c = self._process_rp()
            rp_feed_dict = {
                self.local_network.rp_input: batch_rp_si,
                self.local_network.rp_c_target: batch_rp_c
            }
            feed_dict.update(rp_feed_dict)

        # Calculate gradients and copy them to global network.
        sess.run( self.apply_gradients, feed_dict=feed_dict )
        
        self._print_log(global_t)
        
        # Return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t