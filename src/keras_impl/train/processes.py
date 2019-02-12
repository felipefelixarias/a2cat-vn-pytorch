# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from queue import Queue as queueQueue
from os import path

from datetime import datetime
from multiprocessing import Process, Queue, Value
from environment.environment import create_env

import numpy as np
import time


class ProcessStats(Process):
    def __init__(self, logdir, save_frequency, print_frequency):
        super(ProcessStats, self).__init__()
        self.episode_log_q = Queue(maxsize=100)
        self.episode_count = Value('i', 0)
        self.training_count = Value('i', 0)
        self.should_save_model = Value('i', 0)
        self.trainer_count = Value('i', 0)
        self.predictor_count = Value('i', 0)
        self.agent_count = Value('i', 0)
        self.total_frame_count = 0

        self._results_filename = path.join(logdir, 'train.log')
        self._stat_queue_size = 1000
        self._save_frequency = save_frequency
        self._print_frequency = print_frequency

    def FPS(self):
        # average FPS from the beginning of the training (not current FPS)
        return np.ceil(self.total_frame_count / (time.time() - self.start_time))

    def TPS(self):
        # average TPS from the beginning of the training (not current TPS)
        return np.ceil(self.training_count.value / (time.time() - self.start_time))

    def run(self):
        with open(self._results_filename, 'a') as results_logger:
            rolling_frame_count = 0
            rolling_reward = 0
            results_q = queueQueue(maxsize=self._stat_queue_size)
            
            self.start_time = time.time()
            first_time = datetime.now()
            while True:
                episode_time, reward, length = self.episode_log_q.get()
                results_logger.write('%s, %d, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), reward, length))
                results_logger.flush()

                self.total_frame_count += length
                self.episode_count.value += 1

                rolling_frame_count += length
                rolling_reward += reward

                if results_q.full():
                    old_episode_time, old_reward, old_length = results_q.get()
                    rolling_frame_count -= old_length
                    rolling_reward -= old_reward
                    first_time = old_episode_time

                results_q.put((episode_time, reward, length))

                if self.episode_count.value % self._save_frequency == 0:
                    self.should_save_model.value = 1

                if self.episode_count.value % self._print_frequency == 0:
                    print(
                        '[Time: %8d] '
                        '[Episode: %8d Score: %10.4f] '
                        '[RScore: %10.4f RPPS: %5d] '
                        '[PPS: %5d TPS: %5d] '
                        '[NT: %2d NP: %2d NA: %2d]'
                        % (int(time.time()-self.start_time),
                           self.episode_count.value, reward,
                           rolling_reward / results_q.qsize(),
                           rolling_frame_count / (datetime.now() - first_time).total_seconds(),
                           self.FPS(), self.TPS(),
                           self.trainer_count.value, self.predictor_count.value, self.agent_count.value))
                    sys.stdout.flush()





class ProcessAgent(Process):
    def __init__(self, mode, id, prediction_q, training_q, episode_log_q, gamma,  **kwargs):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self._env = create_env(**kwargs)
        self._num_actions = self._env.action_space.n
        self._mode = mode
        self.actions = np.arange(self._num_actions)

        self._gamma = gamma

        # one frame at a time
        self._wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, gamma, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, -1, 1)
            reward_sum = gamma * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self._num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self._wait_q.get()
        return p, v

    def select_action(self, prediction):
        if self._mode == 'eval':
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self):
        state = self._env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            last_state = state
            prediction, value = self.predict(state)
            action = self.select_action(prediction)
            state, reward, done, _ = self._env.step(action)
            reward_sum += reward
            exp = [last_state, action, reward, state, done]
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))