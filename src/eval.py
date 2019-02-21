from environment.environment import Environment
from common.core import LambdaAgent, RandomAgent
import numpy as np
import gym
import random
from common.console import print_progress
from common import make_agent

def create_baselines(action_space_size, seed = None):
    return [RandomAgent(action_space_size)] + \
        [LambdaAgent('action-%s' % i, lambda _: i) for i in range(action_space_size)]

class Evaluation:
    def __init__(self, env_kwargs, seed = None, number_of_episodes = 1000):
        self._env = gym.make(**env_kwargs) if isinstance(env_kwargs, dict) else env_kwargs
        self._action_space_size = self._env.action_space.n
        self._results = dict()
        self._number_of_episodes = number_of_episodes
        self._histogram_bins = 10
        self._seed = seed or random.random()

    def run(self, agent):
        print('Evaluating %s' % agent.name)
        print_progress(0, self._number_of_episodes)

        if hasattr(self._env, 'seed'):
            self._env.seed(self._seed)

        env = self._env
        env = agent.wrap_env(env)

        episode_lengths = []
        rewards = []
        for i in range(self._number_of_episodes):
            state = env.reset()
            agent.reset_state()

            episode_length = 0
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                episode_length += 1

            episode_lengths.append(episode_length)
            rewards.append(total_reward)

            print_progress(i+1, self._number_of_episodes)
        
        self._results[agent.name] = dict(
            mean_reward = np.mean(rewards),
            mean_episode_length = np.mean(episode_lengths),
            episode_length_hist = np.histogram(episode_lengths),
            episode_lengths = np.array(episode_lengths)
        )

    def run_baselines(self):
        baselines = create_baselines(self._action_space_size)
        for x in baselines:
            self.run(x)

    @property
    def results(self):
        return self._results


def run_evaluation(name, env_kwargs, agents):
    import matplotlib.pyplot as plt
    import csv
    from os import path
    import os

    seed = 1
    bins = 10
    results_dir = './results'
    if not path.exists(results_dir):
        os.makedirs(results_dir)


    eval = Evaluation(env_kwargs, seed = seed)
    for agent in agents:
        eval.run(agent)

    for (key, val) in eval.results.items():
        f = plt.figure()
        plt.hist(val.get('episode_lengths'))
        plt.show()
        f.savefig(path.join(results_dir, '%s-%s-episode-lengths.pdf' % (name, key)), bbox_inches='tight')

    if path.exists(path.join(results_dir, '%s-results.csv' % name)):
        with open(path.join(results_dir, '%s-results.csv' % name), newline='') as inputFile:  
            reader = csv.reader(inputFile)
            rows = list(reader)
    else:
        rows = [['name', 'avg. reward', 'avg. episode length']]

    old_data = { r[0]: r for r in rows[1:] }
    for (key, val) in eval.results.items():
        old_data[key] = [key] + [val['mean_reward'], val['mean_episode_length']]

    rows = [rows[0]] + [x for (key, x) in old_data.items()]

    with open(path.join(results_dir, '%s-results.csv' % name), 'w+') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerows(rows)
        outputFile.flush()

if __name__ == '__main__':
    #run_evaluation(lambda action_space_size: create_baselines(action_space_size))
    def run_dqn(action_space_size, **kwargs):
        from experiments.dqn.dqn_keras import DeepQAgent

        agent = DeepQAgent(action_space_size, './checkpoints')
        return [agent]

    def run_supervised(action_space_size, **kwargs):
        from supervised.experiment import SupervisedAgent, ShortestPathAgent
        return [SupervisedAgent(False), SupervisedAgent(True), ShortestPathAgent()]

    def run_unreal(action_space_size, **kwargs):
        from unreal.agent import UnrealAgent
        return [UnrealAgent(action_space_size, use_goal=True, use_lstm=False)]

    def run_a3c(action_space_size, **kwargs):
        from unreal.agent import UnrealAgent
        return [UnrealAgent(action_space_size, use_goal=True, use_lstm=False, use_pixel_change=False, use_reward_prediction=False, use_value_replay=False)]
    
    #run_evaluation(run_dqn)
    # run_evaluation(run_supervised)
    #run_evaluation(run_a3c)
    import environment.qmaze
    import experiments.qmaze_dqn
    from supervised.experiment import ShortestPathAgent
    # run_evaluation('qmaze', dict(id = 'QMaze-v0'), create_baselines(4))
    #run_evaluation('qmaze', dict(id = 'QMaze-v0'), [make_agent('deepq-qmaze')])
    run_evaluation('qmaze', dict(id = 'QMaze-v0'), [ShortestPathAgent()])

    