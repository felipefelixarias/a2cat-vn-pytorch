from environment.environment import Environment
from common.abstraction import LambdaAgent, RandomAgent
import numpy as np

def create_baselines(action_space_size, seed = None):
    return [RandomAgent(action_space_size)] + \
        [LambdaAgent('action-%s' % i, lambda _: i) for i in range(action_space_size)]

class Evaluation:
    def __init__(self, env_kwargs, seed = None):
        self._env = Environment.create_environment(**env_kwargs)
        self._action_space_size = self._env.get_action_size()
        self._results = dict()

        self._max_episode_length = 100 # TODO:remove max episode length
        self._number_of_episodes = 1000
        self._histogram_bins = 10
        self._seed = seed or random.random()

    def run(self, agent):
        if hasattr(self._env, 'seed'):
            self._env.seed(self._seed)

        episode_lengths = []
        rewards = []
        for _ in range(self._number_of_episodes):
            self._env.reset()
            state = self._env.last_state # TODO: change to env

            episode_length = 0
            total_reward = 0
            done = False
            while not done and episode_length < self._max_episode_length:
                action = agent.act(state)
                state, reward, done, _ = self._env.process(action)
                total_reward += reward
                episode_length += 1

            episode_lengths.append(episode_length)
            rewards.append(reward)
        
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


def run_evaluation(agents):
    import matplotlib.pyplot as plt
    import csv
    from os import path
    import os

    env_kwargs = dict(env_type = 'maze', env_name = 'gr')
    seed = 1
    bins = 10
    results_dir = './results'
    if not path.exists(results_dir):
        os.makedirs(results_dir)


    eval = Evaluation(env_kwargs, seed = seed)
    agents = agents(action_space_size = eval._env.get_action_size())
    for agent in agents:
        eval.run(agent)

    for (key, val) in eval.results.items():
        f = plt.figure()
        plt.hist(val.get('episode_lengths'))
        plt.show()
        f.savefig(path.join(results_dir, '%s-episode-lengths.pdf' % key), bbox_inches='tight')

    if path.exists(path.join(results_dir, 'results.csv')):
        with open(path.join(results_dir, 'results.csv'), newline='') as inputFile:  
            reader = csv.reader(inputFile)
            rows = list(reader)
    else:
        rows = [['name', 'avg. reward', 'avg. episode length']]

    old_data = { r[0]: r for r in rows[1:] }
    for (key, val) in eval.results.items():
        old_data[key] = [key] + [val['mean_reward'], val['mean_episode_length']]

    rows = [rows[0]] + [x for (key, x) in old_data.items()]

    with open(path.join(results_dir, 'results.csv'), 'w+') as outputFile:
        writer = csv.writer(outputFile)
        writer.writerows(rows)
        outputFile.flush()

if __name__ == '__main__':
    # run_evaluation(lambda action_space_size: create_baselines(action_space_size))
    def run_dqn(**kwargs):
        from experiments.dqn.dqn_keras import DeepQAgent

        agent = DeepQAgent('./checkpoints')
        return [agent]
    
    run_evaluation(run_dqn) 

    