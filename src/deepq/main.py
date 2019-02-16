if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import deepq.dqn
from common.train_wrappers import wrap
import gym
import gym_maze
import deepq.catch_experiment

if __name__ == '__main__':
    trainer = deepq.dqn.DeepQTrainer(
        env_kwargs = dict(id='GridWorld-v0'), 
        model_kwargs = dict(action_space_size = 4),
        annealing_steps = 1000 * 50,
        max_episode_steps = 50)

    trainer = wrap(trainer, max_number_of_episodes=10000, episode_log_interval=10)
    trainer.run()

else:
    raise('This script cannot be imported')