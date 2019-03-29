import argparse
from deep_rl import make_agent
import deep_rl
from configuration import configuration
from deep_rl.common.torchsummary import get_shape
import os
from environments.gym_house.video import RenderVideoWrapper
import environments

EXPERIMENTS = [('00cfe094634578865b4384f3adef49e6', [
        ((40.8775749206543, 39.093448638916016, 71.8712387084961),'kitchen'),
    ])
]

def record_videos(agent, path, screen_size):
    seed = 1
    for scene, tasks in EXPERIMENTS:
        env = environments.make('GoalHouse-v1', screen_size = screen_size, scene = scene, goals = [])
        env = RenderVideoWrapper(env, path)
        env = agent.wrap_env(env)
        env.seed(seed)
        for task in tasks:
            agent.reset_state()
            env.unwrapped.set_next_task(task)
            obs = env.reset()
            done = False            
            while not done:
                obs, _, done, _ = env.step(agent.act(obs))


if __name__ == '__main__':
    deep_rl.configure(**configuration)

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type = str, help = 'Experiment name')
    args = parser.parse_args()
    name = args.name


    package_name = 'experiments.%s' % name.replace('-', '_')
    package = __import__(package_name, fromlist=[''])
    screen_size = package.default_args()['env_kwargs'].get('screen_size', None)

    videos_path = os.path.join(configuration.get('videos_path'), name)
    os.makedirs(videos_path, exist_ok=True)

    agent = make_agent(name)
    record_videos(agent, videos_path, screen_size)