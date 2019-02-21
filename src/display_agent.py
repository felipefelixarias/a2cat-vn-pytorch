from environment.environment import Environment
from common import make_agent
from supervised.experiment import SupervisedAgent, ShortestPathAgent
import numpy as np
import pygame
import gym
import random
import gym_maze

def create_baselines(action_space_size, seed = None):
    return [RandomAgent(action_space_size)] + \
        [LambdaAgent('action-%s' % i, lambda _: i) for i in range(action_space_size)]

class Display:
    def __init__(self, env_kwargs, agent, seed = None):
        self._env = gym.make(**env_kwargs) if isinstance(env_kwargs, dict) else env_kwargs
        # self._env = gym.wrappers.TimeLimit(self._env, max_episode_steps=50)
        self._action_space_size = self._env.action_space.n
        self._results = dict()
        self._number_of_episodes = 1000
        self._histogram_bins = 10
        self._seed = seed or random.random()

        pygame.init()
    
        self.surface = pygame.display.set_mode((440,440,), 0, 24)
        self.font = pygame.font.SysFont(None, 20)
        pygame.display.set_caption('visual-navigation')
        
        self.agent = agent
        self._env = agent.wrap_env(self._env)
        self._env.seed(self._seed)

        self._episode_length = 0
        self._total_reward = 0
        self._state = self._env.reset()
        self.agent.reset_state()

    def update(self):
        self.surface.fill((255,255,255,))
        self.process()
        pygame.display.update()

    def scale_image(self, image, scale):
        return image.repeat(scale, axis=0).repeat(scale, axis=1)

    def draw_text(self, str, left, top, color=(0,0,0)):
        text = self.font.render(str, True, color, (255,255,255))
        text_rect = text.get_rect()
        text_rect.left = left    
        text_rect.top = top
        self.surface.blit(text, text_rect)  

    def draw_center_text(self, str, center_x, top):
        text = self.font.render(str, True, (0,0,0), (255,255,255))
        text_rect = text.get_rect()
        text_rect.centerx = center_x
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def show_image(self, state):
        image = pygame.image.frombuffer(state, (300,300), 'RGB')
        self.surface.blit(image, (8, 8))
        self.draw_center_text("input", 150, 316)

    def show_goal(self, state):
        image = pygame.image.frombuffer(state, (84,84), 'RGB')
        self.surface.blit(image, (100, 8))
        # self.draw_center_text("goal", 50, 100)


    def process(self):
        self.step()
        state = self._env.unwrapped.render(mode = 'rgbarray')
        self.show_image(state)
        #self.show_image(state['observation'])
        #self.show_goal(state['desired_goal'])

    def step(self):
        action = self.agent.act(self._state)
        self._state, reward, done, _ = self._env.step(action)
        self._total_reward += reward
        self._episode_length += 1

        if done:
            self._episode_length = 0
            self._total_reward = 0
            self._state = self._env.reset()
  

def run_agent(agent, env = dict(id = 'GoalMaze-v0', fixed_goal = True)):
    seed = 1

    display = Display(env, agent, seed = seed)

    clock = pygame.time.Clock()
  
    running = True
    FPS = 15

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    display._state = display._env.reset()

        
        display.update()
        clock.tick(FPS)


if __name__ == '__main__':
    def run_dqn(action_space_size, **kwargs):
        from experiments.dqn.dqn_keras import DeepQAgent

        agent = DeepQAgent(action_space_size, './checkpoints')
        return [agent]

    def run_supervised_deterministic(action_space_size, **kwargs):
        from experiments.supervised.experiment import SupervisedAgent
        return [SupervisedAgent(action_space_size, './checkpoints', is_deterministic = True)]

    def run_unreal(action_space_size, **kwargs):
        from unreal.agent import UnrealAgent
        return [UnrealAgent(action_space_size, use_goal=True, use_lstm=False)]

    def run_a3c(action_space_size, **kwargs):
        from unreal.agent import UnrealAgent
        return [UnrealAgent(action_space_size, use_goal=True, use_lstm=False, use_pixel_change=False, use_reward_prediction=False, use_value_replay=False)]
    
    #run_evaluation(run_dqn)
    #run_evaluation(run_supervised_deterministic)
    # run_evaluation(run_a3c)
    # run_agent(SupervisedAgent())

    from graph.env import SimpleGraphEnv
    from graph.util import load_graph
    with open('./scenes/dungeon-20-1.pkl', 'rb') as f:
        graph = load_graph(f)
    import experiments.dungeon_dqn
    import experiments.dungeon_dqn_dynamic_complexity


    env = SimpleGraphEnv(graph, graph.goal)
    run_agent(make_agent('deepq-dungeon'), env)