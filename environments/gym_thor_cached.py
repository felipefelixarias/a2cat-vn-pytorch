import gym
import os
import random
import h5py


class THORCachedEnv(gym.Env):
    
    @staticmethod
    def _get_h5_file_path(scene_name):
        path = '/media/data/datasets/visual_navigation_precomputed'
        if 'THOR_DATASET_PATH' in os.environ:
            path = os.environ['THOR_DATASET_PATH']

        return "%s/%s.h5" % (path, scene_name)

    def __init__(self, tasks, image_size = (84,84), **kwargs):
        super(THORCachedEnv, self).__init__()
        self._random = random.Random()
        self.scenes = dict()
        self.tasks = tasks
        self.image_size = image_size
        self.reset()

    def ensure_scene_loaded(self, name):
        if not name in self.scenes:
            h5_file_path = THORCachedEnv._get_h5_file_path(name)
            with h5py.File(h5_file_path, 'r') as f:
                self.scenes[name] = dict(
                    locations = f['location'][()].shape[0],
                    transition_graph = f['graph'][()],
                    observations = f['observation'][()],
                    shortest_path_distances = f['shortest_path_distance'][()]
                )
        return self.scenes[name]

    def _sample_start(self):
        current_state = None
        while True:
            current_state = self._random.randrange(self.current_scene['locations'])
            if self.current_scene['shortest_path_distances'][current_state][self.goal] > 0:
                break
        return current_state

    def reset(self, initial_state_id = None):
        # randomize initial state and goal
        scene_id, self.goal = self._random.choice(self.tasks)
        self.current_scene = self.ensure_scene_loaded(scene_id)
        self.state = self._sample_start()
        return self.observe()

    def observe(self):
        return self.current_scene['observations'][self.state, :, :, :], self.current_scene['observations'][self.goal, :, :, :]

    def _render_observation(self, idx):
        return self.current_scene['observations'][idx, :, :, :]

    def _preprocess_frame(self, image):        
        image = image.astype(np.float32)
        image = image / 255.0
        image = resize(image, self.image_size, anti_aliasing=True)
        return image

    @staticmethod
    def get_action_size(env_name):
        return 4

    @property
    def reward_configuration(self):
        return (1.0, 0.0, 0.0)

    def process(self, action):
        collided = False
        if self._transition_graph[self._current_state_idx][action] != -1:
            self._current_state_idx = self._transition_graph[self._current_state_idx][action]
        else:
            collided = True

        obs = self._render_observation(self._current_state_idx)
        goal = self._render_observation(self._current_goal_idx)
        terminal = self._current_goal_idx == self._current_state_idx
        reward = -self.reward_configuration[1]
        if terminal:
            reward = self.reward_configuration[0]
        if collided:
            reward = self.reward_configuration[2]

        if not terminal:
            state = { 
                'image': self._preprocess_frame(obs),
                'goal': self._preprocess_frame(goal)
            }
        else:
            state = self.last_state
        
        return state, reward, terminal, dict()