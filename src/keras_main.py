




from keras_impl.train.server import Server
from environment.environment import Environment

env_kwargs = dict(env_type = 'maze', env_name = 'gr')
kwargs = dict(
    action_space_size = Environment.get_action_size(env_kwargs['env_type'], env_kwargs['env_name']),
    min_training_batch_size = 0,
    batch_size = 128,
    device = '/gpu:0',
    checkpoint_dir = './checkpoints',
    logdir = './logs',
    save_frequency = 1000,
    gamma = 0.99,
    print_frequency = 1,
    total_episodes = 400000,
    beta = (0.01, 0.01),
    learning_rate = (0.0003, 0.0003),
    **env_kwargs
)

Server('train', **kwargs).run()