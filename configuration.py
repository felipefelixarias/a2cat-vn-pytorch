import os
import json

default_configuration = dict(
    #visdom = dict(
    #    server = 'http://localhost',
    #    port = 8097
    #),
    
    house3d = dict(
        framework_path = '/House3D', # '/House3D',
        dataset_path = os.path.expanduser('~/datasets/suncg') # '/datasets/suncg'
    )
)

os.makedirs('~/.visual_navigation', exist_ok=True)
configuration = dict(**default_configuration)
if not os.path.exists('~/.visual_navigation/config'):
    with open('~/.visual_navigation/config', 'w+') as f:
        json.dump(configuration, f)

with open('~/.visual_navigation/config', 'r') as f:
    configuration.update(**json.load(f))
