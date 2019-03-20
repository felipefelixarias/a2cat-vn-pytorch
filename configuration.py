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

basepath = os.path.expanduser('~/.visual_navigation')
os.makedirs(basepath, exist_ok=True)
configuration = dict(**default_configuration)
if not os.path.exists(os.path.join(basepath, 'config')):
    with open(os.path.join(basepath, 'config'), 'w+') as f:
        json.dump(configuration, f)

with open(os.path.join(basepath, 'config'), 'r') as f:
    configuration.update(**json.load(f))
