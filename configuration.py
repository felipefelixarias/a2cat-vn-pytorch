import os
configuration = dict(
    visdom = dict(
        server = 'http://localhost',
        port = 8097
    ),
    
    house3d = dict(
        framework_path = os.path.expanduser('~/toolbox/House3D'), # '/House3D',
        dataset_path = '/media/data/datasets/suncg' # '/datasets/suncg'
    )
)