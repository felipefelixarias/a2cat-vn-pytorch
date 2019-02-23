import argparse
from common import make_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type = str, help = 'Experiment name')
    args = parser.parse_args()
    name = args.name


    package_name = 'experiments.%s' % name.replace('-', '_')
    package = __import__(package_name, fromlist=[''])
    default_args = package.default_args
        
    trainer = make_trainer(name, **default_args())
    trainer.run()