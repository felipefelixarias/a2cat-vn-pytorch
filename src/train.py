import argparse
from common import make_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type = str, help = 'Experiment name')
    args = parser.parse_args()
    name = args.name
        
    trainer = make_trainer(name)
    trainer.run()