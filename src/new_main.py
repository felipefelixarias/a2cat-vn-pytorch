


import keras_impl.new.a3c

if __name__ == '__main__':
    trainer = keras_impl.new.a3c.Trainer(500000, dict(action_space_size = 4, head = 'ac'))
    trainer.run()