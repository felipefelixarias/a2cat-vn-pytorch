from keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, Lambda
from keras.models import Model

def atari(inputs, action_space_size, **kwargs):
    model = inputs[0]
    
    model = Conv2D(
        filters=32,
        kernel_size=[8,8],
        strides=[4,4],
        activation="relu",
        padding="valid",
        name="conv1")(inputs)

    model = Conv2D(
        filters=32, #TODO: test 64
        kernel_size=[4,4],
        strides=[2,2],
        activation="relu",
        padding="valid",
        name="conv2")(model)

    model = Concatenate(3)(model)

    model = Conv2D(
        filters = 32,
        kernel_size =(1,1),
        strides = (1,1),
        activation = "relu",
        name = "merge"
    )(model)

    model = Flatten()(model)
    action = Dense(units=256, activation="relu", name="action_fc_1")(model)
    action = Dense(units=action_space_size, activation=None, name="action_fc_2")(action)

    state = Dense(units=256, activation="relu", name="state_fc_1")(model)
    state = Dense(units=1, name="state_fc_2")(state)

    model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="merge_q")([state, action])
    return Model(inputs = inputs, outputs = [model])