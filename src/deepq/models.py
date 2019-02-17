from keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, Lambda
from keras.models import Model

def atari_backbone(action_space_size):
    block1 = Conv2D(
        filters=32,
        kernel_size=[8,8],
        strides=[4,4],
        activation="relu",
        padding="valid",
        name="conv1")

    block2 = Conv2D(
        filters=32, #TODO: test 64
        kernel_size=[4,4],
        strides=[2,2],
        activation="relu",
        padding="valid",
        name="conv2")

    concatenate3 = Concatenate(3)

    layer3 = Conv2D(
        filters = 32,
        kernel_size =(1,1),
        strides = (1,1),
        activation = "relu",
        name = "merge"
    )

    flatten3 = Flatten()
    layer4 = Dense(
        units=256,
        activation="relu",
        name="fc3")

    adventage = Dense(
        units=action_space_size,
        activation=None,
        name="policy_fc"
    )

    value = Dense(
        units=1,
        name="value_fc"
    )

    final_merge = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="final_out")

    def call(inputs):
        streams = list(map(lambda x: block2(block1(x)), inputs))
        if len(streams) > 1:
            model = concatenate3(streams)
            model = layer3(model)
        else:
            model = streams[0]

        model = flatten3(model)
        model = layer4(model)
        model = final_merge([value(model), adventage(model)])
        return model

    return call