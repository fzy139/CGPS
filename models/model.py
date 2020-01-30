import tensorflow as tf
from models.utils import Conv2DResize,Conv2DSame


def generate_vgg19_encoder(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # create feaure maps
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # define the model
    model = tf.keras.Model([vgg.input], outputs)
    return model


def generate_vgg19_decoder(network_struct, starting_layer='conv4/conv4_1'):
    model = tf.keras.models.Sequential()

    started_decoding = False
    # load layer parameters
    for layer_name, layer_struct in network_struct:
        # check if the starting layer
        if layer_name == starting_layer:
            conv_type, num_outputs, kernel_size = layer_struct
            started_decoding = True  # flag
            model.add(Conv2DSame(num_outputs, kernel_size, stride=1))
            model.add(tf.keras.layers.ReLU())

        # if started,create this layer and add to model
        elif started_decoding:
            conv_type, num_outputs, kernel_size = layer_struct
            if conv_type == 'c':
                model.add(Conv2DSame(num_outputs, kernel_size, stride=1))
                model.add(tf.keras.layers.ReLU())

            elif conv_type == 'uc':
                model.add(Conv2DResize(num_outputs, kernel_size, stride=2))
            # model.add(tf.keras.layers.ReLU())  # new adding

        else:  # not start yet
            pass
    # final output
    model.add(Conv2DSame(num_outputs=3, kernel_size=7, stride=1))

    return model

def generate_eva_vgg19_decoder(network_struct,starting_layer='conv4/conv4_1'):
    swapped_feature = tf.keras.layers.Input(shape=(None, None, 512), name='swapped_feature',ragged=False)
    mean_input_2 = tf.keras.layers.Input(shape=(None,None,256))
    sigma_input_2 = tf.keras.layers.Input(shape=(None,None,256))

    mean_input_6 = tf.keras.layers.Input(shape=(None,None,128))
    sigma_input_6 = tf.keras.layers.Input(shape=(None,None,128))

    mean_input_8 = tf.keras.layers.Input(shape=(None,None,64))
    sigma_input_8 = tf.keras.layers.Input(shape=(None,None,64))

    mean_input = [mean_input_2,mean_input_6,mean_input_8]
    sigma_input = [sigma_input_2,sigma_input_6,sigma_input_8]  # ugly code

    block = 0
    decoded_feature = []

    started_decoding = False
    for layer_name, layer_struct in network_struct:
        # check if the starting layer
        if layer_name == starting_layer:
            conv_type, num_outputs, kernel_size = layer_struct
            started_decoding = True  # flag
            x = Conv2DSame(num_outputs, kernel_size, stride=1)(swapped_feature)
            x = tf.keras.layers.ReLU()(x)

        # if started,create this layer and add to model
        elif started_decoding:
            conv_type, num_outputs, kernel_size = layer_struct
            if conv_type == 'c':
                x = Conv2DSame(num_outputs, kernel_size, stride=1)(x)
                x = tf.keras.layers.ReLU()(x)

            elif conv_type == 'uc':
                x = tf.keras.layers.LayerNormalization(axis=[1, 2], center=False, scale=False)(x)
                x = x * tf.sqrt(sigma_input[block]) + mean_input[block]
                decoded_feature.append(x)
                x = Conv2DResize(num_outputs, kernel_size, stride=2)(x)
                x = tf.keras.layers.ReLU()(x)  # new adding

                block = block + 1

        else:  # not start yet
            pass
    # final output
    output = Conv2DSame(num_outputs=3, kernel_size=7, stride=1)(x)

    model = tf.keras.models.Model(inputs=[swapped_feature]+mean_input+sigma_input,outputs=decoded_feature+[output])

    return model


layers = ['block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1']

vgg_19_decoder_architecture = [
    ('conv5/conv5_4', ('c', 512, 3)),
    ('conv5/conv5_3', ('c', 512, 3)),
    ('conv5/conv5_2', ('c', 512, 3)),
    ('conv5/conv5_1', ('c', 512, 3)),
    ('conv4/conv4_4', ('uc', 512, 3)),
    ('conv4/conv4_3', ('c', 512, 3)),
    ('conv4/conv4_2', ('c', 512, 3)), ### <-----
    ('conv4/conv4_1', ('c', 256, 3)),
    ('conv3/conv3_4', ('uc', 256, 3)),
    ('conv3/conv3_3', ('c', 256, 3)),
    ('conv3/conv3_2', ('c', 256, 3)),
    ('conv3/conv3_1', ('c', 128, 3)),
    ('conv2/conv2_2', ('uc', 128, 3)),
    ('conv2/conv2_1', ('c', 64, 3)),
    ('conv1/conv1_2', ('uc', 64, 3)),
    ('conv1/conv1_1', ('c', 64, 3)),
]       





