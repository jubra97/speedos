import tensorflow as tf
import numpy as np
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc


# This is hardcoded, idk how to make it variable though
nb_scalar_features = 16


def nature_cnn_mlp_mix(scaled_images, **kwargs):
    activ = tf.nn.relu

    # Take last channel as direct features
    other_features = tf.contrib.slim.flatten(scaled_images[..., -1])
    # Take known amount of direct features, rest are padding zeros
    other_features = other_features[:, :nb_scalar_features]

    # cnn
    scaled_images = scaled_images[..., :-1]
    cnn_1 = activ(conv(scaled_images, 'cnn1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    cnn_2 = activ(conv(cnn_1, 'cnn2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    cnn_3 = activ(conv(cnn_2, 'cnn3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    cnn_3 = conv_to_fc(cnn_3)
    img_output = activ(linear(cnn_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))

    # mlp
    mlp_1 = activ(linear(other_features, 'mlp1', n_hidden=32))
    # mlp_2 = activ(linear(mlp_1, 'mlp2', n_hidden=32))
    # return tf.concat((img_output, mlp_2), axis=1)

    # bring cnn together and have some combined layers
    concat = tf.concat((img_output, mlp_1), axis=1)
    combined_1 = activ(linear(concat, 'combined1', n_hidden=64))
    combined_2 = activ(linear(concat, 'combined2', n_hidden=32))

    return combined_2
