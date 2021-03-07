import tensorflow as tf


def compute_gram_matrix(feature):
    """compute the gram matrix for a layer of feature

    the gram matrix is normalized with respect to the samples and
        the dimensions of the input features

    """
    shape = tf.shape(feature)
    feature_size = tf.math.reduce_prod(shape[1:])  # h*w*c
    vectorized_feature = tf.reshape(
        feature, [shape[0], -1, shape[3]])  # [h * w,c]
    gram_matrix = tf.linalg.matmul(
        vectorized_feature, vectorized_feature, transpose_a=True)  # [c,c] ignore spatial axis
    gram_matrix /= tf.cast(feature_size, tf.float32)  # normalize

    return gram_matrix


def compute_gram_loss(output, style):
    gram_o = compute_gram_matrix(output)
    gram_s = compute_gram_matrix(style)
    return tf.math.reduce_mean(tf.math.square(gram_o - gram_s), axis=[1, 2])

