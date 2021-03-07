import tensorflow as tf

def calc_tv_loss(output_tensor):
    return tf.reduce_sum(tf.image.total_variation(output_tensor))

def calc_content_loss(content_feature_list,stylized_feature_list,weight=[1,1,1,1]):
    num_layers = len(content_feature_list)
    total_loss = 0
    assert len(weight) == num_layers

    for i in range(num_layers):
        loss = weight[i]*tf.reduce_mean(tf.keras.losses.MSE(content_feature_list[i], stylized_feature_list[i]))
        total_loss+=loss
        #print('layer %d: %f'%(i,loss.numpy()))
    return total_loss

# wrong function
def calc_style_loss(style_feature_list,stylized_feature_list,weight=1):

    num_layers = len(style_feature_list)
    loss = 0

    #for i in range(1,2):
    i=3
    loss += tf.math.reduce_mean(\
            tf.math.divide(weight,i+1)*\
            tf.keras.losses.MSE(style_feature_list[i],stylized_feature_list[i]))

        # tf.math.pow(weight,i)*\
        # print("style loss"+str(i))
        # print(loss)

    return loss


def compute_gram_matrix(feature):
    """compute the gram matrix for a layer of feature
    the gram matrix is normalized with respect to the samples and
        the dimensions of the input features

    """
    shape = tf.shape(feature)
    feature_size = tf.reduce_prod(shape[1:])
    vectorized_feature = tf.reshape(
        feature, [shape[0], -1, shape[3]])
    gram_matrix = tf.matmul(
        vectorized_feature, vectorized_feature, transpose_a=True)  # output shape: [c,c]
    gram_matrix /= tf.to_float(feature_size)
    return gram_matrix


def calc_ada_loss(style_feature_list,stylized_feature_list,weight=[1,1,0,0,1]):
    num_layers = len(style_feature_list)
    loss = 0

    for i in range(num_layers):
        s_mean = tf.reduce_mean(style_feature_list[i],axis=[1,2])#tf.nn.moments(style_feature_list[i],axes=[1,2]) # across spatial dimension
        s_variance = tf.math.reduce_variance(style_feature_list[i],axis=[1,2])

        cs_mean = tf.reduce_mean(stylized_feature_list[i],axis=[1,2])

        cs_variance = tf.math.reduce_variance(stylized_feature_list[i],axis=[1,2])
        #print(style_feature_list[i].shape,cs_mean.shape,cs_variance.shape)
        #print(s_mean.shape)
        mean_loss = tf.reduce_mean(tf.keras.losses.MSE(s_mean,cs_mean))
        var_loss = 0.01*tf.reduce_mean(tf.keras.losses.MSE(s_variance, cs_variance))
        loss += weight[i]*mean_loss
        loss += weight[i]*var_loss
        #print('mean l',i,mean_loss)
        #print('var l',i,var_loss)
        #print('ada loss',i,loss)

    return loss/1e4