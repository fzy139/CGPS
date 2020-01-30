import tensorflow as tf
from tensorflow.keras import layers 
import matplotlib.pyplot as plt
from models.data import recover_image


def nearest_patch_swapping(content_features, style_features,patch_size=3,stride=1,metric='cc'):
    # channels for both the content and style, must be the same
    c_shape = tf.shape(content_features)
    s_shape = tf.shape(style_features)
    channel_assertion = tf.Assert(
        tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])


    # Different metric
    if metric is 'cc' or metric is 'ucc':
        content_mean = tf.math.reduce_mean(content_features,axis=[1,2],keepdims=True)
        style_mean = tf.math.reduce_mean(style_features,axis=[1,2],keepdims=True)

        centred_content_features = (content_features - content_mean )#/ content_std
        centred_style_features = (style_features - style_mean)

    elif metric is 'cd':  # cosine distance
        centred_content_features = content_features
        centred_style_features = style_features

    with tf.control_dependencies([channel_assertion]):
        # spatial shapes for style and content features
        c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]

        ########################
        # Conv kernels #
        ########################

        # convert the style features into convolutional kernels
        style_kernels = tf.image.extract_patches(
            centred_style_features, sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')
        # same,incredible if valid

        style_kernels = tf.squeeze(style_kernels, axis=0)
        style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])
        style_kernels = tf.cast(style_kernels,dtype=tf.float32)
        kernels_norm = tf.norm(style_kernels, axis=0, keepdims=True)
        kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, -1))

        # # gather the conv and deconv kernels
        style_kernels = tf.reshape(
            style_kernels, shape=(patch_size, patch_size, c_channel, -1))

        if metric is 'cc' or metric is 'cd':  # normalized correlation
#            print(metric is 'cd' or 'cc')
            style_kernels = tf.divide(style_kernels, kernels_norm + 1e-7)

        ########################
        # Deconv kernels #
        ########################
        # convert the style features into convolutional kernels
        deconv_kernels = tf.image.extract_patches(
            style_features, sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME')  # same

        deconv_kernels = tf.squeeze(deconv_kernels, axis=0)
        deconv_kernels = tf.transpose(deconv_kernels, perm=[2, 0, 1])
        deconv_kernels = tf.cast(deconv_kernels, dtype=tf.float32)
        # gather the conv and deconv kernels
        v_height, v_width = deconv_kernels.get_shape().as_list()[1:3]
        deconv_kernels = tf.reshape(
            deconv_kernels, shape=(patch_size, patch_size, c_channel, -1))

        # kernels_norm = tf.norm(style_kernels, axis=0, keepdims=True)
        # kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, -1))

        # calculate the normalization factor
        #
        ########################
        # average overlapping #
        ########################
        mask = tf.ones((c_height, c_width), tf.float32)
        fullmask = tf.zeros((c_height+patch_size-1, c_width+patch_size-1), tf.float32)
        for x in range(patch_size):
            for y in range(patch_size):
                paddings = [[x, patch_size-x-1], [y, patch_size-y-1]]
                padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
                fullmask += padded_mask
        pad_width = int((patch_size-1)/2)
        deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
        deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))

        ########################
        # starting convolution #
        ########################
        # padding operation
        pad_total = patch_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

        # convolutional operations
        net = tf.pad(centred_content_features, paddings=paddings, mode="REFLECT")
        net = tf.nn.conv2d(
            net,
            style_kernels,
            strides=[1, 1, 1, 1],
            padding='VALID')

        ########################
        # Argmax  #
        ########################
        # # find the maximum locationsbest_match_ids
        best_match_ids = tf.argmax(net, axis=3)
        best_match_ids = tf.cast(
            tf.one_hot(best_match_ids, depth=v_height*v_width), dtype=tf.float32)

        #print('ids',best_match_ids.shape)

        ########################
        # Deconv  #
        ########################
        # find the patches and warping the output
        unnormalized_output = tf.nn.conv2d_transpose(
            input=best_match_ids,
            filters=deconv_kernels,
            output_shape=(c_shape[0], c_height+pad_total, c_width+pad_total, c_channel),
            strides=[1, 1, 1, 1],
            padding='VALID')

        unnormalized_output = tf.slice(unnormalized_output, [0, pad_beg, pad_beg, 0], c_shape)
        output = tf.math.divide(unnormalized_output, deconv_norm)
        output = tf.reshape(output, shape=c_shape)

        # output the swapped feature maps
        return output


def depthwise_nearest_patch_swapping(content_features, style_features,depth=512,patch_size=3,stride=1,metric='cc'):
    # channels for both the content and style, must be the same
    c_shape = tf.shape(content_features)
    s_shape = tf.shape(style_features)
    channel_assertion = tf.Assert(
        tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])

    num_channel = c_shape[3]
    with tf.control_dependencies([channel_assertion]):

        for c in range(0,num_channel,depth):
            if depth==1:
                swapped_channel = nearest_patch_swapping(tf.expand_dims(content_features[:, :, :, c], 3),
                                                         tf.expand_dims(style_features[:, :, :, c], 3),
                                                         patch_size=patch_size, stride=stride, metric=metric)
            else:
                swapped_channel = nearest_patch_swapping(content_features[:,:,:,c:c+depth],
                                                         style_features[:,:,:,c:c+depth],
                                           patch_size=patch_size,stride=stride,metric=metric)
            if c == 0:
                output = swapped_channel

            else:
                output = tf.concat([output,swapped_channel],axis=3)
                # output the swapped feature maps
            #print(swapped_channel.shape)
            #print(output.shape)

        return output


class Conv2DSame(layers.Layer):
    # all input-independent init
    def __init__(self, num_outputs, kernel_size, stride, rate=1):
        super(Conv2DSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.num_outputs = num_outputs

    # based on the shape

    def build(self, inputs_shape):
        self.filter = self.add_weight("filter", shape=[self.kernel_size, self.kernel_size,
                                                       inputs_shape[-1], self.num_outputs])

    # forward calcation

    def call(self, inputs):

        if self.kernel_size == 1:
            print("calling same ks=1 ")
            return tf.nn.relu(
                tf.nn.conv2d(inputs, filters=self.filter, strides=self.stride,
                             dilations=self.rate, padding='SAME'))

        else:
            kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

            # padding
            inputs = tf.pad(inputs, paddings=paddings, mode="REFLECT")
            # conv
            activation = tf.nn.conv2d(inputs, filters=self.filter, strides=self.stride,
                                      dilations=self.rate, padding='VALID')
            # relu
            # outputs = tf.nn.relu(activation)

            return activation
            # return outputs


class Conv2DResize(layers.Layer):

    def __init__(self, num_outputs, kernel_size, stride, rate=1):
        super(Conv2DResize, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.num_outputs = num_outputs

    def build(self, inputs_shape):

        # print("resize building")
        # print(inputs_shape)
        # tf.keras.initializers`
        self.filter = self.add_weight("filter", \
                                      shape=[self.kernel_size, self.kernel_size, int(inputs_shape[-1]),
                                             self.num_outputs])

    def call(self, inputs):
        # print("calling resize")
        if self.stride == 1:  # no use
            same_layer = Conv2DSame(self.num_outputs, self.kernel_size, self.stride, self.rate)
            outputs = same_layer(inputs)
            return outputs

        else:
            # resize up
            stride_larger_than_one = tf.greater(self.stride, 1)
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            new_height, new_width = tf.cond(
                stride_larger_than_one,
                lambda: (height * self.stride, width * self.stride),
                lambda: (height, width))
            inputs_resize = tf.image.resize(inputs, [new_height, new_width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # reflect padding
            kernel_size_effective = self.kernel_size + (self.kernel_size - 1) * (self.rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            inputs_resize = tf.pad(inputs_resize, paddings=paddings, mode="REFLECT")

            # conv
            activation = tf.nn.conv2d(inputs_resize, filters=self.filter, strides=1,
                                      dilations=self.rate, padding='VALID')
            outputs = tf.nn.relu(activation)

            return outputs


def display(output_image, title='', show=True, save=False, out_dir='./'):
    output_image = recover_image(output_image)

    if show:
        plt.axis('off')
        # plt.title(title)
        plt.imshow(output_image)  # www
        plt.show()
        # Image.fromarray(tf.cast(output_image,'uint8').numpy()).show()
    if save:
        plt.axis('off')
        plt.imshow(output_image)  # www
        plt.savefig(out_dir + '/' + title + '.png')
        plt.savefig(out_dir + '/' + title + '.png', bbox_inches='tight')  # ,format='eps')
        # output_image = np.clip(output_image.numpy()*255, 0, 255).astype(np.uint8)
        # Image.fromarray(output_image)\
        #      .save(out_dir+'/'+title + '.jpg', quality=95)