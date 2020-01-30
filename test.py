import os
import tensorflow as tf
from models.utils import nearest_patch_swapping, display
from models.data import get_image_tensor, normalize, get_guiding_image
from models import model

#
# handle arguments
#

checkpoint_path = './checkpoints/4_50'
output_dir = './results'
content_path = './data/content'
style_path = './data/style'
img_filter = 'GaussianBlur'
LAYER = 4
filter_paras = (13, 20)
SHOW = False
SAVE = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 
# load models
#

encoder = model.generate_vgg19_encoder(model.layers[:LAYER])
decoder = model.generate_eva_vgg19_decoder(model.vgg_19_decoder_architecture, starting_layer='conv4/conv4_1')
decoder.load_weights(checkpoint_path, by_name=True)

def evaluate(content, style,
             preserve_content=False,
             mask='hard',
             ratio={'alpha': 0.5, 'beta': 1, 'gamma': 1}):

    # get images
    c = get_image_tensor(content)
    s = get_image_tensor(style)

    # get guiding image
    lc = get_guiding_image(c, img_filter, filter_paras)

    # get feature map and its low pass as surface v
    content_feature = encoder(c)[-1]
    style_feature_list = encoder(s)
    style_feature = style_feature_list[-1]
    content_surface = encoder(lc)[-1]

    # estimate channel mask
    if mask is 'hard':
        surface_mask = tf.cast(tf.reduce_mean(content_surface, [1, 2]) >
                    ratio['alpha'] * tf.reduce_mean(content_feature, [1, 2]), 'float32')
        texture_code = 1 - surface_mask
    else:  # soft
        surface_mask = tf.reduce_mean(content_surface, [1, 2]) \
                                 / tf.reduce_mean(content_feature, [1, 2])
        texture_code = 1 - surface_mask

    # group channel
    content_surface = content_feature * surface_mask
    content_texture = content_feature * texture_code
    style_surface = style_feature * surface_mask
    style_texture = style_feature * texture_code

    # swap surface
    swapped_surface = nearest_patch_swapping(content_surface,
                                             style_surface,
                                             patch_size=3,
                                             metric='ucc')

    # swap textures
    swapped_textures = nearest_patch_swapping(content_texture,
                                              style_texture,
                                              patch_size=3,
                                              metric='ucc')

    # combine
    recon_feature = ratio['beta'] * swapped_surface + ratio['gamma'] * swapped_textures

    # content preserve
    if preserve_content:
        weight = normalize(content_texture, nonlinear=True)
        recon_feature = weight * content_feature + (1 - weight) * recon_feature

    # decode
    output = decode_with_skip_connections(recon_feature, style_feature_list)[0]
    title = os.path.split(content)[1][:-4]+'_'+os.path.split(style)[1][:-4]
    display(output, title, show=SHOW, save=SAVE, out_dir=output_dir)


def decode_with_skip_connections(swapped_feature, style_feature_list):
    mean_input = []
    sigma_input = []

    for l in range(2, 5):
        mean, sigma = tf.nn.moments(style_feature_list[-l], [1, 2], keepdims=True)
        mean_input.append(mean)
        sigma_input.append(sigma)

    outputs = decoder([swapped_feature,mean_input,sigma_input])
    output = outputs[-1]
    return output


def evaluate_images(c_dir, s_dir):
    content_names = os.listdir(c_dir)
    style_names = os.listdir(s_dir)

    idx_c = 0
    for content_name in content_names:
        c_path = c_dir + '/' + content_name

        idx_s = 0
        for style_name in style_names:
            s_path = s_dir + '/' + style_name
            evaluate(c_path, s_path)

            idx_s = idx_s + 1
        idx_c = idx_c + 1


if __name__ == '__main__':
    evaluate_images(content_path, style_path)
