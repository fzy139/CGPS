import os
import tensorflow as tf
from models.utils import nearest_patch_swapping, display, channel_grouping_patching_swapping
from models.data import get_image_tensor
from models import model,metric,loss
import numpy as np
#
# handle arguments
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

checkpoint_path = './checkpoints/4_50'
output_dir = '/home/zhuyan/aams_results_small'
content_path = '/home/zhuyan/test_set/1'
style_path = '/home/zhuyan/test_set/2'

LAYER = 4


SHOW = True
SAVE = False
PURE = False


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 
# load models
#

encoder = model.generate_vgg19_encoder(model.layers[:LAYER])
decoder = model.generate_eva_vgg19_decoder(model.vgg_19_decoder_architecture, starting_layer='conv4/conv4_1')
decoder.load_weights(checkpoint_path, by_name=True)


def evaluate(output_image, style_image, metric_name='gram'):
    """
    evaluate an output image with style metric
    :param metric_name:
    :param output_image:   output image tensor of shape (h,w,3)
    :param style_image::  style image tensor of shape (h,w,3)
    :return: a scale
    """

    output_feature = encoder(output_image)#[-1]
    style_feature = encoder(style_image)#[-1]

    style_metric = 0
    if metric_name == 'gram':
        style_metric = metric.compute_gram_loss(output_feature[-1], style_feature[-1])
    elif metric_name == 'ada':
        style_metric = loss.calc_ada_loss(output_feature,style_feature)

    return style_metric


def forward_and_evaluate(content, style,style_metric='gram',
                         preserve_content=False,
                         mask='hard',
                         ratio={'alpha': 0.5, 'beta': 1, 'gamma': 1}):
    """

    :param content:
    :param style:
    :param style_metric:
    :param preserve_content:
    :param mask:
    :param ratio:
    :return:
    """

    output = model.forward(encoder, decoder, content, style, preserve_content, mask, ratio)
    style_metric = evaluate(output, style, style_metric)
    return style_metric, output[0]


def test_images(c_dir, s_dir, option='fe'):
    """
    Load images from folders, and apply style transfer
    :param c_dir:content image directory
    :param s_dir:style image directory
    :return:a list contains style metric in float
    """
    if os.path.isfile(c_dir):
        c_dir, c_name = os.path.split(c_dir)
        content_names = [c_name]
    else:
        content_names = os.listdir(c_dir)

    if os.path.isfile(s_dir):
        s_dir, s_name = os.path.split(s_dir)
        style_names = [s_name]
    else:
        style_names = os.listdir(s_dir)

    style_metrics = []
    idx_s = 0
    for style_fname in style_names:
        s_path = s_dir + '/' + style_fname
        style_name = os.path.splitext(style_fname)[0]
        output_path = os.path.join(output_dir, style_name)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        idx_c = 0
        folder_name = ''
        if option == 'e':
            folder_name = '/' + style_name
            content_names = os.listdir(os.path.join(c_dir, style_name))
            print(content_names)

        for content_name in content_names:
            c_path = c_dir + folder_name + '/' + content_name
            # get images
            c = get_image_tensor(c_path)
            s = get_image_tensor(s_path)

            if option == 'fe':
                style_metric, output = forward_and_evaluate(c, s)
            elif option == 'f':
                output = model.forward(encoder, decoder, c, s)
            elif option == 'e':
                style_metric = evaluate(c, s)

            if option != 'e':
                title = os.path.split(c_path)[1][:-4] + '_' +\
                        os.path.split(s_path)[1][:-4] + ' ' + str(style_metric.numpy()[0])
                display(output, title, pure=PURE, show=SHOW, save=SAVE, out_dir=output_path)

            style_metrics.append(style_metric.numpy()[0])
            print(style_metric.numpy())
            idx_s = idx_s + 1
        idx_c = idx_c + 1

    return style_metrics


if __name__ == '__main__':
    metric_list = test_images(output_dir, style_path, 'e')

    np.save('./aams_small.npy', np.array(metric_list))
    print(metric_list)
    print(sum(metric_list) / len(metric_list))
