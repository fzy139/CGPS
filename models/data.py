import tensorflow as tf
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

TRAIN_IMG_SIZE = 256


def mean_image(image_tensor):
    channel_r = image_tensor[:,:,0]-R_MEAN
    channel_g = image_tensor[:,:,1]-G_MEAN
    channel_b = image_tensor[:,:,2]-B_MEAN

    return tf.stack([channel_r,channel_g,channel_b],axis=2)


def recover_image(image_tensor,scale=True):
    channel_r = image_tensor[:,:,0]+R_MEAN
    channel_g = image_tensor[:,:,1]+G_MEAN
    channel_b = image_tensor[:,:,2]+B_MEAN

    if scale:
        return tf.stack([channel_r,channel_g,channel_b],axis=2)/255
    else:
        return tf.stack([channel_r, channel_g, channel_b], axis=2)


def get_guiding_image(image, img_filter,filter_paras=()):
    # convert to np
    np_image = image[0].numpy()
    # to 0-255
    np_image = recover_image(np_image, scale=False).numpy()

    if img_filter == 'BilateralFilter':
        (d, sigmaColor,sigmaSpace) = filter_paras
        guiding_image = cv2.bilateralFilter(np_image, d, sigmaColor, sigmaSpace)
    elif img_filter == 'GaussianBlur':
        (ksize, sigma) = filter_paras
        guiding_image = cv2.GaussianBlur(np_image, (ksize, ksize), sigma)
    elif img_filter == 'GuidedFilter':
        (radius, eps) = filter_paras
        guiding_image = cv2.ximgproc.guidedFilter(np_image, np_image, radius=radius, eps=eps)

    guiding_image = mean_image(guiding_image)
    guiding_image_tensor = tf.expand_dims(tf.convert_to_tensor(guiding_image),0)
    return guiding_image_tensor


def normalize(tensor, nonlinear=False):
    tensor = tf.reduce_mean(tensor, axis=[3], keepdims=True)
    if nonlinear:
        tensor = tf.math.pow(tensor/tf.math.reduce_max(tensor),2)
    maxi = tf.math.reduce_max(tensor)
    mini = tf.math.reduce_min(tensor)
    return (tensor-mini)/(maxi-mini)


def get_image_tensor(path,resize=0,crop=0):
    """
        get a image from file 

        Return:
            a tf.tensor of shape [1,H,W,C]

    """
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_image(img_raw,dtype=tf.dtypes.uint8)
    img_tensor = tf.cast(img_tensor,dtype=tf.float32)
    img_tensor = mean_image(img_tensor)

    if resize:
        img_final = tf.image.resize(img_tensor,resize)
    else:
        img_final = img_tensor

    if crop:
        img_final = tf.image.central_crop(img_tensor,0.5)
        # img_final = tf.image.random_crop(img_tensor,[crop,crop,3])

    print("[ ] Loading image "+path+" of shape",img_final.shape)
    return tf.expand_dims(img_final,0)


def check_dataset(image_folder):
    images_name = os.listdir(image_folder)
  
    images_path = []

    for name in images_name:
        name = image_folder+'/'+name
        images_path.append(name) 
    
    for path in images_path:
        try:
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
        except BaseException as e:
           # print(e.msg)
           
            print("Bad Guy",path)


def preprocess_image(image,img_size=TRAIN_IMG_SIZE):
    image = tf.image.decode_jpeg(image, channels=3) #uint8 0-255

    image = tf.cast(image,dtype = tf.float32)
    image = tf.image.resize(image, [img_size, img_size]) # hwc

    image = mean_image(image)

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = preprocess_image(image)
    return image


def build_dataset_from_folder(image_folder,clip=0):
    images_name = os.listdir(image_folder)

    if clip:
        images_name = images_name[:clip]

    count = len(images_name)
    images_path = []

    # adding prefix
    for name in images_name:
        name = image_folder+'/'+name
        images_path.append(name)

    print(images_path[:5],"...total ",count)

    path_ds = tf.data.Dataset.from_tensor_slices(images_path)

    # transform
    ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    return ds,count
