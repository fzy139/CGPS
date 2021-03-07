

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import tensorflow as tf
import matplotlib.pyplot as plt
from models.utils import nearest_patch_swapping,channel_grouping_patching_swapping
from models.data import get_image_tensor,build_dataset_from_folder,recover_image
from models.loss import calc_content_loss,calc_style_loss,calc_ada_loss,calc_tv_loss
from models.metric import compute_gram_loss
from models import  model
gpus = tf.config.experimental.list_logical_devices('GPU')
tf.config.set_soft_device_placement(True)#

#for ada
# weights_loss=[0.3,2,1e-4]
# weights_feature_loss=[0,0,0.1,1]

weights_loss=[0.1,1,1e-5] # feature pixel tv
weights_feature_loss=[1,1,0.1,0.01]


def train_step(content_image):
        with tf.GradientTape() as tape:
                # forward 
                content_feature_map  = encoder(content_image)[-1]
                #print("get feature map ",content_feature_map.shape)

                decoded_img = decoder(content_feature_map)
                #print("decoded ",decoded_img.shape)

                # taping loss
                loss = tf.keras.losses.mean_squared_error(content_image,decoded_img)

              #  print('1!11'+type(loss))
                #print(loss)

        # calc gradients
        gradients = tape.gradient(loss,decoder.trainable_variables)
        
        # optimize models
        optimizer.apply_gradients(zip(gradients,decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def siamese_train_step_with_feature_loss(content_image,style_image):
        with tf.GradientTape() as tape:
                # encode
                content_feature_list = encoder(content_image)
                style_feature_list = encoder(style_image)
                # print("get feature map ",content_feature_map.shape)

                # decode
                # print("decoded ",decoded_img.shape)

                # feature loss
                #with tf.device(gpus[1].name):
                        # decoded_img = decoder(content_feature_list[-1])
                with tf.device(gpus[1].name):
                        [decoded_img, decoded_img_s] = decoder([content_feature_list[-1], style_feature_list[-1]])
                with tf.device(gpus[2].name):
                        reconstructed_feature_list = encoder(decoded_img)
                        reconstructed_feature_list_s = encoder(decoded_img_s)

                #content
                feature_loss = weights_loss[0] * tf.reduce_mean(calc_content_loss(content_feature_list,
                                                                                  reconstructed_feature_list,
                                                                                  weight=weights_feature_loss))
                pixel_loss = weights_loss[1] * tf.keras.losses.mean_squared_error(content_image, decoded_img)

                tv_loss = calc_tv_loss(decoded_img) * weights_loss[2]

                # style
                feature_loss_s = weights_loss[0] * tf.reduce_mean(calc_content_loss(style_feature_list,
                                                                                  reconstructed_feature_list_s,
                                                                                  weight=weights_feature_loss))
                pixel_loss_s = weights_loss[1] * tf.keras.losses.mean_squared_error(style_image, decoded_img_s)

                tv_loss_s = calc_tv_loss(decoded_img_s) * weights_loss[2]

                if iteration % WATCH_POINT == 0:
                        print('feature', feature_loss,feature_loss_s)

                # image loss

                if iteration % WATCH_POINT == 0:
                        print('pixel', tf.reduce_mean(pixel_loss,pixel_loss_s))


                if iteration % WATCH_POINT == 0:
                        print('tv', tf.reduce_mean(tv_loss,tv_loss_s))

                loss = feature_loss + pixel_loss + tv_loss + \
                       feature_loss_s + pixel_loss_s + tv_loss_s

        # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def skip_train_step_with_feature_loss(content_image):
        with tf.GradientTape() as tape:
                # encode
                content_feature_list = encoder(content_image)
                # print("get feature map ",content_feature_map.shape)

                # calc mean sigma
                mean_input = []
                sigma_input = []

                for l in range(2, 5):
                        # print(style_features[l].shape)
                        mean, sigma = tf.nn.moments(content_feature_list[-l], [1, 2], keepdims=True)
                        mean_input.append(mean)
                        sigma_input.append(sigma)

                # decode
                with tf.device(gpus[1].name):
                        decoded_img = decoder([content_feature_list[-1]]+mean_input+sigma_input)[-1]
                        reconstructed_feature_list = encoder(decoded_img)


                feature_loss = weights_loss[0] * tf.reduce_mean(calc_content_loss(content_feature_list,
                                                                                  reconstructed_feature_list,
                                                                                  weights_feature_loss))

                if iteration % WATCH_POINT == 0:
                        print('feature', feature_loss)

                # ada loss

                # image loss
                pixel_loss = weights_loss[1]*tf.keras.losses.mean_squared_error(content_image,decoded_img)
                if iteration % WATCH_POINT == 0:
                        print('pixel',tf.reduce_mean(pixel_loss))

                tv_loss = calc_tv_loss(decoded_img)*weights_loss[2]
                if iteration % WATCH_POINT == 0:
                    print('tv', tf.reduce_mean(tv_loss))

                loss = feature_loss + pixel_loss + tv_loss


        # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def fine_tune_with_gram_loss(content_image,style_image):
        with tf.GradientTape() as tape:

                decoded_image = model.forward(encoder, decoder, content_image, style_image)
                # encode
                content_feature_list = encoder(content_image)
                style_feature_list = encoder(style_image)

                # swap

                # calc mean sigma
                mean_input = []
                sigma_input = []
                for l in range(2, 5):
                        mean, sigma = tf.nn.moments(style_feature_list[-l], [1, 2], keepdims=True)
                        mean_input.append(mean)
                        sigma_input.append(sigma)

                # decode
                with tf.device(gpus[1].name):
                        decoded_img = decoder([content_feature_list[-1]] + mean_input + sigma_input)[-1]
                        # decoded_img =(decoded_img + decoded_img_s) /2 !!!xxxxcan't do that,this cause unequal output
                        reconstructed_feature_list = encoder(decoded_img)

                # content loss
                feature_loss = weights_loss[0] * tf.reduce_mean(calc_content_loss(content_feature_list,
                                                                                  reconstructed_feature_list,
                                                                                  weight=weights_feature_loss))
                if iteration % WATCH_POINT == 0:
                        print('feature', feature_loss)

                # style loss
                style_loss = compute_gram_loss(reconstructed_feature_list[-1], style_feature_list[-1])

                # image loss
                pixel_loss = weights_loss[1] * tf.keras.losses.mean_squared_error(content_image, decoded_img)
                if iteration % WATCH_POINT == 0:
                        print('pixel', tf.reduce_mean(pixel_loss))

                # tv loss
                tv_loss = calc_tv_loss(decoded_img) * weights_loss[2]
                if iteration % WATCH_POINT == 0:
                        print('tv', tf.reduce_mean(tv_loss))

                loss = feature_loss + style_loss + pixel_loss + tv_loss

                # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def train_step_with_feature_loss(content_image):
        with tf.GradientTape() as tape:
                # encode
                content_feature_list = encoder(content_image)
                # print("get feature map ",content_feature_map.shape)

                #decode
                # print("decoded ",decoded_img.shape)

                # feature loss
                with tf.device(gpus[1].name):
                        #decoded_img = decoder(content_feature_list[-1])
                        decoded_img = decoder(content_feature_list[-1])
                        # decoded_img =(decoded_img + decoded_img_s) /2 !!!xxxxcan't do that,this cause unequal output
                        reconstructed_feature_list = encoder(decoded_img)

                feature_loss = weights_loss[0] * tf.reduce_mean(calc_content_loss(content_feature_list,
                                                                                  reconstructed_feature_list,
                                                                                  weight=weights_feature_loss))

                if iteration % WATCH_POINT==0:
                        print('feature',feature_loss)
                # ada loss

                # image loss
                pixel_loss = weights_loss[1]*tf.keras.losses.mean_squared_error(content_image,decoded_img)
                if iteration % WATCH_POINT==0:
                        print('pixel',tf.reduce_mean(pixel_loss))

                tv_loss = calc_tv_loss(decoded_img)*weights_loss[2]
                if iteration % WATCH_POINT==0:
                    print('tv', tf.reduce_mean(tv_loss))

                loss = feature_loss+pixel_loss + tv_loss


        # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def train_step_with_sw(content_image,style_image):
        with tf.GradientTape() as tape:
                # forward
                style_feature_list =encoder(style_image)
                content_feature_list = encoder(content_image)

                #swap
                swapped_feature_map = nearest_patch_swapping(content_feature_list[-1],style_feature_list[-1])
                #decode
                stylized_img = decoder(swapped_feature_map)

                #evaluate loss
                stylized_feature_list = encoder(stylized_img)

                # recon loss
                loss = calc_content_loss(swapped_feature_map,stylized_feature_list[-1])

                #loss = tf.keras.losses.mean_squared_error(content_image,stylized_img)


                # #content
                # content_loss = calc_content_loss(content_feature_list,stylized_feature_list)
                # #print('content',tf.reduce_mean(content_loss))

                # #style
                # style_loss = calc_style_loss(style_feature_list,stylized_feature_list,weight=1)
                # #print('style',tf.reduce_mean(style_loss))
                # loss = content_loss + style_loss


        # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def train_step_with_ada(content_image, style_image):
        with tf.GradientTape() as tape:
                # forward
                style_feature_list = encoder(style_image)
                content_feature_list = encoder(content_image)
                #plt.subplot(131)
               # plt.imshow(recover_image(tf.squeeze(content_image, axis=0)))
                #plt.subplot(132)
                #plt.imshow(recover_image(tf.squeeze(style_image, axis=0)))
                # swap
                with tf.device(gpus[1].name):
                        swapped_feature_map = nearest_patch_swapping(\
                                content_feature_list[-1], style_feature_list[-1])
                # decode
                stylized_img = decoder(swapped_feature_map)

                decoded_img = recover_image(tf.squeeze(stylized_img, axis=0))

                #plt.subplot(133)
                #plt.imshow(decoded_img)
               # plt.show()
                # evaluate loss
                stylized_feature_list = encoder(stylized_img)

                # feature loss
                #feature_loss = tf.reduce_mean( \
                #          calc_content_loss(content_feature_list,stylized_feature_list,weight=[0.1, 0.1, 0.1,0.01]))
                recon_loss = 0.01*tf.keras.losses.MSE(stylized_feature_list[-1],swapped_feature_map)
                feature_loss = 0
                #feature_loss *= 0.1

                # ada loss
                style_feature_list.append(style_image)
                stylized_feature_list.append(stylized_img)
                ada_loss = tf.reduce_mean(calc_ada_loss(style_feature_list,stylized_feature_list,\
                                                        weight=[1,0.1,0.5,0.1,100]))

                ada_loss *=0.1
                #ada_loss = 0
                # image loss
                #pixel_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(content_image, stylized_img))
                pixel_loss = 0

                tv_loss = calc_tv_loss(decoded_img) * 1e-3

                if iteration % WATCH_POINT == 0:
                        print('ada', ada_loss)

                if iteration % WATCH_POINT == 0:
                        print('recon', tf.reduce_mean(recon_loss))
                # if TEST and iteration % WATCH_POINT == 0:
                #         print('feature', feature_loss)
                # if TEST and iteration % WATCH_POINT==0:
                #         print('pixel', tf.reduce_mean(pixel_loss))
                if iteration % WATCH_POINT == 0:
                        print('tv', tf.reduce_mean(tv_loss))
                loss =  ada_loss + tv_loss + recon_loss
                #loss =tv_loss
        # calc gradients
        gradients = tape.gradient(loss, decoder.trainable_variables)

        # optimize models
        optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))

        # mean all iter
        train_loss(loss)


def test_sw(out_path,show=False,cpath="./data/content/lenna.jpg",spath='./data/style/candy.jpg'):
        #path = "./data/contents/bike/bike.jpg"
        content_image = get_image_tensor(cpath)
        style_image = get_image_tensor(spath)

        content_feature_map  = encoder(content_image)[-1]
        style_feature_list  = encoder(style_image)


        style_feature_map = style_feature_list[-1]

        mean_input = []
        sigma_input = []

        swapped_feature_map = nearest_patch_swapping(content_feature_map,
                                                     style_feature_map,stride=1)

        for l in range(2, 5):
                # print(style_features[l].shape)
                mean, sigma = tf.nn.moments(style_feature_list[-l], [1, 2], keepdims=True)
                mean_input.append(mean)
                sigma_input.append(sigma)

        outputs = decoder([swapped_feature_map, mean_input, sigma_input])
        decoded_img = outputs[-1]

       # # print("get feature map ",content_feature_map.shape)
       #  swapped_feature_map = nearest_patch_swapping(content_feature_map,style_feature_map,stride=1)
       #  #decoded_img = decoder(swapped_feature_map)
       #  [decoded_img,decoded_img_s] = decoder([swapped_feature_map,
       #                                       style_feature_map])
       #
       #  # print("decoded ",decoded_img.shape)
       #
        decoded_img = recover_image(tf.squeeze(decoded_img,axis=0))
        plt.imshow(decoded_img)
        plt.axis('off')

        if show:
                plt.show()
        else:
                plt.savefig(out_path)


def test(out_path,show=False,path="./data/content/bike.jpg"):
        #path = "./data/contents/bike/bike.jpg"
        content_image = get_image_tensor(path)

        content_feature_map  = encoder(content_image)[-1]
       # print("get feature map ",content_feature_map.shape)
        decoded_img = decoder(content_feature_map)
       # print("decoded ",decoded_img.shape)
        #decoded_img = (decoded_img +M) / 255
        decoded_img = recover_image(tf.squeeze(decoded_img,axis=0))

        plt.imshow(decoded_img)
        plt.axis('off')

        if show:
                plt.show()
        plt.savefig(out_path)


EPOCH = 10
BATCH_SIZE = 16
LR=0.0003
TEST = False
LOAD = False

c_path = "./data/content/bridge.jpg"
s_path = "./data/style/starry_night.jpg"
image_folder = "/home/zhuyan/train2017"
style_folder = "/home/zhuyan/painter"
checkpoints_dir = './checkpoints/rag'
output_dir="./output/rag"

if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

#####
# assemble model
#####

encoder = model.generate_vgg19_encoder(model.layers[:4])
#print(encoder.summary())

# decoder = model.generate_vgg19_decoder(model.vgg_19_decoder_architecture,\
#                                         starting_layer='conv4/conv4_1')

decoder = model.generate_eva_vgg19_decoder(model.vgg_19_decoder_architecture,\
                                        starting_layer='conv4/conv4_1')
#print(decoder.summary())

##### Loading Weights #####
if LOAD:

        #checkpoint_path = './checkpoints/41_dm/14'
        #checkpoint_path = './checkpoints/41_wfl_tv1/8'
        checkpoint_path = './checkpoints/41_tv/0_10'
        #checkpoint_path = './checkpoints/41_ada/050000'
        decoder.load_weights(checkpoint_path)
        print('[!!]loaded weight from '+checkpoint_path)

#####
# 
#####
train_loss = tf.metrics.Mean(name="train")
optimizer = tf.optimizers.Adam(learning_rate=LR)
print("[!!!]weight loss",weights_loss," feature loss",weights_feature_loss)
#####
# Prepare Dataset
#####
if TEST:
        show = True
        WATCH_POINT = 10
        SHOW_POINT = 10
        training_set,count = build_dataset_from_folder(image_folder,clip=1000)
        art_set,count_art = build_dataset_from_folder(style_folder,clip=500)
else:
        show = False
        WATCH_POINT = 50
        SHOW_POINT = 500
        training_set,count = build_dataset_from_folder(image_folder)#,clip=True)
        art_set,count_art = build_dataset_from_folder(style_folder)#,clip=True)


training_set = training_set.repeat(1)\
        .shuffle(buffer_size = 5000)\
        .batch(BATCH_SIZE)\
        .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

art_set = art_set.repeat()\
         .shuffle(buffer_size = 5000)\
         .batch(BATCH_SIZE)


print('[!!!]Test=',TEST,'Load=',LOAD)


content =['bike.jpg','archi.jpg','bridge.jpg','lenna.jpg','chicago.png','brad_pitt.jpg']
content_path = './data/content/'+content[3]

style = ['candy','starry_night','monai2','flower','freud']
style_path = './data/style/'+style[0]+'.jpg'


epoch=0
for epoch in range(EPOCH):
        print("--- epoch %d ---"%epoch)
        iteration = 0
        #it = iter(art_set) # create a instance
        # iter
        for batch in training_set: # no need to iter() a batch already

                #art_batch = next(it) # must call on instance


                #train_step_with_sw(batch,art_batch)

                #siamese_train_step_with_feature_loss(batch,batch)

                skip_train_step_with_feature_loss(batch)

                #train_step_with_ada(batch,art_batch)

                # test every 100 iters
                if iteration % WATCH_POINT==0 and iteration > 0:
                        print("Ep %d Iter %d Loss %f" %(epoch,iteration,train_loss.result()))
                if iteration % SHOW_POINT ==0 and iteration >0:
                        test_sw(out_path=output_dir+"ep"+str(epoch)+"_it"+str(iteration)+".png",show=show)
                #         decoder.save_weights('./checkpoints/'+str(epoch))
                iteration += 1


                #saving model
                if not TEST and iteration % SHOW_POINT==0:
                        checkpoints_filename = checkpoints_dir+str(epoch)+'_'+str(iteration)[0:2]
                        decoder.save_weights(checkpoints_filename, save_format='h5')  # for partial load
                        print("[!!!]model saved "+checkpoints_filename)

        #test
        test_sw(out_path=output_dir+str(epoch)+".png")

        #evaluate_image(c_path, s_path, func='sw', show=True, save=False, title=None)
        #reset loss for next epoch
        #training_set.repeat(1)
        train_loss.reset_states()


#TODO 1. Adding Swap
#TODO 2. Style Loss
