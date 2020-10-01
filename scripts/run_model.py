"""
Runs a trained model on input data.
"""
import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

from skimage.io import imsave, imread

K.clear_session()
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 448
img_cols = 448

smooth = 1e-5


def create_test_data():
    from skimage.io import imread
    data_path = 'data/'
    image_rows = 448
    image_cols = 448

    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)  # // 2

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    # imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=object)

    i = 0
    print('-' * 30)
    print('Loading testing images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        # image_mask_name = image_name.split('.')[0] + '_mask.bmp'
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
        # img_mask = imread(os.path.join(test_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        # img_mask = np.array([img_mask])

        imgs[i] = img
        # imgs_mask[i] = img_mask
        imgs_id[i] = img_id

        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # np.save('imgs_test.npy', imgs)
    # np.save('imgs_mask_test.npy', imgs_mask)
    # np.save('imgs_id_test.npy', imgs_id)
    # print('Saving to .npy files done.')
    return imgs, imgs_id


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten((y_true > 0.5).astype(int))
    y_pred_f = K.flatten((y_pred > 0.5).astype(int))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def similarity(y_true, y_pred):
    y_true_f = K.flatten((y_true > 0.5).astype(int))
    y_pred_f = K.flatten((y_pred > 0.5).astype(int))
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return ((intersection) / (K.sum(K.abs(y_true_f) + K.abs(y_pred_f)) - intersection))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[similarity, dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)

    counter = 0
    for i in range(imgs.shape[0]):
        imgs_p[counter] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
        counter += 1

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def predict():
    print('-' * 30)
    print('Loading and preprocessing input data...')
    print('-' * 30)
    imgs_test, imgs_id_test = create_test_data()

    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model = get_unet()
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test_pred = model.predict(imgs_test, verbose=1)

    # np.save('imgs_mask_test.npy', imgs_mask_test_pred)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'out'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    predict()
