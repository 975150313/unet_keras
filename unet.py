from keras.models import Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import BatchNormalization, GaussianNoise
from keras.layers import concatenate
from keras.preprocessing import sequence
from keras import backend as K

def conv2d(layer_input, filters, axis=-1):
    d = Conv2D(filters, (3, 3), padding='same')(layer_input)
    d = BatchNormalization(axis=axis)(d)
    d = Activation('relu')(d)
    d = Conv2D(filters, (3, 3), padding='same')(d)
    d = BatchNormalization(axis=axis)(d)
    d = Activation('relu')(d)
    return d

def deconv2d(layer_input, skip_input, filters, axis=-1):
    u = concatenate([UpSampling2D(size=(2, 2))(layer_input), skip_input], axis=axis)
    u = Conv2D(filters, (3, 3), padding='same')(u)
    u = BatchNormalization(axis = axis)(u)
    u = Activation('relu')(u)
    u = Conv2D(filters, (3, 3), padding='same')(u)
    u = BatchNormalization(axis = axis)(u)
    u = Activation('relu')(u)
    return u

def load_2d_unet(input_shape=(512,512,1), num_labels=1, noise=0.1, init_filter=32):
    d0 = Input(shape=input_shape)
    d1 = GaussianNoise(noise)(d0)
    d1 = conv2d(d1, init_filter)
    d1_p = MaxPooling2D(pool_size=(2, 2))(d1)
    d2 = conv2d(d1_p, init_filter*2)
    d2_p = MaxPooling2D(pool_size=(2, 2))(d2)
    d3 = conv2d(d2_p, init_filter*4)
    d3_p = MaxPooling2D(pool_size=(2, 2))(d3)
    d4 = conv2d(d3_p, init_filter*8)
    d4_p = MaxPooling2D(pool_size=(2, 2))(d4)
    d5 = conv2d(d4_p, init_filter*16)
    d5_p = MaxPooling2D(pool_size=(2, 2))(d5)
    d6 = conv2d(d5_p, init_filter*32)

    u1 = deconv2d(d6, d5, init_filter*16)
    u2 = deconv2d(u1, d4, init_filter*8)
    u3 = deconv2d(u2, d3, init_filter*4)
    u4 = deconv2d(u3, d2, init_filter*2)
    u5 = deconv2d(u4, d1, init_filter)

    output_img = Conv2D(num_labels, (1, 1), activation='sigmoid')(u5)
    model = Model(inputs=d0, outputs=output_img)
    return model

def load_3d_unet(input_shape=(128,160,160,1), num_labels=3, noise=0.1, init_filter=24):
    inputs = Input(shape = input_shape)
    conv1 = GaussianNoise(noise)(inputs)
    conv1 = ZeroPadding3D((1, 1, 1))(conv1)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis = -1)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding3D((1, 1, 1))(conv1)
    conv1 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv1)
    conv1 = BatchNormalization(axis = -1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = ZeroPadding3D((1, 1, 1))(pool1)
    conv2 = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis = -1)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = ZeroPadding3D((1, 1, 1))(conv2)
    conv2 = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv2)
    conv2 = BatchNormalization(axis = -1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = ZeroPadding3D((1, 1, 1))(pool2)
    conv3 = Conv3D(init_filter*4, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis = -1)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = ZeroPadding3D((1, 1, 1))(conv3)
    conv3 = Conv3D(init_filter*4, (3, 3, 3), strides=(1, 1, 1))(conv3)
    conv3 = BatchNormalization(axis = -1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = ZeroPadding3D((1, 1, 1))(pool3)
    conv4 = Conv3D(init_filter*8, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis = -1)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = ZeroPadding3D((1, 1, 1))(conv4)
    conv4 = Conv3D(init_filter*8, (3, 3, 3), strides=(1, 1, 1))(conv4)
    conv4 = BatchNormalization(axis = -1)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = ZeroPadding3D((1, 1, 1))(pool4)
    conv5 = Conv3D(init_filter*16, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5 = BatchNormalization(axis = -1)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = ZeroPadding3D((1, 1, 1))(conv5)
    conv5 = Conv3D(init_filter*16, (3, 3, 3), strides=(1, 1, 1))(conv5)
    conv5 = BatchNormalization(axis = -1)(conv5)
    conv5 = Activation('relu')(conv5)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=-1)
    conv8 = ZeroPadding3D((1, 1, 1))(up8)
    conv8 = Conv3D(init_filter*8, (3, 3, 3), strides=(1, 1, 1))(conv8)
    conv8 = BatchNormalization(axis = -1)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = ZeroPadding3D((1, 1, 1))(conv8)
    conv8 = Conv3D(init_filter*8, (3, 3, 3), strides=(1, 1, 1))(conv8)
    conv8 = BatchNormalization(axis = -1)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv3], axis=-1)
    conv9 = ZeroPadding3D((1, 1, 1))(up9)
    conv9 = Conv3D(init_filter*4, (3, 3, 3), strides=(1, 1, 1))(conv9)
    conv9 = BatchNormalization(axis = -1)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = ZeroPadding3D((1, 1, 1))(conv9)
    conv9 = Conv3D(init_filter*4, (3, 3, 3), strides=(1, 1, 1))(conv9)
    conv9 = BatchNormalization(axis = -1)(conv9)
    conv9 = Activation('relu')(conv9)

    up10 = concatenate([UpSampling3D(size=(2, 2, 2))(conv9), conv2], axis=-1)
    conv10 = ZeroPadding3D((1, 1, 1))(up10)
    conv10 = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv10)
    conv10 = BatchNormalization(axis = -1)(conv10)
    conv10 = Activation('relu')(conv10)
    conv10 = ZeroPadding3D((1, 1, 1))(conv10)
    conv10 = Conv3D(init_filter*2, (3, 3, 3), strides=(1, 1, 1))(conv10)
    conv10 = BatchNormalization(axis = -1)(conv10)
    conv10 = Activation('relu')(conv10)

    up11 = concatenate([UpSampling3D(size=(2, 2, 2))(conv10), conv1], axis=-1)
    conv11 = ZeroPadding3D((1, 1, 1))(up11)
    conv11 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv11)
    conv11 = BatchNormalization(axis = -1)(conv11)
    conv11 = Activation('relu')(conv11)
    conv11 = ZeroPadding3D((1, 1, 1))(conv11)
    conv11 = Conv3D(init_filter, (3, 3, 3), strides=(1, 1, 1))(conv11)
    conv11 = BatchNormalization(axis = -1)(conv11)
    conv11 = Activation('relu')(conv11)

    conv12 = Conv3D(num_labels, (1, 1, 1), activation='sigmoid')(conv11)
    model = Model(inputs=inputs, outputs=conv12)
    return model