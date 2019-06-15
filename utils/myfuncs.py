import random
from scipy import ndimage
from skimage import morphology
import cv2
from skimage import measure
import os
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import shift
import tensorflow as tf
from skimage import io
from skimage import transform as transf

def tf_get_batch_size(input_var):
    return tf.shape(input_var)[0]

#Fully connected layer
def tf_dense(input_, output_size, stddev=0.02, bias_start=0.0,
             is_training=False, reuse=False, name=None, activation=None, bn=False):
    shape = input_.get_shape().as_list()
    scope = name
    with tf.variable_scope(scope or "Dense", reuse=reuse):
        matrix = tf.get_variable("matrix", [shape[1], output_size], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
        result = tf.matmul(input_, matrix) + bias
        if bn:
            result =  tf.layers.batch_normalization(result, training=is_training)
        if activation is not None:
            result = activation(result)
    return result

#3D Convolution Layer: 3Dconv + add_bias + batch_norm + non-linearity
def tf_conv3d(input_, output_dim, is_training=False,
              k_d=5, k_h=5, k_w=5,  # kernel
              d_d=1, d_h=1, d_w=1,  # strides
              stddev=0.02, name="conv3d",
              reuse=False, activation=None, padding='SAME', bn=True):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #batch norm and non-linearity
        if bn:
            conv = tf.layers.batch_normalization(conv, training=is_training)
        if activation is not None:
            conv = activation(conv)
        return conv

# 3D Transposed Convolution Layer: conv3d_trans + add_bias + batch_norm + non-linearity
def tf_deconv3d(input_, output_shape, is_training=False,
             k_d=5, k_h=5, k_w=5,
             d_d=1, d_h=1, d_w=1,
             bn=False,
             name="deconv3d", padding='SAME', reuse=False, activation=None):
    with tf.variable_scope(name, reuse=reuse):
        batch_size = tf.shape(input_)[0]
        up_filt_shape = [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]]
        up_kernel = upsampling_filter(up_filt_shape)
        deconv = tf.nn.conv3d_transpose(input_, up_kernel, [batch_size, ] + output_shape,
                                        strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, [batch_size, ] + output_shape)

        #batch norm and non-linearity
        if bn:
            deconv = tf.layers.batch_normalization(deconv, training=is_training)
        if activation is not None:
            deconv = activation(deconv)
        return deconv

def upsampling_filter(filter_shape):
    """Bilinear upsampling filter."""
    size = filter_shape[0:3]
    factor = (np.array(size) + 1)

    center = np.zeros_like(factor, np.float)

    for i in range(len(factor)):
        if size[i] % 2 == 1:
            center[i] = factor[i] - 1
        else:
            center[i] = factor[i] - 0.5

    og = np.ogrid[:size[0], :size[1], :size[2]]

    x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
    y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
    z_filt = (1 - abs(og[2] - center[2]) / np.float(factor[2]))

    filt = x_filt * y_filt * z_filt

    weights = np.zeros(filter_shape)
    for i in range(np.min(filter_shape[3:5])):
        weights[:, :, :, i, i] = filt

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    return tf.get_variable(name="upsampling_filter", initializer=init,
                            shape=weights.shape, trainable=True)

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    # """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    # of two batch of data, usually be used for binary image segmentation
    # i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    # Parameters
    # -----------
    # output : Tensor
    #     A distribution with shape: [batch_size, ....], (any dimensions).
    # target : Tensor
    #     The target distribution, format the same with `output`.
    # loss_type : str
    #     ``jaccard`` or ``sorensen``, default is ``jaccard``.
    # axis : tuple of int
    #     All dimensions are reduced, default ``[1,2,3]``.
    # smooth : float
    #     This small value will be added to the numerator and denominator.
    #         - If both output and target are empty, it makes sure dice is 1.
    #         - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    # Examples
    # ---------
    # >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    # References
    # -----------
    # - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__
    # """

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice

def dice_coe_mean(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    # """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    # of two batch of data, usually be used for binary image segmentation
    # i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    # Parameters
    # -----------
    # output : Tensor
    #     A distribution with shape: [batch_size, ....], (any dimensions).
    # target : Tensor
    #     The target distribution, format the same with `output`.
    # loss_type : str
    #     ``jaccard`` or ``sorensen``, default is ``jaccard``.
    # axis : tuple of int
    #     All dimensions are reduced, default ``[1,2,3]``.
    # smooth : float
    #     This small value will be added to the numerator and denominator.
    #         - If both output and target are empty, it makes sure dice is 1.
    #         - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    # Examples
    # ---------
    # >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    # References
    # -----------
    # - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__
    # """

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    ## dice = tf.reduce_mean(dice, axis=axis)
    return dice

###########################################
"""      Rparameterization Tricks       """
###########################################
# def epsilon(_shape, _stddev=1.):
#     return tf.truncated_normal(_shape, mean=0, stddev=_stddev)

def sampler(mu, sigma):
    """
    mu,sigma : (BATCH_SIZE, z_size)
    """
    # return mu + sigma * epsilon(tf.shape(mu))
    eps = tf.random_normal([tf.shape(mu)[0], tf.shape(mu)[1]], 0.0, 1.0, dtype=tf.float32)
    # print("eps:", eps.get_shape().as_list())
    return tf.add(mu, tf.multiply(sigma, eps))

    # return mu + sigma*self.epsilon( tf.shape(mu)[0], tf.shape(mu)[1] )

def vae_sampler(scope, x, size, activation=None, is_training = False):
    # for LVAE
    eps = 1e-8  # epsilon for numerical stability
    with tf.variable_scope(scope, reuse=False):
        mu = tf_dense(x, size, name=scope+'_vae_mu', bn=False, is_training=is_training, activation=None)
        print("mu :", mu.get_shape().as_list())
        logsigma = tf_dense(x, size, name=scope+'_vae_sigma', bn=False, is_training=is_training, activation=tf.nn.softplus)
        logsigma = tf.clip_by_value(logsigma, eps, 5)
        sigma = tf.exp(logsigma)
        print("sigma :", sigma.get_shape().as_list())
    return sampler(mu, sigma), mu, sigma

def vae_sampler2(scope, x, size, activation=None, is_training = False):
    # for LVAE
    eps = 1e-8  # epsilon for numerical stability
    with tf.variable_scope(scope, reuse=False):
        mu = tf_dense(x, size, name=scope+'_vae_mu', bn=False, is_training=is_training, activation=None)
        print("mu :", mu.get_shape().as_list())
        logsigma = tf_dense(x, size, name=scope+'_vae_sigma', bn=False, is_training=is_training, activation=None)
        sigma = tf.exp(0.5 * logsigma)
        print("sigma :", sigma.get_shape().as_list())
    return sampler(mu, sigma), mu, sigma

def precision_weighted(musigma1, musigma2):
    eps = 1e-8  # epsilon for numerical stability
    mu1, sigma1 = musigma1
    mu2, sigma2 = musigma2
    sigma1__2 = 1 / tf.square(sigma1)
    sigma2__2 = 1 / tf.square(sigma2)
    mu = (mu1 * sigma1__2 + mu2 * sigma2__2) / (sigma1__2 + sigma2__2)
    sigma = 1 / (sigma1__2 + sigma2__2)
    logsigma = tf.log(sigma + eps)
    return (mu, logsigma, sigma)

def precision_weighted_sampler(scope, musigma1, musigma2, is_training = False):
    # assume input Tensors are (BATCH_SIZE, dime)
    mu1, sigma1 = musigma1
    mu2, sigma2 = musigma2
    size_1 = mu1.get_shape().as_list()[1]
    size_2 = mu2.get_shape().as_list()[1]

    if size_1 > size_2:
        print('convert 1d to 1d:', size_2, '->', size_1)
        with tf.variable_scope(scope, reuse=False):
            mu2 = tf_dense(mu2, size_1, name=scope+'_lvae_mu', bn=False, is_training=is_training, activation=None)
            sigma2 = tf_dense(sigma2, size_1, name=scope+'_lvae_logsigma', bn=False, is_training=is_training, activation=None)
            musigma2 = (mu2, sigma2)
    elif size_1 < size_2:
        raise ValueError("musigma1 must be equal or bigger than musigma2.")
    else:
        # not need to convert
        pass

    mu, logsigma, sigma = precision_weighted(musigma1, musigma2)
    # return (mu + sigma * epsilon(tf.shape(mu)), mu, logsigma)

    eps = tf.random_normal([tf.shape(mu)[0], tf.shape(mu)[1]], 0.0, 1.0, dtype=tf.float32)
    return (tf.add(mu, tf.multiply(sigma, eps)), mu, sigma)


def data_augmenter(image2):
    """
        Online data augmentation
        Perform affine transformation on image and label,

        image: XYZC
    """
    for ci in range(image2.shape[0]):
        # Create Affine transform
        shear_val = np.random.normal() * 0.05
        afine_tf = transf.AffineTransform(shear=-shear_val)
        shearme = random.sample(range(1, 4), 1)

        if(shearme[0]==1):
            # Apply transform to image data
            for z in range(image2.shape[2]):
                image2[ci, :, :, z, 0] = transf.warp(image2[ci,:, :, z, 0], inverse_map=afine_tf)
                image2[ci, :, :, z, 1] = transf.warp(image2[ci,:, :, z, 1], inverse_map=afine_tf)

        if(shearme[0]==2):
            for y in range(image2.shape[2]):
                image2[ci,:, y, :, 0] = transf.warp(image2[ci,:, y, :, 0], inverse_map=afine_tf)
                image2[ci,:, y, :, 1] = transf.warp(image2[ci,:, y, :, 1], inverse_map=afine_tf)

        if(shearme[0] == 3):
            for x in range(image2.shape[2]):
                image2[ci,x, :, :, 0] = transf.warp(image2[ci,x, :, :, 0], inverse_map=afine_tf)
                image2[ci,x, :, :, 1] = transf.warp(image2[ci,x, :, :, 1], inverse_map=afine_tf)

        image2[image2 > 0.5] = 1
        image2[image2 <= 0.5] = 0

        # Generate random parameters using the Gaussian distribution
        rotate_val = np.random.normal()*9
        rotate_val2 = rotate_val3 = np.random.normal()*6
        scale_val = np.clip(((0.01*np.random.normal())+1), 0.95, 1.05)
        row, col = image2.shape[1], image2.shape[2]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, scale_val)

        for z in range(image2.shape[3]):
            image2[ci,:, :, z, 0] = ndimage.interpolation.affine_transform(image2[ci,:, :, z, 0],
                                                                            M[:, :2], M[:, 2],
                                                                            order=0)
            image2[ci,:, :, z, 1] = ndimage.interpolation.affine_transform(image2[ci,:, :, z, 1],
                                                                            M[:, :2], M[:, 2],
                                                                            order=0)

        row, col = image2.shape[1], image2.shape[3]

        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val2, 1)
        for y in range(image2.shape[2]):
            image2[ci,:, y, :, 0] = ndimage.interpolation.affine_transform(image2[ci,:, y, :, 0],
                                                                            M[:, :2], M[:, 2],
                                                                            order=0)
            image2[ci,:, y, :, 1] = ndimage.interpolation.affine_transform(image2[ci,:, y, :, 1],
                                                                        M[:, :2], M[:, 2],
                                                                        order=0)
        row, col = image2.shape[2], image2.shape[3]

        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val3, 1)
        for x in range(image2.shape[1]):
                image2[ci,x, :, :, 0] = ndimage.interpolation.affine_transform(image2[ci,x, :, :, 0],
                                                                            M[:, :2], M[:, 2],
                                                                            order=0)
                image2[ci,x, :, :, 1] = ndimage.interpolation.affine_transform(image2[ci,x, :, :, 1],
                                                                            M[:, :2], M[:, 2],
                                                                            order=0)

        if(np.random.normal()<0):
            image2[ci,:, :, :, 0] = ndimage.binary_closing(image2[ci,:, :, :, 0], structure=np.ones((2, 2, 2))).astype(image2.dtype)
            image2[ci,:, :, :, 1] = ndimage.binary_closing(image2[ci,:, :, :, 1], structure=np.ones((2, 2, 2))).astype(image2.dtype)

    return image2
