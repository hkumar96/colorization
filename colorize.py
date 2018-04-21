from __future__ import print_function

__author__ = "Hemant Kumar, Shubham Jain"
'''
Tensorflow implemenation of Image colorization using Adversarial loss
'''
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_LaMemDataset as lamem
# import read_FlowersDataset as flowers
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/LaMem/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.9", "Beta 1 value to use in Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
IMAGE_SIZE = 128
ADVERSARIAL_LOSS_WEIGHT = 1e-3
vec_size = 256


def vgg_net(weights, image):
    layers = (
        # 'conv1_1', 'relu1_1',
        'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i + 2][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def generator(images, train_phase):
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    weights = np.squeeze(model_data['layers'])
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE) as scope:
        W0 = utils.weight_variable([3, 3, 1, 64], name="W0")
        b0 = utils.bias_variable([64], name="b0")
        conv0 = utils.conv2d_basic(images, W0, b0)
        hrelu0 = tf.nn.relu(conv0, name="relu")

        image_net = vgg_net(weights, hrelu0)
        vgg_final_layer = image_net["relu5_3"]

        pool5 = utils.max_pool_2x2(vgg_final_layer)

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, pool5.get_shape()[3].value], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(pool5, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        # shape = tf.shape(images)
        # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], 2])
        # W_t3 = utils.weight_variable([16, 16, 2, deconv_shape2[3].value], name="W_t3")
        # b_t3 = utils.bias_variable([2], name="b_t3")
        # pred = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        shape = tf.shape(images)
        deconv_shape3 = tf.stack([shape[0], 32 , 32 , 256])
        W_t3 = utils.weight_variable([4, 4, 256 , deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([256], name="b_t3")
        pred1 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=2)

        shape = tf.shape(images)
        deconv_shape4 = tf.stack([shape[0], 128 , 128, 256])
        W_t4 = utils.weight_variable([8, 8, 256, 256], name="W_t4")
        b_t4 = utils.bias_variable([256], name="b_t4")
        pred = utils.conv2d_transpose_strided(pred1, W_t4, b_t4, output_shape=deconv_shape4, stride=4)

    # return tf.concat(axis=3, values=[images, pred], name="pred_image")
    return pred 

def train(loss, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def convert_image(prob_image,l_image):
    # T = 0.0003
    # shape = prob_image.shape
    # print(prob_image[1,1,:],shape)
    # exp_probim = np.exp(np.log(prob_image)/T)
    # norm_prob = exp_probim/np.sum(exp_probim,axis = 2).reshape((128,128,1))
    # a_centre = np.repeat(np.arange(1,17)*16-8,16)
    # b_centre = np.tile(np.arange(1,17)*16-8,16)
    
    # a_centre = np.reshape(a_centre,(1,1,256))
    # a_centre = np.reshape(b_centre,(1,1,256))
    
    # a = np.sum(np.multiply(norm_prob,a_centre),axis = 2)
    # b = np.sum(np.multiply(norm_prob,b_centre),axis = 2)

    max_prob = np.argmax(prob_image,axis = 2)
    [a,b] = np.unravel_index(max_prob,(16,16))    
    a = a*16 + 8
    b = b*16 + 8

    a = a.reshape((128,128,1)).astype(np.uint8)
    b = b.reshape((128,128,1)).astype(np.uint8)
    # print(a,b)
    # max_ab = np.argmax(prob_image,axis=2)
    return np.concatenate((l_image,a,b),axis = 2)



def main(argv=None):
    print("Setting up network...")
    train_phase = tf.placeholder(tf.bool, name="train_phase")
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1],
                            name='inputs')
    # lab_images = tf.placeholder(tf.float32, shape=[None, None, None, 3],
    #                             name="LAB_image")
    labels = tf.placeholder(tf.float32, shape=[None, None, None, vec_size],
                            name="labels")

    logits = generator(inputs, train_phase)
    rebal_weights = tf.placeholder(tf.float32, shape=[None, None, 128],
                                   name="REBAL_WGTS")
    # gen_loss_mse = tf.reduce_mean(2 * tf.nn.l2_loss(pred_image - lab_images)) / (IMAGE_SIZE * IMAGE_SIZE * 100 * 100)
    reshaped_logits = tf.reshape(logits, [-1, vec_size])
    reshaped_labels = tf.reshape(labels, [-1, vec_size])

    pre_loss = tf.nn.softmax_cross_entropy_with_logits(logits = 
        reshaped_logits, labels = reshaped_labels)

    #Loss without rebalancing 
    # final_loss = tf.reduce_sum(pre_loss)/FLAGS.batch_size

    final_loss = tf.reduce_sum(tf.multiply(pre_loss, tf.reshape(rebal_weights, [-1,])))/FLAGS.batch_size

    tf.summary.scalar("Generator_loss_MSE", final_loss)

    train_variables = tf.trainable_variables()
    for v in train_variables:
        utils.add_to_regularization_and_summary(var=v)

    train_op = train(final_loss, train_variables)

    print("Reading image dataset...")
    # train_images, testing_images, validation_images = flowers.read_dataset(FLAGS.data_dir)
    train_images = lamem.read_dataset(FLAGS.data_dir)
    image_options = {"resize": True, "resize_size": IMAGE_SIZE, "color": "LAB", "Q":16}
    batch_reader = dataset.BatchDatset(train_images, image_options)

    print("Setting up session")
    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == 'train':
        for itr in xrange(MAX_ITERATION):
            l_image, prob_images,rweights_batch = batch_reader.next_batch(FLAGS.batch_size)
            feed_dict = {inputs: l_image, labels: prob_images, rebal_weights:rweights_batch, train_phase: True}

            if itr % 10 == 0:
                mse, summary_str = sess.run([final_loss, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, itr)
                print("Step: %d, MSE: %g" % (itr, mse))

            if itr % 50 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
                pred = sess.run(logits, feed_dict=feed_dict)  
                idx = np.random.randint(0, FLAGS.batch_size)
                e_x = np.exp(pred[idx] - np.max(pred[idx]))
                pred_prob = e_x / e_x.sum(axis=2).reshape(128,128,1) 
                save_dir = os.path.join(FLAGS.logs_dir, "image_checkpoints")
                # utils.save_image(l_image[idx], save_dir, "gt" + str(itr),False) 
                utils.save_image(convert_image(pred_prob.astype(np.float64),l_image[idx]), save_dir, "pred" + str(itr))
                print("%s --> Model saved" % datetime.datetime.now())

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10000 == 0:
                FLAGS.learning_rate /= 2
    elif FLAGS.mode == "test":
        count = 130
        l_image, prob_images,rweights_batch,colorImage = batch_reader.get_random_batch(count)
        feed_dict = {inputs: l_image, labels: prob_images, rebal_weights:rweights_batch, train_phase: False}
        save_dir = os.path.join(FLAGS.logs_dir, "image_pred")
        pred = sess.run(logits, feed_dict=feed_dict)
        np.save('predicted',pred)
        for itr in range(count):
            utils.save_image(colorImage[itr], save_dir, "gt" + str(itr))
            e_x = np.exp(pred[itr] - np.max(pred[itr]))
            pred_prob = e_x / e_x.sum(axis=2).reshape(128,128,1) 
            utils.save_image(convert_image(pred_prob.astype(np.float64),l_image[itr]), save_dir, "pred" + str(itr))
        print("--- Images saved on test run ---")

if __name__ == "__main__":
    tf.app.run()
