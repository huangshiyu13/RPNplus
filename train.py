import inspect
import os
import time
import  sys
import numpy as np
import tensorflow as tf
import shutil
import data_engine

VGG_MEAN = [103.939, 116.779, 123.68]

image_height = 720
image_width = 960
feature_height = int(np.ceil(image_height / 16.))
feature_width = int(np.ceil(image_width / 16.))


class RPN:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, 'vgg16.npy')
            vgg16_npy_path = path
            print path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print('npy file loaded')

    def build(self, rgb, label, label_weight, bbox_target, bbox_loss_weight, learning_rate):
       
        start_time = time.time()
        print('build model started')

        # Convert RGB to BGR
        red, green, blue = tf.split(rgb, 3, 3)
        assert red.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert green.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert blue.get_shape().as_list()[1:] == [image_height, image_width, 1]
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ],3)
        assert bgr.get_shape().as_list()[1:] == [image_height, image_width, 3]
        # Conv layer 1
        self.conv1_1 = self.conv_layer_const(bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer_const(self.conv1_1, 'conv1_2')
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        # Conv layer 2
        self.conv2_1 = self.conv_layer_const(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer_const(self.conv2_1, 'conv2_2')
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Conv layer 3
        self.conv3_1, conv3_1_wd = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2, conv3_2_wd = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3, conv3_3_wd = self.conv_layer(self.conv3_2, 'conv3_3')
        self.weight_dacay = conv3_1_wd + conv3_2_wd + conv3_3_wd
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Conv layer 4
        self.conv4_1, conv4_1_wd = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2, conv4_2_wd = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3, conv4_3_wd = self.conv_layer(self.conv4_2, 'conv4_3')
        self.weight_dacay += conv4_1_wd + conv4_2_wd + conv4_3_wd
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Conv layer 5
        self.conv5_1, conv5_1_wd = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2, conv5_2_wd = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3, conv5_3_wd = self.conv_layer(self.conv5_2, 'conv5_3')
        self.weight_dacay += conv5_1_wd + conv5_2_wd + conv5_3_wd

        # RPN_TEST_6(>=7)
        normalization_factor = tf.sqrt(tf.reduce_mean(tf.square(self.conv5_3)))
        self.gamma3 = tf.Variable(np.sqrt(2), dtype=tf.float32, name='gamma3')
        self.gamma4 = tf.Variable(1.0, dtype=tf.float32, name='gamma4')
        # Pooling to the same size
        self.pool3_p = tf.nn.max_pool(self.pool3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      name='pool3_proposal')
        # L2 Normalization
        self.pool3_p = self.pool3_p / (
            tf.sqrt(tf.reduce_mean(tf.square(self.pool3_p))) / normalization_factor) * self.gamma3
        self.pool4_p = self.pool4 / (
            tf.sqrt(tf.reduce_mean(tf.square(self.pool4))) / normalization_factor) * self.gamma4
        # Proposal Convolution
        self.conv_proposal_3, conv_proposal_3_wd = self.conv_layer_new(self.pool3_p, 'conv_proposal_3',
                                                                       kernel_size=[5, 2], out_channel=256, stddev=0.01)
        self.relu_proposal_3 = tf.nn.relu(self.conv_proposal_3)
        self.conv_proposal_4, conv_proposal_4_wd = self.conv_layer_new(self.pool4_p, 'conv_proposal_4',
                                                                       kernel_size=[5, 2], out_channel=512, stddev=0.01)
        self.relu_proposal_4 = tf.nn.relu(self.conv_proposal_4)
        self.conv_proposal_5, conv_proposal_5_wd = self.conv_layer_new(self.conv5_3, 'conv_proposal_5',
                                                                       kernel_size=[5, 2], out_channel=512, stddev=0.01)
        self.relu_proposal_5 = tf.nn.relu(self.conv_proposal_5)
        self.weight_dacay += conv_proposal_3_wd + conv_proposal_4_wd + conv_proposal_5_wd
        # Concatrate
        self.relu_proposal_all = tf.concat( [self.relu_proposal_3, self.relu_proposal_4, self.relu_proposal_5],3)
        # RPN_TEST_6(>=7)

        self.conv_cls_score, conv_cls_wd = self.conv_layer_new(self.relu_proposal_all, 'conv_cls_score',
                                                               kernel_size=[1, 1], out_channel=18, stddev=0.01)
        self.conv_bbox_pred, conv_bbox_wd = self.conv_layer_new(self.relu_proposal_all, 'conv_bbox_pred',
                                                                kernel_size=[1, 1], out_channel=36, stddev=0.01)
        self.weight_dacay += conv_cls_wd + conv_bbox_wd

        assert self.conv_cls_score.get_shape().as_list()[1:] == [feature_height, feature_width, 18]
        assert self.conv_bbox_pred.get_shape().as_list()[1:] == [feature_height, feature_width, 36]

        self.cls_score = tf.reshape(self.conv_cls_score, [-1, 2])
        self.bbox_pred = tf.reshape(self.conv_bbox_pred, [-1, 4])

        self.prob = tf.nn.softmax(self.cls_score, name="prob")
        self.cross_entropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=label,
                                                    logits=self.cls_score) * label_weight) / tf.reduce_sum(label_weight)

        bbox_error = tf.abs(self.bbox_pred - bbox_target)
        bbox_loss = 0.5 * bbox_error * bbox_error * tf.cast(bbox_error < 1, tf.float32) + (bbox_error - 0.5) * tf.cast(
            bbox_error >= 1, tf.float32)
        self.bb_loss = tf.reduce_sum(
            tf.reduce_sum(bbox_loss, reduction_indices=[1]) * bbox_loss_weight) / tf.reduce_sum(bbox_loss_weight)

        self.loss = self.cross_entropy + 0.0005 * self.weight_dacay + 0.5 * self.bb_loss

        self.train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)

        self.data_dict = None
        print('build model finished: %ds' % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            weight_dacay = tf.nn.l2_loss(filt, name='weight_dacay')
            return relu, weight_dacay

    def conv_layer_const(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter_const(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias_const(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv_layer_new(self, bottom, name, kernel_size=[3, 3], out_channel=512, stddev=0.01):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()[-1]
            filt = tf.Variable(
                tf.random_normal([kernel_size[0], kernel_size[1], shape, out_channel], mean=0.0, stddev=stddev),
                name='filter')
            conv_biases = tf.Variable(tf.zeros([out_channel]), name='biases')

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            weight_dacay = tf.nn.l2_loss(filt, name='weight_dacay')
            return bias, weight_dacay

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name='filter')

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name='biases')

    def get_conv_filter_const(self, name):
        return tf.constant(self.data_dict[name][0], name='filter')

    def get_bias_const(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')

    def save(self, save_dir, step=None):
        params = {}
        for var in tf.trainable_variables():
            param_name = var.name.split('/')
            if param_name[1] in params.keys():
                params[param_name[1]].append(sess.run(var))
            else:
                params[param_name[1]] = [sess.run(var)]

        if step == None:
            step = 100000
        np.save(save_dir + 'params_' + str(step) + '.npy', params)


def checkFile(fileName):
    if os.path.isfile(fileName):
        return True
    else:
        print fileName, 'is not found!'
        exit()


def checkDir(fileName, creat=False):
    if os.path.isdir(fileName):
        if creat:
            shutil.rmtree(fileName)
            os.mkdir(fileName)
    else:
        if creat:
            os.mkdir(fileName)
        else:
            print fileName, 'is not found!'
            exit()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'please input GPU index'
        exit()

    gpuNow = '/gpu:'+sys.argv[1]
    print_time = 100
    step = 10000
    batch_size = 256
    saveTime = 2000

    modelSaveDir = './models/'
    vggModelPath = './models/vgg16.npy'

    imageLoadDir = './yourImagePath/'
    anoLoadDir = './yourAnnotationPath/'

    checkDir(modelSaveDir, False)
    checkDir(imageLoadDir, False)
    checkDir(anoLoadDir, False)

    with tf.device(gpuNow):
        sess = tf.Session() 
        image = tf.placeholder(tf.float32, [1, image_height, image_width, 3])
        label = tf.placeholder(tf.float32, [None, 2])
        label_weight = tf.placeholder(tf.float32, [None])
        bbox_target = tf.placeholder(tf.float32, [None, 4])
        bbox_loss_weight = tf.placeholder(tf.float32, [None])
        learning_rate = tf.placeholder(tf.float32)

        cnn = RPN(vggModelPath)
        with tf.name_scope('content_rpn'):
            cnn.build(image, label, label_weight, bbox_target, bbox_loss_weight, learning_rate)

        sess.run(tf.initialize_all_variables())
        for var in tf.trainable_variables():
            print var.name, var.get_shape().as_list(), sess.run(tf.nn.l2_loss(var))

        
        cnnData = data_engine.CNNData(batch_size, imageLoadDir, anoLoadDir)
        print 'Training Begin'
    
        train_loss = []
        train_cross_entropy = []
        train_bbox_loss = []
        start_time = time.time()

        for i in xrange(1, step + 1):
            batch = cnnData.prepare_data()
            if i <= 7000:
                l_r = 0.001
            else:
                if i <= 9000:
                    l_r = 0.0001
                else:
                    l_r = 0.00001
            (_, train_loss_iter, train_cross_entropy_iter, train_bbox_loss_iter, cls, bbox) = sess.run(
                [cnn.train_step, cnn.loss, cnn.cross_entropy, cnn.bb_loss, cnn.cls_score, cnn.bbox_pred],
                feed_dict={image: batch[0], label: batch[1], label_weight: batch[2], bbox_target: batch[3],
                           bbox_loss_weight: batch[4], learning_rate: l_r})

            train_loss.append(train_loss_iter)
          

            if i % print_time == 0:
              
                print ' step :', i, 'time :', time.time() - start_time, 'loss :', np.mean(
                    train_loss), 'l_r :', l_r
                train_loss = []

            if i% saveTime == 0:
                cnn.save(modelSaveDir, i)
