from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

#BASE_X = 7
#BASE_Y = 7
IMAGE_SIZE = 28,28,1
CONVOLUTIONS = [-32, -64, 128]
HIDDEN_SIZE = 1000
NUM_CLASSES = 10
LEARNING_RATE = 2e-5
MOMENTUM = .5
BATCH_SIZE = 16
FILEPATH = '/Data/FashionMNIST/'
TRAIN_INPUT = FILEPATH + 'train-images-idx3-ubyte'
TRAIN_LABEL = FILEPATH + 'train-labels-idx1-ubyte'
PERM_MODEL_FILEPATH = '/Models/FashionAttention/SimpleModel.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/FashionAttention/Summaries/'

RESTORE = False
WHEN_DISP = 100
NUM_OUTPUTS = 20
WHEN_SAVE = 2000
MAX_OUTPUTS = 16
ITERATIONS = 1000000

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [1])
    return  label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 1, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def create_discriminator(x_image, reuse = False):
    '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''
    with tf.variable_scope("d_discriminator") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        xs, ys = IMAGE_SIZE[0],IMAGE_SIZE[1]
        inputs = InputLayer(x_image, name='d_inputs')

        convVals = inputs
        res = inputs
        for i,v in enumerate(CONVOLUTIONS):
            '''Similarly tile for constant reference to class'''
            convVals = Conv2d(convVals,abs(v), (1, 1), act=tf.nn.leaky_relu,strides =(1,1),name='d_conv_0_%i'%(i))
            batch = BatchNormLayer(convVals, act=tf.nn.leaky_relu, is_train=True,name='d_bn_1_%i'%(i))
            conv = Conv2d(batch,abs(v), (3, 3), strides =(1,1),name='d_conv_1_%i'%(i))
            batch = BatchNormLayer(conv,act=tf.nn.leaky_relu, is_train=True,name='d_bn_2_%i'%(i))
            conv = Conv2d(batch,abs(v), (3, 3), strides =(1,1),name='d_conv_2_%i'%(i))
            convVals = InputLayer(convVals.outputs + conv.outputs, name='d_res_sum_%i'%(i))
            if v < 0:
                v *= -1
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.leaky_relu,strides =(2,2),name='d_conv_3_%i'%(i))
            else:
                convVals = Conv2d(convVals,v, (3, 3), act=tf.nn.leaky_relu,strides =(1,1),name='d_conv_3_%i'%(i))

            #add necessary convolutional layers
                # fully connecter layer
        flat3 = FlattenLayer(convVals, name = 'd_flatten')
        #inputClass =InputLayer(classes, name='d_class_inputs')
        #concat = ConcatLayer([flat3, inputClass], 1, name ='d_concat_layer')
        hid3 = DenseLayer(flat3, HIDDEN_SIZE,act = tf.nn.relu, name = 'd_fcl')
        y_conv = DenseLayer(hid3, NUM_CLASSES,  name = 'd_output').outputs
        return y_conv

def build_model(x, og_classes):
    # classes = tf.squeeze(classes, axis = 1)
    classes = tf.squeeze(og_classes, 1)

    y = tf.one_hot(classes, NUM_CLASSES)
    #fake_classes_one = tf.one_hot(classes, NUM_CLASSES)
    #y = tf.expand_dims(tf.ones_like(classes, dtype=tf.float32),-1)
    y_conv = create_discriminator(x) #real image discrimator
        # print(classes.get_shape())
        # m = classes.get_shape()
    with tf.variable_scope('logistics') as scope:
        real_input_summary = tf.summary.image("real_inputs", x,max_outputs = NUM_OUTPUTS)#show real image
        t_vars = tf.trainable_variables()
    # print(t_vars)

        d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
        print(d_vars)
        #en_vars = [var for var in t_vars if 'gen_' in var.name] #find trainable discriminator variable
        # print(gen_vars)
        print(y.get_shape())
        print(y_conv.get_shape())
        d_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv))
        d_cross_entropy_summary = tf.summary.scalar('d_loss',d_cross_entropy)
        final_true = tf.argmax(tf.nn.softmax(y_conv,1), 1)
        accuracy_real = tf.reduce_mean(tf.cast(tf.equal(final_true, tf.cast(classes, tf.int64)), tf.float32))#determine various accuracies
        accuracy_summary_real = tf.summary.scalar('accuracy_real',accuracy_real)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=MOMENTUM).minimize(d_cross_entropy, var_list = d_vars)
    scalar_summary = tf.summary.merge([d_cross_entropy_summary, accuracy_summary_real])
    return scalar_summary, real_input_summary, train_step


if __name__ == "__main__":
    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    sess.run([train_iterator.initializer])
    scalar_summary, image_summary, train_step = build_model(train_input, train_label)

    ##########################################################################
    # Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())
    t_vars = tf.trainable_variables()
    #_ =testerModel.init_model(sess)
    
    d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
    #print(d_vars)
    #gen_vars = [var for var in t_vars if 'gen_' in var.name]
    saver_perm = tf.train.Saver(var_list = d_vars)
    
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    #trains = [train_step,train_step,gen_train_step]
    train = train_step
    for i in range(ITERATIONS):
        #train = trains[i%3]
        if not i % WHEN_DISP:
            input_summary_ex, image_summary_ex,_= sess.run([scalar_summary, image_summary, train])
            train_writer.add_summary(image_summary_ex, i)
        else:
            input_summary_ex, _= sess.run([scalar_summary, train])
        train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)
        # if not i % EST_ITERATIONS:
        #     print('Epoch' + str(i / EST_ITERATIONS))
